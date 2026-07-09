import asyncio
import logging
import time
from typing import Dict, List, Optional

import httpx

from .config import Config
from .datamodel import ApiResult
from .parsing import classify_api_error, parse_context_overflow, parse_output


class BudgetController:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.window = cfg.context_window

    def set_window(self, window: int):
        if window and window != self.window:
            logging.info(f"Context window set to {window} (was {self.window})")
        self.window = window

    def correct_window(self, server_ctx: int):
        if server_ctx and server_ctx != self.window:
            logging.warning(f"Server reports window {server_ctx}; correcting from {self.window}")
            self.window = server_ctx

    def fit_budget(self, prompt_tokens: int) -> int:
        return self.window - prompt_tokens - self.cfg.context_safety_margin

    def clamp(self, requested: int, prompt_tokens: int) -> int:
        fit = self.fit_budget(prompt_tokens)
        return min(requested, fit)


async def detect_context_limit(client: httpx.AsyncClient, cfg: Config) -> Optional[int]:
    try:
        r = await client.get(f"{cfg.vllm_base_url}/models",
                             headers={"Authorization": f"Bearer {cfg.vllm_api_key}"})
        if r.status_code == 200:
            for m in r.json().get("data", []):
                ml = m.get("max_model_len")
                if ml:
                    return int(ml)
    except Exception:
        pass
    return None


def _server_root(cfg: Config) -> str:
    return cfg.vllm_base_url.rstrip("/").removesuffix("/v1")


def _thinking_off_body(cfg: Config) -> Dict:
    mode = cfg.thinking_off_mode
    if mode == "chat_template":
        return {"chat_template_kwargs": {"enable_thinking": False}}
    if mode == "enable_thinking":
        return {"enable_thinking": False}
    if mode == "reasoning":
        return {"reasoning": {"enabled": False}}
    return {}


async def count_prompt_tokens(client: httpx.AsyncClient, cfg: Config,
                              messages: List[Dict[str, str]]) -> Optional[int]:
    cached = getattr(cfg, "_tokenize_url", None)
    if cached == "":
        return None
    payload = {"model": cfg.model_name, "messages": messages, "add_generation_prompt": True}
    hdrs = {"Authorization": f"Bearer {cfg.vllm_api_key}"}
    base = cfg.vllm_base_url.rstrip("/")
    candidates = [cached] if cached else [f"{_server_root(cfg)}/tokenize", f"{base}/tokenize"]
    for url in candidates:
        try:
            r = await client.post(url, json=payload, headers=hdrs)
            if r.status_code == 200:
                j = r.json()
                cfg._tokenize_url = url
                if j.get("count") is not None:
                    return int(j["count"])
                if "tokens" in j:
                    return len(j["tokens"])
                return None
        except Exception:
            continue
    if not cached:
        cfg._tokenize_url = ""
        logging.warning("Tokenizer endpoint not found at /tokenize or /v1/tokenize; "
                        "falling back to char-based token estimates for this run")
    return None


def _continue_thinking_messages(original, partial):
    user = ("Your previous response was cut off while you were still reasoning. "
            "Continue your analysis, finish your reasoning, and then output the "
            "final header cells in the format: row col | cell text (one per line). "
            "When the list is complete, write DONE on its own line.")
    return original + [{"role": "assistant", "content": partial},
                       {"role": "user", "content": user}]


def _force_answer_messages(original, partial):
    user = ("Stop reasoning now. You have analysed enough. Output ONLY the header "
            "cells immediately, one per line, in the format: row col | cell text. "
            "Do not explain, do not think further. When complete, write DONE.")
    return original + [{"role": "assistant", "content": partial},
                       {"role": "user", "content": user}]


def _extend_list_messages(original, first_response, last_row):
    user = (f"Your previous response was cut off. Continue listing header cells "
            f"starting from row {last_row}. Use the same format: row col | cell text. "
            f"Output ONLY the remaining headers, no repetition.")
    return original + [{"role": "assistant", "content": first_response},
                       {"role": "user", "content": user}]


def _last_row(parsed):
    return max((h.get("row", 0) for h in parsed), default=0)


async def async_api_call(client: httpx.AsyncClient, cfg: Config, budget: BudgetController,
                         messages: List[Dict[str, str]], requested_max_tokens: int,
                         known_prompt_tokens: Optional[int] = None,
                         disable_thinking: bool = False,
                         allow_continuation: bool = True) -> ApiResult:
    url = f"{cfg.vllm_base_url}/chat/completions"

    prompt_toks = known_prompt_tokens
    if prompt_toks is None and cfg.use_tokenizer:
        prompt_toks = await count_prompt_tokens(client, cfg, messages)
    if prompt_toks is None:
        prompt_toks = sum(len(m.get("content", "")) for m in messages) // 3

    fit = budget.fit_budget(prompt_toks)
    if fit <= 0:
        return ApiResult(
            api_success=False, requested_max_tokens=requested_max_tokens,
            effective_max_tokens=0, budget_clamped=True,
            error_type="context_length_exceeded",
            error_message=f"prompt {prompt_toks} toks exceeds usable window {budget.window}")

    effective = budget.clamp(requested_max_tokens, prompt_toks)
    budget_clamped = effective < requested_max_tokens

    payload = {
        "model": cfg.model_name,
        "messages": messages,
        "temperature": cfg.temperature,
        "max_tokens": effective,
    }
    if cfg.seed is not None:
        payload["seed"] = cfg.seed
    if disable_thinking and cfg.disable_thinking_supported:
        payload.update(_thinking_off_body(cfg))
    if cfg.extra_body:
        payload.update(cfg.extra_body)

    hdrs = {"Content-Type": "application/json",
            "Authorization": f"Bearer {cfg.vllm_api_key}"}

    last_error = ""
    for attempt in range(1, cfg.max_retries + 1):
        try:
            t0 = time.time()
            resp = await client.post(url, json=payload, headers=hdrs)
            if resp.status_code == 429 and cfg.honor_retry_after:
                ra = None
                try:
                    ra = resp.headers.get("retry-after")
                except Exception:
                    ra = None
                delay = float(ra) if (ra and str(ra).replace(".", "", 1).isdigit()) \
                    else min(cfg.retry_backoff_base ** attempt, 60.0)
                logging.warning(f"429 rate limited; waiting {delay:.0f}s (attempt {attempt})")
                last_error = "HTTP 429 rate limited"
                await asyncio.sleep(delay)
                continue
            if resp.status_code >= 400:
                try:
                    eb = resp.json()
                    em = eb.get("message") or eb.get("detail") or resp.text[:300]
                    if isinstance(em, dict):
                        em = em.get("message", str(em))
                except Exception:
                    em = resp.text[:300]
                raise RuntimeError(f"HTTP {resp.status_code}: {em}")

            data_r = resp.json()
            duration = time.time() - t0
            msg = data_r["choices"][0]["message"]
            raw = msg.get("content") or ""
            reasoning = msg.get("reasoning_content") or ""

            ok, parsed, pe = parse_output(raw)
            usage = data_r.get("usage") or {}
            comp = usage.get("completion_tokens", 0) or 0
            capped = comp >= effective
            output_complete = (pe == "done_marker")

            result = ApiResult(
                api_success=True, raw_response=raw, reasoning=reasoning,
                parse_success=ok, parsed_headers=parsed,
                parse_error=(pe if pe != "done_marker" else ""),
                output_complete=output_complete, duration_sec=duration,
                retry_attempts=attempt, requested_max_tokens=requested_max_tokens,
                effective_max_tokens=effective, budget_clamped=budget_clamped,
                capped=capped,
                tokens_used={"prompt": usage.get("prompt_tokens"),
                             "completion": comp,
                             "total": (usage.get("prompt_tokens") or 0) + comp})

            if (allow_continuation and cfg.enable_continuation
                    and capped and not output_complete):
                await _run_continuation(client, cfg, budget, messages, requested_max_tokens,
                                        raw, reasoning, parsed, ok, result)
            return result

        except asyncio.CancelledError:
            raise
        except Exception as e:
            last_error = str(e)
            etype = classify_api_error(last_error)
            if etype in ("model_not_found", "insufficient_balance"):
                logging.critical(f"Permanent error ({etype}): {last_error[:200]}")
                return ApiResult(
                    api_success=False, requested_max_tokens=requested_max_tokens,
                    effective_max_tokens=effective, budget_clamped=budget_clamped,
                    retry_attempts=attempt, error_type=etype, error_message=last_error)
            logging.warning(f"Attempt {attempt}/{cfg.max_retries} failed: {last_error[:200]}")
            ptoks, server_ctx = parse_context_overflow(last_error)
            if server_ctx is not None:
                budget.correct_window(server_ctx)
            if ptoks is not None:
                refit = budget.fit_budget(ptoks)
                if refit >= cfg.min_completion_tokens and refit < effective:
                    logging.info(f"Context refit: prompt={ptoks}, max_tokens {effective} -> {refit}")
                    effective = refit
                    payload["max_tokens"] = refit
                    budget_clamped = True
                    continue
                else:
                    break
            if attempt < cfg.max_retries:
                await asyncio.sleep(min(cfg.retry_backoff_base ** attempt, 60.0))

    return ApiResult(
        api_success=False, requested_max_tokens=requested_max_tokens,
        effective_max_tokens=effective, budget_clamped=budget_clamped,
        retry_attempts=cfg.max_retries,
        error_type=classify_api_error(last_error), error_message=last_error)


_HARD_CONTINUATION_CAP = 50


async def _run_continuation(client, cfg, budget, messages, requested_max_tokens,
                            raw, reasoning, parsed, ok, result: ApiResult):
    current = list(parsed)
    last_raw = raw
    last_partial = raw.strip() or reasoning.strip()

    max_rounds = cfg.max_continuation_rounds if cfg.max_continuation_rounds > 0 else _HARD_CONTINUATION_CAP
    max_rounds = min(max_rounds, _HARD_CONTINUATION_CAP)

    rounds = 0
    truncated = True
    completed = False
    while rounds < max_rounds:
        rounds += 1
        if current:
            cont_msgs = _extend_list_messages(messages, last_raw, _last_row(current))
            mode = "extend list"
        else:
            cont_msgs = _continue_thinking_messages(messages, last_partial)
            mode = "continue thinking"
        logging.info(f"Continuation round {rounds} ({mode})")
        r = await async_api_call(client, cfg, budget, cont_msgs,
                                 requested_max_tokens, allow_continuation=False)
        _accumulate(result, r)
        result.continuation_used = True
        result.continuation_tier = rounds
        if not r.api_success:
            logging.info(f"Continuation stopped at round {rounds}: "
                         f"{r.error_type or 'no response'} (budget likely exhausted)")
            break
        if r.parsed_headers:
            seen = {(h["row"], h["col"]) for h in current}
            for h in r.parsed_headers:
                key = (h["row"], h["col"])
                if key not in seen:
                    seen.add(key)
                    current.append(h)
        if r.raw_response:
            last_raw = r.raw_response
        if r.raw_response or r.reasoning:
            last_partial = (last_partial + "\n" + (r.raw_response or r.reasoning)).strip()
        if r.output_complete:
            completed = True
            truncated = False
            break
        if not r.capped:
            truncated = False
            break

    if current:
        current.sort(key=lambda x: (x["row"], x["col"]))
        result.parsed_headers = current
        result.parse_success = True
        if completed:
            result.parse_error = ""
        result.capped = truncated
        return

    result.capped = truncated
    if cfg.force_answer_when_exhausted:
        logging.info("Continuation salvage: forcing an answer (thinking off)")
        rf = await async_api_call(client, cfg, budget,
                                  _force_answer_messages(messages, last_partial),
                                  requested_max_tokens, disable_thinking=True,
                                  allow_continuation=False)
        _accumulate(result, rf)
        result.continuation_used = True
        result.continuation_forced = True
        if rf.api_success and rf.parse_success and rf.parsed_headers:
            result.parsed_headers = sorted(rf.parsed_headers, key=lambda x: (x["row"], x["col"]))
            result.parse_success = True
            result.parse_error = ""
            result.capped = False
        else:
            logging.warning("Continuation exhausted, no parseable answer")


def _accumulate(result: ApiResult, other: ApiResult):
    ot = other.tokens_used or {}
    if result.tokens_used is None:
        result.tokens_used = {"prompt": 0, "completion": 0, "total": 0}
    result.tokens_used["completion"] = (result.tokens_used.get("completion") or 0) + (ot.get("completion") or 0)
    result.tokens_used["total"] = (result.tokens_used.get("total") or 0) + (ot.get("total") or 0)
    result.duration_sec = (result.duration_sec or 0) + (other.duration_sec or 0)
