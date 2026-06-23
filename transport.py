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
        r = await client.get(f"{cfg.vllm_base_url}/models")
        if r.status_code == 200:
            for m in r.json().get("data", []):
                ml = m.get("max_model_len")
                if ml:
                    return int(ml)
    except Exception:
        pass
    return None


async def count_prompt_tokens(client: httpx.AsyncClient, cfg: Config,
                              messages: List[Dict[str, str]]) -> Optional[int]:
    try:
        r = await client.post(
            f"{cfg.vllm_base_url}/tokenize",
            json={"model": cfg.model_name, "messages": messages, "add_generation_prompt": True},
            headers={"Authorization": f"Bearer {cfg.vllm_api_key}"},
        )
        if r.status_code == 200:
            j = r.json()
            if j.get("count") is not None:
                return int(j["count"])
            if "tokens" in j:
                return len(j["tokens"])
    except Exception:
        pass
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
    if prompt_toks is None:
        prompt_toks = await count_prompt_tokens(client, cfg, messages)
    if prompt_toks is None:
        prompt_toks = sum(len(m.get("content", "")) for m in messages) // 3
        logging.warning("Tokenizer endpoint unavailable; using a rough char-based token "
                        "estimate for budget — token-dependent fields may be approximate")

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
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    hdrs = {"Content-Type": "application/json",
            "Authorization": f"Bearer {cfg.vllm_api_key}"}

    last_error = ""
    for attempt in range(1, cfg.max_retries + 1):
        try:
            t0 = time.time()
            resp = await client.post(url, json=payload, headers=hdrs)
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

        except Exception as e:
            last_error = str(e)
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


async def _run_continuation(client, cfg, budget, messages, requested_max_tokens,
                            raw, reasoning, parsed, ok, result: ApiResult):
    if not parsed:
        partial = raw.strip() or reasoning.strip()
        logging.info("Continuation tier 2 (continue thinking)")
        r2 = await async_api_call(client, cfg, budget,
                                  _continue_thinking_messages(messages, partial),
                                  requested_max_tokens, allow_continuation=False)
        _accumulate(result, r2)
        if r2.api_success and r2.parse_success and r2.parsed_headers:
            result.parsed_headers = r2.parsed_headers
            result.parse_success = True
            result.parse_error = ""
            result.capped = False
            result.continuation_used = True
            result.continuation_tier = 2
            return
        logging.info("Continuation tier 3 (force answer, thinking off)")
        combined = (partial + "\n" + (r2.raw_response or "")).strip()
        r3 = await async_api_call(client, cfg, budget,
                                  _force_answer_messages(messages, combined),
                                  requested_max_tokens, disable_thinking=True,
                                  allow_continuation=False)
        _accumulate(result, r3)
        result.continuation_used = True
        result.continuation_tier = 3
        if r3.api_success and r3.parse_success and r3.parsed_headers:
            result.parsed_headers = r3.parsed_headers
            result.parse_success = True
            result.parse_error = ""
            result.capped = False
        else:
            logging.warning("Continuation exhausted, keeping empty result")
    elif ok and parsed:
        last_row = _last_row(parsed)
        logging.info("Continuation tier 1 (extend list)")
        r1 = await async_api_call(client, cfg, budget,
                                  _extend_list_messages(messages, raw, last_row),
                                  requested_max_tokens, allow_continuation=False)
        _accumulate(result, r1)
        result.continuation_used = True
        result.continuation_tier = 1
        if r1.api_success and r1.parse_success:
            seen = {(h["row"], h["col"]) for h in result.parsed_headers}
            for h in r1.parsed_headers:
                key = (h["row"], h["col"])
                if key not in seen:
                    seen.add(key)
                    result.parsed_headers.append(h)
            result.parsed_headers.sort(key=lambda x: (x["row"], x["col"]))
            result.capped = False


def _accumulate(result: ApiResult, other: ApiResult):
    ot = other.tokens_used or {}
    if result.tokens_used is None:
        result.tokens_used = {"prompt": 0, "completion": 0, "total": 0}
    result.tokens_used["completion"] = (result.tokens_used.get("completion") or 0) + (ot.get("completion") or 0)
    result.tokens_used["total"] = (result.tokens_used.get("total") or 0) + (ot.get("total") or 0)
    result.duration_sec = (result.duration_sec or 0) + (other.duration_sec or 0)
