import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

from .chunking import header_aware_chunks, merge_predictions, whole_table_chunk
from .config import Config, FATAL_ERROR_TYPES
from .datamodel import ApiResult
from .evaluation import (coords_to_set, evaluate_coord_sets, evaluate_spanning_soft,
                         evaluate_text_metrics)
from .loading import TableLoader, slugify
from .persistence import Persistence, build_metrics
from .prompts import get_requested_max_tokens, prepare_messages
from .transport import (BudgetController, async_api_call, count_prompt_tokens,
                        detect_context_limit)


class Collector:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.budget = BudgetController(cfg)
        self.model_alias = cfg.model_alias or slugify(cfg.model_name.split("/")[-1])[:12]

        ts = datetime.now().strftime("%d.%m.%Y")
        run_id = f"{ts}_{self.model_alias}"
        base = Path(cfg.output_dir)
        run_base = base / f"run_{run_id}"
        if run_base.exists():
            counter = 2
            while (base / f"run_{run_id}_{counter}").exists():
                counter += 1
            run_id = f"{run_id}_{counter}"
        self.run_dir = base / f"run_{run_id}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        loader = TableLoader(cfg)
        self.system_prompt = loader.load_system_prompt()
        self.prompts = loader.load_prompt_configs()
        self.table_map = loader.load_all(self.run_dir)

        self.responses: List[Dict[str, Any]] = []
        self.valid_responses: List[Dict[str, Any]] = []
        self.api_failed: List[Dict[str, Any]] = []
        self.parse_failed: List[Dict[str, Any]] = []

        self.completed = 0
        self._consec_fail = 0
        self._abort = False
        self._lock = asyncio.Lock()

        self.meta = {
            "model": cfg.model_name, "model_alias": self.model_alias,
            "temperature": cfg.temperature, "seed": cfg.seed,
            "max_tokens": cfg.max_tokens, "chunk_strategy": cfg.chunk_strategy,
            "header_zone_rows": cfg.header_zone_rows,
        }
        self.persistence = Persistence(self.run_dir, self.meta, cfg.snapshot_every)

    async def _check_server(self, client: httpx.AsyncClient) -> bool:
        if self.cfg.auto_detect_window:
            detected = await detect_context_limit(client, self.cfg)
            if detected:
                self.budget.set_window(detected)
            else:
                logging.warning(f"Could not auto-detect window; using {self.budget.window}")
        base = self.cfg.vllm_base_url.rstrip("/").removesuffix("/v1")
        for url in [f"{base}/health", f"{base}/ping", f"{self.cfg.vllm_base_url}/models"]:
            try:
                r = await client.get(url)
                if r.status_code in (200, 405):
                    logging.info(f"Server reachable via {url}")
                    return True
            except Exception:
                continue
        logging.critical("Server not reachable")
        return False

    def _build_request_id(self, prompt_name, tr, table_format) -> str:
        base = (f"{prompt_name}__{tr['source_group']}__{tr['source_stem']}"
                f"__t{tr['table_index']}__{tr['table_hash']}__{table_format}__{self.model_alias}")
        return slugify(base)

    async def _measure(self, client, table_repr, info, prompt_config, table_format):
        msgs = prepare_messages(self.system_prompt, prompt_config, table_repr, table_format, info)
        return await count_prompt_tokens(client, self.cfg, msgs)

    async def _call(self, client, sem, messages, requested_mt, known_prompt_tokens=None) -> ApiResult:
        async with sem:
            res = await async_api_call(client, self.cfg, self.budget, messages,
                                       requested_mt, known_prompt_tokens=known_prompt_tokens)
            if self.cfg.inter_request_delay > 0:
                await asyncio.sleep(self.cfg.inter_request_delay)
        return res

    async def _process_one(self, client, sem, prompt_idx, prompt_config, tr, table_format) -> Dict:
        if self._abort:
            ar = ApiResult(api_success=False, error_type="aborted", error_message="early stop")
            return self._make_result(prompt_idx, prompt_config, tr, ar, table_format)

        pname = prompt_config.get("name", "")
        requested_mt = get_requested_max_tokens(
            self.cfg, pname, tr["table_rows"], tr["table_cols"])

        if table_format == "html":
            table_repr = tr.get("table_html") or tr["table_json"]
            msgs = prepare_messages(self.system_prompt, prompt_config, table_repr, "html")
            ar = await self._call(client, sem, msgs, requested_mt)
            return self._make_result(prompt_idx, prompt_config, tr, ar, table_format)

        table_repr = tr["table_json"]
        target_input = max(self.budget.window // 2,
                           self.budget.window - requested_mt - self.cfg.context_safety_margin)
        full_msgs = prepare_messages(self.system_prompt, prompt_config, table_repr, "json")
        full_toks = await count_prompt_tokens(client, self.cfg, full_msgs)

        if full_toks is not None and full_toks <= target_input:
            ar = await self._call(client, sem, full_msgs, requested_mt, known_prompt_tokens=full_toks)
            return self._make_result(prompt_idx, prompt_config, tr, ar, table_format)

        if self.cfg.chunk_strategy == "whole":
            chunks = whole_table_chunk(tr)
        else:
            async def measure(repr_str, info):
                return await self._measure(client, repr_str, info, prompt_config, "json")
            chunks = await header_aware_chunks(tr, self.cfg.header_zone_rows, target_input, measure)

        logging.info(f"Chunking {tr['source_stem']} ({tr['table_rows']}x{tr['table_cols']}, "
                     f"~{full_toks if full_toks is not None else '?'} toks) -> {len(chunks)} chunks")

        chunk_headers: List[List[Dict]] = []
        total_dur = total_pt = total_ct = total_ret = 0
        n_api_failed = 0
        n_parse_failed = 0
        n_responded = 0
        any_capped = False
        any_cont = False
        any_forced = False
        max_tier = 0
        raw_parts = []

        for ci, (chunk_repr, info) in enumerate(chunks):
            if self._abort:
                n_api_failed += 1
                break
            msgs = prepare_messages(self.system_prompt, prompt_config, chunk_repr, "json", info)
            ar = await self._call(client, sem, msgs, requested_mt)
            if not ar.api_success:
                n_api_failed += 1
                logging.warning(f"Chunk {ci+1}/{len(chunks)} of {tr['source_stem']} api failed: "
                                f"{ar.error_message[:80]}")
            else:
                n_responded += 1
                if ar.parse_success:
                    chunk_headers.append(ar.parsed_headers)
                else:
                    n_parse_failed += 1
                    logging.warning(f"Chunk {ci+1}/{len(chunks)} of {tr['source_stem']} parse failed: "
                                    f"{ar.parse_error}")
            any_capped = any_capped or ar.capped
            any_cont = any_cont or ar.continuation_used
            any_forced = any_forced or ar.continuation_forced
            max_tier = max(max_tier, ar.continuation_tier)
            total_dur += ar.duration_sec or 0
            total_ret += ar.retry_attempts or 1
            raw_parts.append(ar.raw_response or "")
            tu = ar.tokens_used or {}
            total_pt += tu.get("prompt", 0) or 0
            total_ct += tu.get("completion", 0) or 0

        merged = merge_predictions(chunk_headers)
        n_failed = n_api_failed + n_parse_failed
        api_ok = n_responded > 0
        parse_ok = api_ok and len(chunk_headers) > 0
        if not api_ok:
            err_type = "api_error"
            err_msg = f"{n_failed}/{len(chunks)} chunks failed (no usable response)"
        elif not parse_ok:
            err_type = "chunk_parse_failed"
            err_msg = f"{n_parse_failed}/{len(chunks)} chunks unparseable"
        else:
            err_type = ""
            err_msg = ""
        combined = ApiResult(
            api_success=api_ok,
            parse_success=parse_ok,
            parsed_headers=merged,
            parse_error=("" if n_failed == 0 else f"{n_failed}/{len(chunks)}_chunks_failed"),
            raw_response=" | ".join(r for r in raw_parts if r)[:500],
            duration_sec=total_dur, retry_attempts=total_ret,
            requested_max_tokens=requested_mt, effective_max_tokens=requested_mt,
            capped=any_capped, continuation_used=any_cont, continuation_tier=max_tier,
            continuation_forced=any_forced,
            error_type=err_type, error_message=err_msg,
            tokens_used={"prompt": total_pt, "completion": total_ct, "total": total_pt + total_ct})
        combined.n_chunks_failed = n_failed
        return self._make_result(prompt_idx, prompt_config, tr, combined, table_format,
                                 chunked=True, n_chunks=len(chunks))

    def _make_result(self, prompt_idx, prompt_config, tr, ar: ApiResult,
                     table_format, chunked=False, n_chunks=1) -> Dict[str, Any]:
        pname = str(prompt_config.get("name", f"prompt_{prompt_idx}"))
        true_set = coords_to_set(tr["true_headers_raw"])
        pred_set = (coords_to_set(ar.parsed_headers)
                    if ar.api_success and ar.parse_success else set())
        overall = evaluate_coord_sets(true_set, pred_set, tr["table_rows"], tr["table_cols"])
        nr, nc = tr["table_rows"], tr["table_cols"]
        pred_set_f = {(r, c) for r, c in pred_set if 0 <= r < nr and 0 <= c < nc}
        oob_pred_count = len(pred_set - pred_set_f)

        raw_gt = tr.get("true_headers_text", {})
        gt_text_map = {(int(k.split(",")[0]), int(k.split(",")[1])): v for k, v in raw_gt.items()}
        text_metrics = evaluate_text_metrics(gt_text_map, ar.parsed_headers, true_set, pred_set_f)

        tu = ar.tokens_used or {}
        ct = tu.get("completion")
        tef = (ct / overall["f1"]) if (ct and overall["f1"] > 0) else None

        type_metrics: Dict[str, Any] = {}
        for tname in ["column_headers", "projected_row_headers", "spanning"]:
            if tr.get("has_type_info"):
                ts = coords_to_set(tr["true_headers_by_type_raw"].get(tname, []))
                for k, v in evaluate_coord_sets(ts, pred_set, nr, nc).items():
                    type_metrics[f"{tname}_{k}"] = v
            else:
                for k in ["support", "pred_count", "tp", "fp", "fn", "precision", "recall",
                          "f1", "jaccard", "exact_match", "partial_match", "header_coverage"]:
                    type_metrics[f"{tname}_{k}"] = None
        type_metrics.update(evaluate_spanning_soft(tr.get("spanning_zones", []), pred_set_f, nr, nc))

        result = {
            "request_id": self._build_request_id(pname, tr, table_format),
            "timestamp": datetime.now().isoformat(),
            "model": self.cfg.model_name, "model_alias": self.model_alias,
            "table_format": table_format, "prompt_idx": prompt_idx, "prompt_name": pname,
            "prompt_file": prompt_config.get("file", ""),
            "source_group": tr["source_group"], "source_file": tr["source_file"],
            "source_stem": tr["source_stem"], "table_index": tr["table_index"],
            "table_kind": tr["table_kind"], "table_rows": tr["table_rows"],
            "table_cols": tr["table_cols"], "table_rows_bin": tr["table_rows_bin"],
            "table_hash": tr["table_hash"],
            "true_headers_raw": tr["true_headers_raw"],
            "true_headers_1based": tr["true_headers_1based"],
            "true_headers_count": tr["true_headers_count"],
            "true_headers_count_bin": tr["true_headers_count_bin"],
            "true_headers_by_type_raw": tr["true_headers_by_type_raw"],
            "has_type_info": tr["has_type_info"],
            "column_header_cell_count": tr["column_header_cell_count"],
            "projected_row_header_cell_count": tr["projected_row_header_cell_count"],
            "spanning_cell_count": tr["spanning_cell_count"],
            "spanning_cell_count_bin": tr["spanning_cell_count_bin"],
            "chunked": chunked, "n_chunks": n_chunks,
            "n_chunks_failed": getattr(ar, "n_chunks_failed", 0),
            "oob_pred_count": oob_pred_count,
            "continuation_used": ar.continuation_used,
            "continuation_tier": ar.continuation_tier,
            "continuation_forced": ar.continuation_forced,
            "output_complete": ar.output_complete,
            "budget_clamped": ar.budget_clamped,
            "api_success": ar.api_success, "parse_success": ar.parse_success,
            "status": ("api_failed" if not ar.api_success
                       else ("ok" if ar.parse_success else "parse_failed")),
            "raw_response": ar.raw_response, "parsed_headers": ar.parsed_headers,
            "parse_error": ar.parse_error, "error_type": ar.error_type,
            "error_message": ar.error_message, "duration_sec": ar.duration_sec,
            "retry_attempts": ar.retry_attempts,
            "requested_max_tokens": ar.requested_max_tokens,
            "effective_max_tokens": ar.effective_max_tokens,
            "completion_capped": ar.capped,
            "tokens_used": ar.tokens_used,
            "prompt_tokens": tu.get("prompt"), "completion_tokens": ct,
            "total_tokens": tu.get("total"), "token_efficiency": tef,
        }
        result.update(overall)
        result.update(type_metrics)
        result.update(text_metrics)
        return result

    def _register(self, result: Dict[str, Any]):
        if result["api_success"]:
            self.responses.append(result)
            (self.valid_responses if result["parse_success"] else self.parse_failed).append(result)
        else:
            self.api_failed.append(result)
        self.completed += 1

    def _build_tasks(self) -> List[Tuple]:
        tasks = []
        for src in self.cfg.experiment_plan():
            allowed = src["prompts"]
            for fmt in ("json", "html"):
                records = self.table_map[src["name"]][fmt]
                for pi, pc in enumerate(self.prompts):
                    if allowed is None or pc["name"] in allowed:
                        for tr in records:
                            tasks.append((pi, pc, tr, fmt))
        return tasks

    async def _run_tasks(self, tasks: List[Tuple], timestamp: str):
        total = len(tasks)
        sem = asyncio.Semaphore(self.cfg.concurrency)
        queue: asyncio.Queue = asyncio.Queue()
        for t in tasks:
            await queue.put(t)
        limits = httpx.Limits(max_connections=self.cfg.concurrency + 2,
                              max_keepalive_connections=self.cfg.concurrency)
        timeout = httpx.Timeout(self.cfg.request_timeout_sec if self.cfg.request_timeout_sec > 0 else None)

        async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
            if not await self._check_server(client):
                logging.critical("Aborting: server not healthy")
                return

            async def worker():
                while True:
                    try:
                        item = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        return
                    pi, pc, tr, fmt = item
                    result = await self._process_one(client, sem, pi, pc, tr, fmt)
                    async with self._lock:
                        self._register(result)
                        if (not result["api_success"]
                                and result.get("error_type") in FATAL_ERROR_TYPES):
                            self._consec_fail += 1
                        else:
                            self._consec_fail = 0
                        if self._consec_fail >= self.cfg.early_stop_failures and not self._abort:
                            self._abort = True
                            logging.critical(f"EARLY STOP: {self._consec_fail} consecutive failures")
                        self.persistence.append(result)
                        do_snapshot = (self.completed % self.cfg.snapshot_every == 0)
                        dur = f"{result['duration_sec']:.1f}s" if result.get("duration_sec") else "n/a"
                        cont = f" cont{result['continuation_tier']}" if result.get("continuation_used") else ""
                        cap = " [CAP]" if result.get("completion_capped") else ""
                        ch = f" [x{result.get('n_chunks',1)}ch]" if result.get("chunked") else ""
                        f1 = result.get("f1", 0) or 0
                        logging.info(f"[{self.completed}/{total}] {result['prompt_name']} "
                                     f"[{result['table_format']}] | {result['source_stem']}{ch} | "
                                     f"status={result['status']} F1={f1:.3f}{cap}{cont} | dur={dur}")
                        if do_snapshot:
                            self.persistence.snapshot(self.responses, self.api_failed,
                                                      self.parse_failed, self.completed)
                    queue.task_done()

            workers = [asyncio.ensure_future(worker()) for _ in range(self.cfg.concurrency)]
            await queue.join()
            for w in workers:
                w.cancel()

        self.persistence.snapshot(self.responses, self.api_failed, self.parse_failed, self.completed)

    async def _run_async(self):
        ts = datetime.now().strftime("%d.%m.%Y")
        tasks = self._build_tasks()
        logging.info(f"Total {len(tasks)} tasks | concurrency={self.cfg.concurrency} "
                     f"window={self.budget.window} temp={self.cfg.temperature} seed={self.cfg.seed}")
        await self._run_tasks(tasks, ts)
        self.persistence.save_final(self.responses, self.api_failed, self.parse_failed, ts)
        build_metrics(self.responses, self.api_failed, self.persistence.metrics_dir, self.meta)

    def run(self):
        asyncio.run(self._run_async())

    def _table_lookup(self):
        lut = {}
        for fmt_records in self.table_map.values():
            for records in fmt_records.values():
                for tr in records:
                    lut[(tr["source_group"], tr["source_stem"], tr["table_index"], tr["table_hash"])] = tr
        return lut

    async def _run_retry_async(self, checkpoint_path: str, mode: str):
        import json as _json
        ts = datetime.now().strftime("%d.%m.%Y") + ("_capped_retry" if mode == "capped" else "_retry")
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            ckpt = _json.load(f)
        self.responses = ckpt.get("responses", [])
        self.parse_failed = ckpt.get("parse_failed_requests", [])
        self.valid_responses = [r for r in self.responses if r.get("parse_success")]
        self.api_failed = ckpt.get("api_failed_requests", [])

        if mode == "capped":
            candidates = [r for r in self.responses
                          if r.get("completion_capped")
                          and (r.get("f1", 1.0) == 0.0
                               or r.get("parse_error") == "truncated_inside_think_block")]
            cand_ids = {r["request_id"] for r in candidates}
            self.responses = [r for r in self.responses if r["request_id"] not in cand_ids]
            self.valid_responses = [r for r in self.valid_responses if r["request_id"] not in cand_ids]
        else:
            candidates = self.api_failed
            self.api_failed = []

        logging.info(f"Retry({mode}): {len(candidates)} candidates")
        if not candidates:
            logging.info("Nothing to retry")
            return

        pbn = {pc["name"]: (pi, pc) for pi, pc in enumerate(self.prompts)}
        lut = self._table_lookup()
        tasks, skipped = [], 0
        for rec in candidates:
            pname = rec.get("prompt_name", "")
            if pname not in pbn:
                skipped += 1
                continue
            pi, pc = pbn[pname]
            tr = lut.get((rec.get("source_group"), rec.get("source_stem"),
                          rec.get("table_index", 0), rec.get("table_hash", "")))
            if tr is None:
                skipped += 1
                continue
            tasks.append((pi, pc, tr, rec.get("table_format", "json")))
        logging.info(f"Rebuilt {len(tasks)} retry tasks (skipped {skipped})")
        self.completed = len(self.responses) + len(self.parse_failed) + len(self.api_failed)
        await self._run_tasks(tasks, ts)
        self.persistence.save_final(self.responses, self.api_failed, self.parse_failed, ts)
        build_metrics(self.responses, self.api_failed, self.persistence.metrics_dir, self.meta)

    def run_retry(self, checkpoint_path: str):
        asyncio.run(self._run_retry_async(checkpoint_path, "failed"))

    def run_retry_capped(self, checkpoint_path: str):
        asyncio.run(self._run_retry_async(checkpoint_path, "capped"))