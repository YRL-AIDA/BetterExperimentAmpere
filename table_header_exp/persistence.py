import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def _df_friendly(records: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for rec in records:
        row = dict(rec)
        for key in ["true_headers_raw", "true_headers_1based",
                    "true_headers_by_type_raw", "true_headers_text",
                    "parsed_headers", "tokens_used", "spanning_zones"]:
            if key in row and isinstance(row[key], (list, dict)):
                row[key] = json.dumps(row[key], ensure_ascii=False)
        rows.append(row)
    return pd.DataFrame(rows)


class Persistence:
    def __init__(self, run_dir: Path, meta: Dict[str, Any], snapshot_every: int = 500):
        self.run_dir = run_dir
        self.meta = meta
        self.snapshot_every = snapshot_every
        self.logs_dir = run_dir / "logs"
        self.results_dir = run_dir / "results"
        self.metrics_dir = run_dir / "metrics"
        self.ckpt_dir = run_dir / "checkpoints"
        for d in [self.logs_dir, self.results_dir, self.metrics_dir, self.ckpt_dir]:
            d.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.ckpt_dir / "results.jsonl"
        self._since_snapshot = 0

    def append(self, result: Dict[str, Any]):
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        self._since_snapshot += 1
        if self._since_snapshot >= self.snapshot_every:
            self._since_snapshot = 0
            return True
        return False

    def snapshot(self, responses, api_failed, parse_failed, completed):
        path = self.ckpt_dir / "checkpoint_latest.json"
        meta = dict(self.meta)
        meta.update({
            "timestamp": datetime.now().isoformat(),
            "completed_count": completed,
            "responses": len(responses),
            "api_failed": len(api_failed),
            "parse_failed": len(parse_failed),
        })
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({
                "metadata": meta,
                "responses": responses,
                "api_failed_requests": api_failed,
                "parse_failed_requests": parse_failed,
            }, f, ensure_ascii=False, indent=2)
        tmp.replace(path)
        logging.info(f"Snapshot ({completed} done): {path.name}")

    def save_final(self, responses, api_failed, parse_failed, timestamp: str):
        base = self.results_dir
        for name, d in [
            (f"responses_{timestamp}.json", responses),
            (f"api_failed_{timestamp}.json", api_failed),
            (f"parse_failed_{timestamp}.json", parse_failed),
        ]:
            with open(base / name, "w", encoding="utf-8") as f:
                json.dump(d, f, ensure_ascii=False, indent=2)
        for name, d in [
            (f"responses_{timestamp}.csv", responses),
            (f"api_failed_{timestamp}.csv", api_failed),
            (f"parse_failed_{timestamp}.csv", parse_failed),
        ]:
            if d:
                _df_friendly(d).to_csv(base / name, index=False, encoding="utf-8-sig")
        try:
            with pd.ExcelWriter(base / f"results_{timestamp}.xlsx", engine="openpyxl") as w:
                _df_friendly(responses).to_excel(w, sheet_name="responses", index=False)
                if api_failed:
                    _df_friendly(api_failed).to_excel(w, sheet_name="api_failed", index=False)
                if parse_failed:
                    _df_friendly(parse_failed).to_excel(w, sheet_name="parse_failed", index=False)
        except Exception as e:
            logging.warning(f"XLSX save failed: {e}")
        logging.info(f"Results -> {base}")


def _summarize(df: pd.DataFrame, group_cols=None, metric_prefix: str = "") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    metrics = ["support", "pred_count", "tp", "fp", "fn", "precision", "recall",
               "f1", "jaccard", "exact_match", "partial_match", "header_coverage"]
    rows = []
    iterable = df.groupby(group_cols, dropna=False) if group_cols else [((), df)]
    for key, sub in iterable:
        row: Dict[str, Any] = {}
        if group_cols:
            ks = [key] if len(group_cols) == 1 else list(key)
            for col, val in zip(group_cols, ks):
                row[col] = val
        row["count"] = len(sub)
        for col, alias in [("api_success", "api_success_rate"),
                           ("parse_success", "parse_success_rate"),
                           ("capped", "capped_rate"),
                           ("budget_clamped", "budget_clamped_rate"),
                           ("continuation_used", "continuation_used_rate"),
                           ("output_complete", "output_complete_rate")]:
            if col in sub.columns:
                row[alias] = float(pd.to_numeric(sub[col], errors="coerce").mean())
        for m in metrics:
            col = f"{metric_prefix}_{m}" if metric_prefix else m
            if col in sub.columns:
                s = pd.to_numeric(sub[col], errors="coerce")
                if s.notna().any():
                    row[f"{m}_mean"] = float(s.mean())
                    row[f"{m}_median"] = float(s.median())
        for col in ["duration_sec", "prompt_tokens", "completion_tokens",
                    "total_tokens", "token_efficiency", "effective_max_tokens"]:
            if col in sub.columns:
                s = pd.to_numeric(sub[col], errors="coerce").dropna()
                if len(s):
                    row[f"{col}_mean"] = float(s.mean())
                    row[f"{col}_median"] = float(s.median())
        rows.append(row)
    return pd.DataFrame(rows)


def build_metrics(responses, api_failed, metrics_dir: Path, meta: Dict[str, Any]):
    all_r = responses + api_failed
    if not all_r:
        return
    df = pd.DataFrame(all_r)
    views = {
        "overall": _summarize(df),
        "by_model": _summarize(df, ["model_alias"]),
        "by_prompt": _summarize(df, ["prompt_name"]),
        "by_format": _summarize(df, ["table_format"]),
        "by_prompt_format": _summarize(df, ["prompt_name", "table_format"]),
        "by_source": _summarize(df, ["source_group"]),
        "by_rows_bin": _summarize(df, ["table_rows_bin"]),
        "by_headers_bin": _summarize(df, ["true_headers_count_bin"]),
        "by_spanning_bin": _summarize(df, ["spanning_cell_count_bin"]),
        "by_chunked": _summarize(df, ["chunked"]),
    }
    type_frames = []
    for ht in ["column_headers", "projected_row_headers", "spanning", "spanning_soft"]:
        col = f"{ht}_f1"
        if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().any():
            tmp = _summarize(df, ["prompt_name", "table_format"], metric_prefix=ht)
            if not tmp.empty:
                tmp.insert(2, "header_type", ht)
                type_frames.append(tmp)
    views["by_prompt_type"] = pd.concat(type_frames, ignore_index=True) if type_frames else pd.DataFrame()

    for key, vdf in views.items():
        if not vdf.empty:
            vdf.to_csv(metrics_dir / f"metrics_{key}.csv", index=False, encoding="utf-8-sig")
    try:
        with pd.ExcelWriter(metrics_dir / "metrics.xlsx", engine="openpyxl") as w:
            for key, vdf in views.items():
                if not vdf.empty:
                    vdf.to_excel(w, sheet_name=key[:31], index=False)
    except Exception as e:
        logging.warning(f"Metrics XLSX failed: {e}")
    with open(metrics_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({k: v.to_dict(orient="records") for k, v in views.items()},
                  f, ensure_ascii=False, indent=2)

    ov = views["overall"].iloc[0].to_dict() if not views["overall"].empty else {}
    total = len(all_r); ok = len(responses)

    f1 = pd.to_numeric(df.get("f1"), errors="coerce") if "f1" in df.columns else pd.Series(dtype=float)
    support = pd.to_numeric(df.get("support"), errors="coerce") if "support" in df.columns else pd.Series(dtype=float)
    pred_count = pd.to_numeric(df.get("pred_count"), errors="coerce") if "pred_count" in df.columns else pd.Series(dtype=float)
    api_ok = df.get("api_success").astype(str).str.lower().isin(["true", "1", "1.0"]) if "api_success" in df.columns else pd.Series([False] * len(df))
    parse_ok = df.get("parse_success").astype(str).str.lower().isin(["true", "1", "1.0"]) if "parse_success" in df.columns else pd.Series([False] * len(df))
    has_hdr = support > 0
    trivial = (support == 0) & (pred_count == 0)

    def _mean(series, mask=None):
        s = series if mask is None else series[mask]
        s = s.dropna()
        return float(s.mean()) if len(s) else float("nan")

    f1_all = _mean(f1)
    f1_answered = _mean(f1, parse_ok)
    f1_with_headers = _mean(f1, parse_ok & has_hdr)
    oob = pd.to_numeric(df.get("oob_pred_count"), errors="coerce") if "oob_pred_count" in df.columns else pd.Series(dtype=float)
    ncf = pd.to_numeric(df.get("n_chunks_failed"), errors="coerce") if "n_chunks_failed" in df.columns else pd.Series(dtype=float)
    partial_scored = int((parse_ok & (ncf.fillna(0) > 0)).sum()) if len(ncf) else 0

    with open(metrics_dir / "metrics_summary.txt", "w", encoding="utf-8") as f:
        f.write("METRICS SUMMARY\n" + "=" * 80 + "\n")
        for k, v in meta.items():
            f.write(f"{k}: {v}\n")
        f.write(f"Total tasks:        {total}\n")
        f.write(f"API success rate:   {api_ok.mean():.1%} ({int(api_ok.sum())}/{total})\n")
        f.write(f"Parse success rate: {parse_ok.mean():.1%} ({int(parse_ok.sum())}/{total})\n")
        f.write(f"Tables w/ headers:  {int(has_hdr.sum())}  |  no-header (trivial) tasks: {int(trivial.sum())} ({trivial.mean():.1%})\n")
        f.write(f"OOB predictions:    {int(oob.fillna(0).sum()) if len(oob) else 0} total across all tasks\n")
        f.write(f"Scored w/ failed chunk: {partial_scored} parsed tables include >=1 failed chunk "
                f"(counted in metrics; filter on n_chunks_failed to exclude)\n\n")
        f.write("F1 (three honest cuts):\n")
        f.write(f"  F1 over ALL tasks (failures=0):        {f1_all:.4f}\n")
        f.write(f"  F1 over ANSWERED (parsed) tasks:       {f1_answered:.4f}\n")
        f.write(f"  F1 over tables WITH headers (parsed):  {f1_with_headers:.4f}   <- model quality\n\n")
        for label, key in [("Precision", "precision_mean"), ("Recall", "recall_mean"),
                           ("Jaccard", "jaccard_mean"), ("Exact match", "exact_match_mean"),
                           ("Partial match", "partial_match_mean")]:
            f.write(f"  {label:14s}: {ov.get(key, 0):.4f}\n")
        f.write(f"  Capped rate:        {ov.get('capped_rate', 0):.1%}\n")
        f.write(f"  Budget clamped:     {ov.get('budget_clamped_rate', 0):.1%}\n")
        f.write(f"  Continuation used:  {ov.get('continuation_used_rate', 0):.1%}\n")
        if "continuation_forced" in df.columns:
            forced = df["continuation_forced"].astype(str).str.lower().isin(["true", "1", "1.0"])
            f.write(f"  Forced (salvage):   {forced.mean():.1%} ({int(forced.sum())} tasks)\n")
    logging.info(f"Metrics -> {metrics_dir}")
    logging.info(f"F1(with-headers,parsed)={f1_with_headers:.4f} F1(all)={f1_all:.4f} "
                 f"api_ok={api_ok.mean():.1%} parse_ok={parse_ok.mean():.1%}")
