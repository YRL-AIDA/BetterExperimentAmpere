import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import (Config, EXCLUDE_DIR_NAMES, EXCLUDE_FILE_PREFIXES,
                     EXCLUDE_PROMPT_FILES, LABEL_KEYS)
from .evaluation import (extract_true_coords_from_cells,
                         extract_true_coords_from_headers,
                         extract_type_coords_from_cells, to_one_based_coords)


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9._-]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_") or "item"


def stable_hash(text: str, length: int = 12) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def sanitize_for_prompt(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: sanitize_for_prompt(v) for k, v in obj.items()
                if k not in LABEL_KEYS and not k.startswith("is_")}
    if isinstance(obj, list):
        return [sanitize_for_prompt(x) for x in obj]
    return obj


def rows_bin(n: int) -> str:
    if n <= 10:
        return "<=10"
    if n <= 25:
        return "11-25"
    return ">25"


def count_bin(n: int) -> str:
    if n == 0:
        return "0"
    if n <= 2:
        return "1-2"
    if n <= 5:
        return "3-5"
    return "6+"


def parse_format_ratio(ratio_str: str) -> Tuple[float, float]:
    try:
        j, h = ratio_str.strip().split(":")
        j, h = float(j), float(h)
        total = j + h
        if total <= 0:
            raise ValueError
        return j / total, h / total
    except Exception:
        raise ValueError(f"Invalid FORMAT_RATIO '{ratio_str}'. Use e.g. '50:50'.")


def strip_html_header_hints(html: str) -> str:
    html = re.sub(r"<th(\b[^>]*)>", r"<td\1>", html, flags=re.IGNORECASE)
    html = re.sub(r"</th>", "</td>", html, flags=re.IGNORECASE)
    html = re.sub(r'\s+(?:class|id|style)=["\'][^"\']*["\']', "", html, flags=re.IGNORECASE)
    return html


class TableLoader:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def load_system_prompt(self) -> str:
        path = self.cfg.prompts_dir / "system.txt"
        if not path.exists():
            raise FileNotFoundError(f"System prompt not found: {path}")
        return path.read_text(encoding="utf-8").strip()

    def load_prompt_configs(self) -> List[Dict[str, Any]]:
        files = [p for p in sorted(self.cfg.prompts_dir.glob("*.txt"), key=lambda p: p.name)
                 if p.name not in EXCLUDE_PROMPT_FILES]
        if not files:
            raise FileNotFoundError(f"No prompt .txt files in {self.cfg.prompts_dir}")
        return [{"name": p.stem, "file": str(p), "user": p.read_text(encoding="utf-8").strip()}
                for p in files if p.read_text(encoding="utf-8").strip()]

    def _should_skip_path(self, path: Path) -> bool:
        return (any(part in EXCLUDE_DIR_NAMES for part in path.parts)
                or path.name.startswith(EXCLUDE_FILE_PREFIXES))

    def _iter_json_files(self, root: Path):
        seen = set()
        for path in sorted(root.rglob("*.json"), key=str):
            if self._should_skip_path(path):
                continue
            r = str(path.resolve())
            if r not in seen:
                seen.add(r)
                yield path

    @staticmethod
    def _table_dims_cells(cells) -> Tuple[int, int]:
        rows, cols = set(), set()
        for cell in cells or []:
            for r in cell.get("row_nums", []) or []:
                try:
                    rows.add(int(r))
                except Exception:
                    pass
            for c in cell.get("column_nums", []) or []:
                try:
                    cols.add(int(c))
                except Exception:
                    pass
        return (max(rows) + 1 if rows else 0, max(cols) + 1 if cols else 0)

    @staticmethod
    def _table_dims_matrix(data) -> Tuple[int, int]:
        if not isinstance(data, list):
            return 0, 0
        return len(data), max((len(r) for r in data if isinstance(r, list)), default=0)

    def _make_table_record(self, filepath: Path, item: Dict,
                           idx: int, source_name: str) -> Optional[Dict[str, Any]]:
        if not isinstance(item, dict):
            return None
        if "cells" not in item and "data" not in item:
            return None

        prompt_obj = sanitize_for_prompt(item)
        table_json = json.dumps(prompt_obj, ensure_ascii=False, separators=(",", ":"))
        table_hash = stable_hash(table_json, 12)

        if "cells" in item:
            ti = extract_type_coords_from_cells(item.get("cells", []))
            true_raw, gt_text = extract_true_coords_from_cells(item.get("cells", []))
            kind = "cells"
            nr, nc = self._table_dims_cells(item.get("cells", []))
            has_ti = True
        else:
            true_raw, gt_text = extract_true_coords_from_headers(item.get("headers", []))
            kind = "matrix"
            nr, nc = self._table_dims_matrix(item.get("data", []))
            has_ti = False
            ti = {"column_headers": [], "projected_row_headers": [], "spanning": [],
                  "column_header_cell_count": 0, "projected_row_header_cell_count": 0,
                  "spanning_cell_count": 0, "spanning_zones": []}

        th_by_type = {k: ti[k] for k in ["column_headers", "projected_row_headers", "spanning"]}

        return {
            "source_group": source_name,
            "source_file": str(filepath),
            "source_stem": filepath.stem,
            "table_index": idx,
            "table_kind": kind,
            "table_rows": nr,
            "table_cols": nc,
            "table_rows_bin": rows_bin(nr),
            "table_hash": table_hash,
            "table_json": table_json,
            "table_html": "",
            "true_headers_raw": true_raw,
            "true_headers_1based": to_one_based_coords(true_raw),
            "true_headers_count": len(true_raw),
            "true_headers_count_bin": count_bin(len(true_raw)),
            "true_headers_by_type_raw": th_by_type,
            "true_headers_text": {f"{r},{c}": t for (r, c), t in gt_text.items()},
            "has_type_info": has_ti,
            "spanning_zones": ti.get("spanning_zones", []),
            "column_header_cell_count": ti["column_header_cell_count"],
            "projected_row_header_cell_count": ti["projected_row_header_cell_count"],
            "spanning_cell_count": ti["spanning_cell_count"],
            "spanning_cell_count_bin": count_bin(ti["spanning_cell_count"]),
        }

    def _load_json_records(self, json_root: Path, source_name: str, limit: int) -> List[Dict[str, Any]]:
        if not json_root.exists():
            raise FileNotFoundError(f"JSON root does not exist: {json_root}")
        records = []
        for fp in self._iter_json_files(json_root):
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                if isinstance(raw, dict):
                    items = [raw]
                elif isinstance(raw, list):
                    items = [x for x in raw if isinstance(x, dict) and ("cells" in x or "data" in x)]
                    if not items:
                        items = [x for x in raw if isinstance(x, dict)]
                else:
                    items = []
                for idx, item in enumerate(items):
                    rec = self._make_table_record(fp, item, idx, source_name)
                    if rec:
                        records.append(rec)
                if len(records) >= limit:
                    break
            except json.JSONDecodeError as e:
                logging.error(f"JSON parse error {fp}: {e}")
            except Exception as e:
                logging.error(f"Error reading {fp}: {e}")
        return records[:limit]

    def _attach_html(self, records, html_root: Path):
        if not html_root.exists():
            logging.warning(f"HTML root not found: {html_root}. HTML format unavailable.")
            return records
        html_files: Dict[str, Path] = {}
        for hp in sorted(html_root.rglob("*.html"), key=str):
            html_files[hp.stem] = hp
        attached = 0
        for rec in records:
            stem = rec["source_stem"]
            if stem in html_files:
                try:
                    rec["table_html"] = strip_html_header_hints(html_files[stem].read_text(encoding="utf-8"))
                    attached += 1
                except Exception as e:
                    logging.warning(f"Could not read HTML for {stem}: {e}")
        logging.info(f"  HTML attached: {attached}/{len(records)} records")
        return records

    def load_all(self, run_dir: Path) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        json_frac, html_frac = parse_format_ratio(self.cfg.format_ratio)
        model_alias = self.cfg.model_alias or slugify(self.cfg.model_name.split("/")[-1])[:12]

        seed: Optional[Dict] = None
        if self.cfg.table_seed_path:
            with open(self.cfg.table_seed_path, "r", encoding="utf-8") as f:
                seed = json.load(f)
            logging.info(f"Loaded table seed from {self.cfg.table_seed_path}")

        plan = self.cfg.experiment_plan()
        raw_map: Dict[str, List[Dict[str, Any]]] = {}
        for src in plan:
            records = self._load_json_records(src["json_root"], src["name"], int(src["limit"]))
            records = self._attach_html(records, src["html_root"])
            raw_map[src["name"]] = records
            logging.info(f"Loaded {len(records)} records from {src['name']}")

        if self.cfg.total_tables > 0:
            n_sources = len(plan)
            per_source = self.cfg.total_tables // n_sources
            remainder = self.cfg.total_tables - per_source * n_sources
        else:
            per_source = None
            remainder = 0

        table_map: Dict[str, Dict[str, List[Dict]]] = {}
        total_json = total_html = 0
        seed_data: Dict[str, Dict[str, List[str]]] = {}

        for i, src in enumerate(plan):
            records = raw_map[src["name"]]
            if seed:
                allowed_json = set(seed.get("sources", {}).get(src["name"], {}).get("json", []))
                allowed_html = set(seed.get("sources", {}).get(src["name"], {}).get("html", []))
                json_sample = [r for r in records if r["source_stem"] in allowed_json]
                html_sample = [r for r in records if r.get("table_html") and r["source_stem"] in allowed_html]
            else:
                if per_source is not None:
                    n_src = per_source + (1 if i < remainder else 0)
                    records = records[:n_src]
                if html_frac == 0:
                    json_sample, html_sample = records, []
                elif json_frac == 0:
                    json_sample, html_sample = [], [r for r in records if r.get("table_html")]
                else:
                    json_sample = records
                    html_sample = [r for r in records if r.get("table_html")]

            table_map[src["name"]] = {"json": json_sample, "html": html_sample}
            seed_data[src["name"]] = {
                "json": [r["source_stem"] for r in json_sample],
                "html": [r["source_stem"] for r in html_sample],
            }
            total_json += len(json_sample)
            total_html += len(html_sample)
            logging.info(f"  {src['name']}: json={len(json_sample)} html={len(html_sample)}")

        logging.info(f"Total: json={total_json} html={total_html} sum={total_json + total_html}")

        seed_out = run_dir / "selected_tables.json"
        with open(seed_out, "w", encoding="utf-8") as f:
            json.dump({
                "model": self.cfg.model_name,
                "model_alias": model_alias,
                "total_tables": self.cfg.total_tables,
                "format_ratio": f"json:{json_frac:.0%} html:{html_frac:.0%}",
                "sources": seed_data,
            }, f, ensure_ascii=False, indent=2)
        logging.info(f"Table selection saved: {seed_out}")
        return table_map
