import json
import logging
from typing import Any, Awaitable, Callable, Dict, List, Tuple

from .loading import sanitize_for_prompt


def _subset_cells(obj: Dict, row_set: set) -> Dict:
    out = []
    for cell in obj.get("cells", []):
        rows = cell.get("row_nums", []) or []
        if any(r in row_set for r in rows):
            out.append(dict(cell))
    res = {k: v for k, v in obj.items() if k != "cells"}
    res["cells"] = out
    return res


def _subset_matrix(obj: Dict, row_indices: List[int]) -> Dict:
    data = obj.get("data") or []
    rows = [data[i] for i in row_indices if 0 <= i < len(data)]
    res = {k: v for k, v in obj.items() if k not in ("data",)}
    res["row_indices"] = row_indices
    res["data"] = rows
    return res


def _render(obj: Dict, kind: str, rows: List[int]) -> str:
    row_set = set(rows)
    if kind == "cells" or (kind != "matrix" and "cells" in obj):
        sub = _subset_cells(obj, row_set)
    else:
        sub = _subset_matrix(obj, sorted(rows))
    return json.dumps(sanitize_for_prompt(sub), ensure_ascii=False, separators=(",", ":"))


def merge_predictions(chunk_headers: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    by_coord: Dict[Tuple[int, int], str] = {}
    for headers in chunk_headers:
        for h in headers or []:
            try:
                key = (int(h["row"]), int(h["col"]))
            except Exception:
                continue
            txt = str(h.get("text", "")).strip()
            if key not in by_coord or (not by_coord[key] and txt):
                by_coord[key] = txt
    merged = [{"row": r, "col": c, "text": t} for (r, c), t in by_coord.items()]
    merged.sort(key=lambda x: (x["row"], x["col"]))
    return merged


def whole_table_chunk(record: Dict) -> List[Tuple[str, str]]:
    return [(record["table_json"], "")]


async def header_aware_chunks(
    record: Dict, header_zone_rows: int, target_input_tokens: int,
    measure: Callable[[str, str], Awaitable[int]],
) -> List[Tuple[str, str]]:
    obj = json.loads(record["table_json"])
    kind = record["table_kind"]
    nrows = record["table_rows"]
    hz = list(range(0, min(header_zone_rows, nrows)))
    data = list(range(len(hz), nrows))

    if not data:
        info = ("This is the full header region of the table; coordinates are "
                "absolute row/col indices.")
        return [(_render(obj, kind, hz), info)]

    chunks: List[Tuple[str, str]] = []
    stack: List[Tuple[int, int]] = [(0, len(data))]
    guard = 0
    max_iter = 20 * len(data) + 100
    while stack:
        guard += 1
        if guard > max_iter:
            logging.error(f"Chunking guard hit for a table with {nrows} rows; "
                          f"{len(stack)} ranges left unprocessed — rows may be missing")
            break
        a, b = stack.pop()
        window = data[a:b]
        rows = hz + window
        repr_str = _render(obj, kind, rows)
        info = (f"Rows {hz[0] if hz else 0}..{hz[-1] if hz else 0} are the header "
                f"region; rows {window[0]}..{window[-1]} are data rows being examined. "
                f"All coordinates are ABSOLUTE row/col indices of the full table.")
        toks = await measure(repr_str, info)
        if toks is None or toks <= target_input_tokens or (b - a) <= 1:
            chunks.append((repr_str, info))
        else:
            mid = (a + b) // 2
            stack.append((mid, b))
            stack.append((a, mid))

    def first_data_row(chunk):
        try:
            o = json.loads(chunk[0])
            if "row_indices" in o:
                idx = [i for i in o["row_indices"] if i >= len(hz)]
                return min(idx) if idx else 0
            rows = set()
            for c in o.get("cells", []):
                for r in (c.get("row_nums") or []):
                    if r >= len(hz):
                        rows.add(r)
            return min(rows) if rows else 0
        except Exception:
            return 0

    chunks.sort(key=first_data_row)
    return chunks
