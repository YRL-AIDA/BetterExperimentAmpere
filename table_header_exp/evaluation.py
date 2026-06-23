import re
from collections import Counter
from typing import Any, Dict, List, Tuple


def coords_to_set(coords: List[Dict[str, int]]) -> set:
    out = set()
    for h in coords or []:
        try:
            out.add((int(h["row"]), int(h["col"])))
        except Exception:
            continue
    return out


def to_one_based_coords(coords: List[Dict[str, int]]) -> List[Dict[str, int]]:
    out = set()
    for h in coords:
        try:
            out.add((int(h["row"]) + 1, int(h["col"]) + 1))
        except Exception:
            continue
    return [{"row": r, "col": c} for r, c in sorted(out)]


def extract_true_coords_from_cells(cells):
    coords = set()
    gt_text: Dict[Tuple[int, int], str] = {}
    for cell in cells or []:
        if (bool(cell.get("is_column_header"))
                or bool(cell.get("is_projected_row_header"))
                or bool(cell.get("is_spanning"))):
            row_nums = cell.get("row_nums", []) or []
            col_nums = cell.get("column_nums", []) or []
            if row_nums and col_nums:
                try:
                    anchor = (int(row_nums[0]), int(col_nums[0]))
                    coords.add(anchor)
                    gt_text[anchor] = str(cell.get("xml_text_content") or "").strip()
                except Exception:
                    continue
    return [{"row": r, "col": c} for r, c in sorted(coords)], gt_text


def extract_type_coords_from_cells(cells) -> Dict[str, Any]:
    col_c, proj_c, span_c = set(), set(), set()
    col_n = proj_n = span_n = 0
    span_zones: Dict[Tuple[int, int], set] = {}
    for cell in cells or []:
        row_nums = cell.get("row_nums", []) or []
        col_nums = cell.get("column_nums", []) or []
        if not row_nums or not col_nums:
            continue
        try:
            anchor = (int(row_nums[0]), int(col_nums[0]))
        except Exception:
            continue
        if cell.get("is_column_header"):
            col_c.add(anchor); col_n += 1
        if cell.get("is_projected_row_header"):
            proj_c.add(anchor); proj_n += 1
        if cell.get("is_spanning"):
            span_c.add(anchor); span_n += 1
            zone = set()
            for r in row_nums:
                for c in col_nums:
                    try:
                        zone.add((int(r), int(c)))
                    except Exception:
                        pass
            if zone:
                span_zones[anchor] = zone
    return {
        "column_headers": [{"row": r, "col": c} for r, c in sorted(col_c)],
        "projected_row_headers": [{"row": r, "col": c} for r, c in sorted(proj_c)],
        "spanning": [{"row": r, "col": c} for r, c in sorted(span_c)],
        "column_header_cell_count": col_n,
        "projected_row_header_cell_count": proj_n,
        "spanning_cell_count": span_n,
        "spanning_zones": [
            {"anchor": list(a), "zone": [list(p) for p in sorted(z)]}
            for a, z in span_zones.items()
        ],
    }


def extract_true_coords_from_headers(headers):
    coords = set()
    gt_text: Dict[Tuple[int, int], str] = {}
    for h in headers or []:
        if isinstance(h, dict) and "row" in h and "col" in h:
            try:
                anchor = (int(h["row"]), int(h["col"]))
                coords.add(anchor)
                gt_text[anchor] = str(h.get("text", "")).strip()
            except Exception:
                continue
    return [{"row": r, "col": c} for r, c in sorted(coords)], gt_text


def evaluate_spanning_soft(span_zones, pred_set, table_rows=0, table_cols=0):
    if not span_zones:
        return {"spanning_soft_precision": None,
                "spanning_soft_recall": None,
                "spanning_soft_f1": None}
    if table_rows > 0 and table_cols > 0:
        pred_filtered = {(r, c) for r, c in pred_set if 0 <= r < table_rows and 0 <= c < table_cols}
    else:
        pred_filtered = pred_set
    all_span = set()
    for z in span_zones:
        for pos in z.get("zone", []):
            try:
                all_span.add((int(pos[0]), int(pos[1])))
            except Exception:
                pass
    n_true = len(span_zones)
    soft_tp = 0
    for z in span_zones:
        zone_set = set()
        for pos in z.get("zone", []):
            try:
                zone_set.add((int(pos[0]), int(pos[1])))
            except Exception:
                pass
        if pred_filtered & zone_set:
            soft_tp += 1
    soft_fp = len(pred_filtered - all_span)
    soft_p = soft_tp / (soft_tp + soft_fp) if (soft_tp + soft_fp) else 0.0
    soft_r = soft_tp / n_true if n_true else 0.0
    soft_f = 2 * soft_p * soft_r / (soft_p + soft_r) if (soft_p + soft_r) else 0.0
    return {"spanning_soft_precision": soft_p,
            "spanning_soft_recall": soft_r,
            "spanning_soft_f1": soft_f}


def evaluate_coord_sets(true_set, pred_set, table_rows=0, table_cols=0) -> Dict[str, Any]:
    support = len(true_set); pred_count = len(pred_set)
    if support == 0 and pred_count == 0:
        return dict(support=0, pred_count=0, tp=0, fp=0, fn=0,
                    precision=1.0, recall=1.0, f1=1.0, jaccard=1.0,
                    exact_match=True, partial_match=True, header_coverage=1.0)
    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    prec = tp / pred_count if pred_count else 0.0
    rec = tp / support if support else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    union = len(true_set | pred_set)
    jacc = tp / union if union else 1.0
    exact = true_set == pred_set
    partial = exact or (support > 0 and rec >= 0.5)
    return dict(support=support, pred_count=pred_count, tp=tp, fp=fp, fn=fn,
                precision=prec, recall=rec, f1=f1, jaccard=jacc,
                exact_match=exact, partial_match=partial, header_coverage=rec)


def token_f1(pred_text: str, true_text: str) -> float:
    pred_tokens = pred_text.lower().split()
    true_tokens = true_text.lower().split()
    if not pred_tokens and not true_tokens:
        return 1.0
    if not pred_tokens or not true_tokens:
        return 0.0
    pc = Counter(pred_tokens); tc = Counter(true_tokens)
    common = sum((pc & tc).values())
    if common == 0:
        return 0.0
    p = common / len(pred_tokens)
    r = common / len(true_tokens)
    return 2 * p * r / (p + r)


def _normalize_text(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"\[note\s*\d+\]", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\(note\s*\d+\)", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\[\d+\]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def text_containment(pred_text: str, true_text: str) -> float:
    pred_n = _normalize_text(pred_text)
    true_n = _normalize_text(true_text)
    if not pred_n and not true_n:
        return 1.0
    if not pred_n or not true_n:
        return 0.0
    if pred_n == true_n:
        return 1.0
    if pred_n in true_n or true_n in pred_n:
        return 1.0
    return token_f1(pred_n, true_n)


def evaluate_text_metrics(true_text_map, pred_headers, true_set, pred_set_filtered) -> Dict[str, Any]:
    empty = {
        "text_exact_match_rate": None, "text_token_f1_mean": None,
        "text_containment_mean": None, "joint_f1": None,
        "joint_precision": None, "joint_recall": None,
    }
    if not true_text_map:
        return empty
    pred_text_map: Dict[Tuple[int, int], str] = {}
    for h in pred_headers or []:
        try:
            key = (int(h["row"]), int(h["col"]))
            pred_text_map[key] = str(h.get("text", "")).strip()
        except Exception:
            continue
    tp_coords = true_set & pred_set_filtered
    exact_scores: List[float] = []
    token_scores: List[float] = []
    cont_scores: List[float] = []
    joint_tp = 0
    for coord in tp_coords:
        gt = true_text_map.get(coord, "")
        pred = pred_text_map.get(coord, "")
        gt_l = gt.lower().strip(); pred_l = pred.lower().strip()
        exact = float(gt_l == pred_l)
        exact_scores.append(exact)
        token_scores.append(token_f1(pred_l, gt_l))
        cont_scores.append(text_containment(pred, gt))
        if exact:
            joint_tp += 1
    n_tp = len(tp_coords)
    text_exact = sum(exact_scores) / n_tp if n_tp else None
    text_tf1 = sum(token_scores) / n_tp if n_tp else None
    text_cont = sum(cont_scores) / n_tp if n_tp else None
    n_pred = len(pred_set_filtered); n_true = len(true_set)
    if n_pred == 0 and n_true == 0:
        joint_p = joint_r = joint_f = 1.0
    else:
        joint_p = joint_tp / n_pred if n_pred else 0.0
        joint_r = joint_tp / n_true if n_true else 0.0
        joint_f = 2 * joint_p * joint_r / (joint_p + joint_r) if (joint_p + joint_r) else 0.0
    return {
        "text_exact_match_rate": text_exact, "text_token_f1_mean": text_tf1,
        "text_containment_mean": text_cont, "joint_f1": joint_f,
        "joint_precision": joint_p, "joint_recall": joint_r,
    }
