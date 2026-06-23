import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
METRIC_COLS = ['f1_mean', 'f1_median', 'precision_mean', 'recall_mean', 'jaccard_mean', 'exact_match_mean', 'partial_match_mean', 'header_coverage_mean', 'support_mean', 'pred_count_mean', 'tp_mean', 'fp_mean', 'fn_mean']
DISPLAY_METRICS = [('F1', 'f1_mean'), ('Precision', 'precision_mean'), ('Recall', 'recall_mean'), ('Jaccard', 'jaccard_mean'), ('Exact', 'exact_match_mean'), ('Partial', 'partial_match_mean')]

def find_checkpoints(run_dirs: List[Path]) -> List[Tuple[Path, Path]]:
    found = []
    for rd in run_dirs:
        ckpt_dir = rd / 'checkpoints'
        if not ckpt_dir.exists():
            logging.warning(f'No checkpoints/ in {rd}, skipping')
            continue
        ckpts = [p for p in ckpt_dir.glob('checkpoint_*.json') if p.name != 'checkpoint_latest.json']
        ckpts = sorted(ckpts, key=lambda p: p.stat().st_mtime)
        if not ckpts:
            latest = ckpt_dir / 'checkpoint_latest.json'
            if latest.exists():
                found.append((rd, latest))
                continue
            logging.warning(f'No checkpoint files in {ckpt_dir}, skipping')
            continue
        found.append((rd, ckpts[-1]))
    return found

def load_checkpoint(ckpt_path: Path) -> Tuple[Dict, List[Dict]]:
    with open(ckpt_path, 'r', encoding='utf-8') as f:
        ckpt = json.load(f)
    meta = ckpt.get('metadata', {})
    records = ckpt.get('responses', []) + ckpt.get('api_failed_requests', []) + ckpt.get('parse_failed_requests', [])
    return (meta, records)

def load_all_runs(run_dirs: List[Path]) -> pd.DataFrame:
    all_rows = []
    ckpts = find_checkpoints(run_dirs)
    if not ckpts:
        logging.error('No valid checkpoints found.')
        sys.exit(1)
    for rd, ckpt_path in ckpts:
        meta, records = load_checkpoint(ckpt_path)
        logging.info(f'Loaded {len(records)} records from {rd.name} (model={meta.get('model_alias', '?')})')
        for rec in records:
            if 'model_alias' not in rec:
                rec['model_alias'] = meta.get('model_alias', rd.name.split('_')[-1])
            if 'model' not in rec:
                rec['model'] = meta.get('model', 'unknown')
            rec['_run_dir'] = str(rd)
            all_rows.append(rec)
    df = pd.DataFrame(all_rows)
    if 'pred_count' in df.columns and 'support' in df.columns:
        df['predicted_no_headers'] = (pd.to_numeric(df['pred_count'], errors='coerce').fillna(0) == 0) & (pd.to_numeric(df['support'], errors='coerce').fillna(0) > 0) & (pd.to_numeric(df.get('api_success', 1), errors='coerce').fillna(1) == 1)
    num_cols = ['f1', 'precision', 'recall', 'jaccard', 'exact_match', 'partial_match', 'header_coverage', 'support', 'pred_count', 'tp', 'fp', 'fn', 'duration_sec', 'prompt_tokens', 'completion_tokens', 'total_tokens', 'column_headers_f1', 'projected_row_headers_f1', 'spanning_f1', 'text_token_f1_mean', 'text_exact_match_rate', 'text_containment_mean', 'joint_f1', 'spanning_soft_f1', 'output_complete']
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    logging.info(f'Total records loaded: {len(df)} from {len(ckpts)} runs')
    return df

def build_task_key(r: Dict) -> str:
    return '|'.join([str(r.get('model_alias', r.get('model', '?'))), str(r.get('prompt_name', '?')), str(r.get('source_group', '?')), str(r.get('source_stem', '?')), str(r.get('table_index', '0')), str(r.get('table_format', '?'))])

def merge_runs(df: pd.DataFrame, strategy: str='best') -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df['_task_key'] = df.apply(build_task_key, axis=1)
    if strategy == 'latest' and 'timestamp' in df.columns:
        df['_ts'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.sort_values(['_task_key', '_ts'], ascending=[True, False])
        sort_cols = ['_task_key', '_ts']
    else:
        if strategy == 'latest':
            logging.warning("strategy='latest' requested but no 'timestamp' column found; falling back to 'best'.")
        df['_api_ok'] = pd.to_numeric(df.get('api_success', False), errors='coerce').fillna(0)
        df['_parse_ok'] = pd.to_numeric(df.get('parse_success', False), errors='coerce').fillna(0)
        df['_f1'] = pd.to_numeric(df.get('f1', 0), errors='coerce').fillna(0)
        df['_not_capped'] = 1 - pd.to_numeric(df.get('completion_capped', False), errors='coerce').fillna(0)
        df = df.sort_values(['_task_key', '_api_ok', '_parse_ok', '_f1', '_not_capped'], ascending=[True, False, False, False, False])
        sort_cols = ['_task_key', '_api_ok', '_parse_ok', '_f1', '_not_capped']
    before = len(df)
    df = df.drop_duplicates(subset=['_task_key'], keep='first')
    after = len(df)
    dupes = before - after
    if dupes > 0:
        logging.info(f"Merged {before} records → {after} unique tasks (strategy='{strategy}', collapsed {dupes} duplicates)")
    df = df.drop(columns=['_task_key', '_ts'] + sort_cols, errors='ignore')
    return df.reset_index(drop=True)

def agg_group(df: pd.DataFrame, group_cols: List[str], metric_prefix: str='') -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    base_metrics = ['f1', 'precision', 'recall', 'jaccard', 'exact_match', 'partial_match', 'header_coverage', 'support', 'pred_count', 'tp', 'fp', 'fn', 'duration_sec', 'completion_tokens', 'prompt_tokens', 'text_token_f1_mean', 'text_exact_match_rate', 'text_containment_mean', 'joint_f1', 'joint_precision', 'joint_recall']
    if metric_prefix:
        base_metrics = [m for m in [f'{metric_prefix}_f1', f'{metric_prefix}_precision', f'{metric_prefix}_recall'] if m in df.columns]
    rows = []
    for key, sub in df.groupby(group_cols, dropna=False):
        row: Dict[str, Any] = {}
        ks = [key] if len(group_cols) == 1 else list(key)
        for col, val in zip(group_cols, ks):
            row[col] = val
        row['n'] = len(sub)
        row['api_success'] = float(pd.to_numeric(sub.get('api_success', pd.Series(dtype=float)), errors='coerce').mean())
        row['parse_success'] = float(pd.to_numeric(sub.get('parse_success', pd.Series(dtype=float)), errors='coerce').mean())
        if 'completion_capped' in sub.columns:
            row['capped_rate'] = float(pd.to_numeric(sub['completion_capped'], errors='coerce').mean())
        for m in base_metrics:
            if m not in sub.columns:
                continue
            s = pd.to_numeric(sub[m], errors='coerce')
            if s.notna().any():
                alias = m.replace(f'{metric_prefix}_', '') if metric_prefix else m
                row[f'{alias}_mean'] = float(s.mean())
                row[f'{alias}_median'] = float(s.median())
                row[f'{alias}_std'] = float(s.std())
        rows.append(row)
    return pd.DataFrame(rows)

def build_format_consistency(df: pd.DataFrame) -> pd.DataFrame:
    if 'table_format' not in df.columns or 'parsed_headers' not in df.columns:
        return pd.DataFrame()
    json_df = df[df['table_format'] == 'json'].copy()
    html_df = df[df['table_format'] == 'html'].copy()
    key_cols = ['model_alias', 'prompt_name', 'source_group', 'source_stem', 'table_index']
    merge_cols = [c for c in key_cols if c in json_df.columns and c in html_df.columns]
    if not merge_cols:
        return pd.DataFrame()
    merged = json_df[merge_cols + ['parsed_headers']].merge(html_df[merge_cols + ['parsed_headers']], on=merge_cols, suffixes=('_json', '_html'))
    rows = []
    for _, row in merged.iterrows():
        try:
            ph_j = row['parsed_headers_json']
            ph_h = row['parsed_headers_html']
            if isinstance(ph_j, str):
                import json as _json
                ph_j = _json.loads(ph_j)
            if isinstance(ph_h, str):
                import json as _json
                ph_h = _json.loads(ph_h)
            set_j = {(h['row'], h['col']) for h in ph_j or []}
            set_h = {(h['row'], h['col']) for h in ph_h or []}
            union = set_j | set_h
            inter = set_j & set_h
            jacc = len(inter) / len(union) if union else 1.0
        except Exception:
            jacc = None
        entry = {c: row[c] for c in merge_cols}
        entry['format_consistency_jaccard'] = jacc
        rows.append(entry)
    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows)
    group_cols = [c for c in ['model_alias', 'prompt_name'] if c in result.columns]
    if not group_cols:
        return result
    summary_rows = []
    for key, sub in result.groupby(group_cols, dropna=False):
        r: dict = {}
        ks = [key] if len(group_cols) == 1 else list(key)
        for col, val in zip(group_cols, ks):
            r[col] = val
        s = pd.to_numeric(sub['format_consistency_jaccard'], errors='coerce').dropna()
        r['n_paired'] = len(s)
        r['consistency_mean'] = float(s.mean()) if len(s) else None
        r['consistency_median'] = float(s.median()) if len(s) else None
        r['consistency_low_pct'] = float((s < 0.5).mean()) if len(s) else None
        summary_rows.append(r)
    return pd.DataFrame(summary_rows)

def bootstrap_ci(values, n_boot: int=2000, alpha: float=0.05, seed: int=0):
    vals = np.asarray([v for v in values if v == v], dtype=float)
    if len(vals) == 0:
        return (None, None, None)
    if len(vals) == 1:
        return (float(vals[0]), float(vals[0]), float(vals[0]))
    rng = np.random.default_rng(seed)
    means = rng.choice(vals, size=(n_boot, len(vals)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(vals.mean()), float(lo), float(hi))

def build_model_quality(df: pd.DataFrame, metric: str='f1') -> pd.DataFrame:
    if 'model_alias' not in df.columns or metric not in df.columns:
        return pd.DataFrame()
    rows = []
    for model, sub in df.groupby('model_alias', dropna=False):
        f1 = pd.to_numeric(sub[metric], errors='coerce')
        api_ok = sub['api_success'].astype(str).str.lower().isin(['true', '1', '1.0']) if 'api_success' in sub.columns else pd.Series([True] * len(sub))
        parse_ok = sub['parse_success'].astype(str).str.lower().isin(['true', '1', '1.0']) if 'parse_success' in sub.columns else pd.Series([True] * len(sub))
        support = pd.to_numeric(sub['support'], errors='coerce') if 'support' in sub.columns else pd.Series([1] * len(sub))
        pred_count = pd.to_numeric(sub['pred_count'], errors='coerce') if 'pred_count' in sub.columns else pd.Series([0] * len(sub))
        has_hdr = support > 0

        def m(mask=None):
            s = f1 if mask is None else f1[mask]
            s = s.dropna()
            return float(s.mean()) if len(s) else None
        rows.append({'model_alias': model, 'n': len(sub),
                     'api_success_rate': float(api_ok.mean()),
                     'parse_success_rate': float(parse_ok.mean()),
                     'n_with_headers': int(has_hdr.sum()),
                     'trivial_rate': float(((support == 0) & (pred_count == 0)).mean()),
                     f'{metric}_all': m(),
                     f'{metric}_answered': m(parse_ok),
                     f'{metric}_with_headers': m(parse_ok & has_hdr)})
    return pd.DataFrame(rows)

def build_model_ci(df: pd.DataFrame, metric: str='f1') -> pd.DataFrame:
    if 'model_alias' not in df.columns or metric not in df.columns:
        return pd.DataFrame()
    rows = []
    for model, sub in df.groupby('model_alias', dropna=False):
        mean, lo, hi = bootstrap_ci(pd.to_numeric(sub[metric], errors='coerce').values)
        rows.append({'model_alias': model, 'n': len(sub), f'{metric}_mean': mean, f'{metric}_ci_low': lo, f'{metric}_ci_high': hi})
    return pd.DataFrame(rows)

def build_paired_model_comparison(df: pd.DataFrame, metric: str='f1') -> pd.DataFrame:
    needed = {'model_alias', 'prompt_name', 'source_group', 'source_stem', 'table_index', 'table_format'}
    if not needed.issubset(df.columns) or metric not in df.columns:
        return pd.DataFrame()
    d = df.copy()
    d['_pair_key'] = d['prompt_name'].astype(str) + '|' + d['source_group'].astype(str) + '|' + d['source_stem'].astype(str) + '|' + d['table_index'].astype(str) + '|' + d['table_format'].astype(str)
    wide = d.pivot_table(index='_pair_key', columns='model_alias', values=metric, aggfunc='mean')
    models = list(wide.columns)
    rows = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            a, b = (models[i], models[j])
            sub = wide[[a, b]].dropna()
            if len(sub) == 0:
                continue
            md, lo, hi = bootstrap_ci((sub[a] - sub[b]).values)
            rows.append({'model_a': a, 'model_b': b, 'n_paired': len(sub), f'{metric}_a_mean': float(sub[a].mean()), f'{metric}_b_mean': float(sub[b].mean()), 'delta_mean': md, 'delta_ci_low': lo, 'delta_ci_high': hi, 'significant_95': lo is not None and (lo > 0 or hi < 0)})
    return pd.DataFrame(rows)

def build_paired_format_comparison(df: pd.DataFrame, metric: str='f1') -> pd.DataFrame:
    needed = {'model_alias', 'prompt_name', 'source_group', 'source_stem', 'table_index', 'table_format'}
    if not needed.issubset(df.columns) or metric not in df.columns:
        return pd.DataFrame()
    d = df.copy()
    d = d[d['table_format'].isin(['json', 'html'])]
    if 'parse_success' in d.columns:
        d = d[d['parse_success'].astype(str).str.lower().isin(['true', '1', '1.0'])]
    if d.empty:
        return pd.DataFrame()
    d['_task_key'] = d['prompt_name'].astype(str) + '|' + d['source_group'].astype(str) + '|' + d['source_stem'].astype(str) + '|' + d['table_index'].astype(str)
    rows = []
    for model, sub in d.groupby('model_alias', dropna=False):
        wide = sub.pivot_table(index='_task_key', columns='table_format', values=metric, aggfunc='mean')
        if 'json' not in wide.columns or 'html' not in wide.columns:
            continue
        paired = wide[['json', 'html']].dropna()
        n_json_total = int(wide['json'].notna().sum())
        n_html_total = int(wide['html'].notna().sum())
        if len(paired) == 0:
            rows.append({'model_alias': model, 'n_paired': 0, 'n_json_total': n_json_total, 'n_html_total': n_html_total, f'{metric}_json_mean': None, f'{metric}_html_mean': None, 'delta_json_minus_html_mean': None, 'delta_ci_low': None, 'delta_ci_high': None, 'significant_95': False})
            continue
        md, lo, hi = bootstrap_ci((paired['json'] - paired['html']).values)
        rows.append({'model_alias': model, 'n_paired': len(paired), 'n_json_total': n_json_total, 'n_html_total': n_html_total, f'{metric}_json_mean': float(paired['json'].mean()), f'{metric}_html_mean': float(paired['html'].mean()), 'delta_json_minus_html_mean': md, 'delta_ci_low': lo, 'delta_ci_high': hi, 'significant_95': lo is not None and (lo > 0 or hi < 0)})
    return pd.DataFrame(rows)

def build_continuation_stats(df: pd.DataFrame) -> pd.DataFrame:
    if 'model_alias' not in df.columns:
        return pd.DataFrame()
    rows = []
    for model, sub in df.groupby('model_alias', dropna=False):
        n = len(sub)
        row = {'model_alias': model, 'n': n}
        capped = pd.to_numeric(sub.get('completion_capped', 0), errors='coerce').fillna(0)
        row['capped_rate'] = float(capped.mean()) if n else None
        if 'output_complete' in sub.columns:
            oc = pd.to_numeric(sub['output_complete'], errors='coerce').fillna(0)
            row['output_complete_rate'] = float(oc.mean())
        if 'continuation_used' in sub.columns:
            cu = pd.to_numeric(sub['continuation_used'], errors='coerce').fillna(0)
            row['continuation_used_rate'] = float(cu.mean())
            cont_rows = sub[cu == 1]
            if len(cont_rows) and 'f1' in cont_rows.columns:
                row['continuation_f1_mean'] = float(pd.to_numeric(cont_rows['f1'], errors='coerce').mean())
                row['continuation_zero_f1_rate'] = float((pd.to_numeric(cont_rows['f1'], errors='coerce') == 0).mean())
        rows.append(row)
    return pd.DataFrame(rows)

def build_model_summary(df: pd.DataFrame) -> pd.DataFrame:
    return agg_group(df, ['model_alias'])

def build_model_prompt(df: pd.DataFrame) -> pd.DataFrame:
    return agg_group(df, ['model_alias', 'prompt_name'])

def build_model_format(df: pd.DataFrame) -> pd.DataFrame:
    return agg_group(df, ['model_alias', 'table_format'])

def _add_strategy_column(df: pd.DataFrame) -> pd.DataFrame:
    if 'prompt_name' not in df.columns:
        return df
    df = df.copy()

    def get_strategy(p: str) -> str:
        p = str(p).lower()
        if 'domain' in p:
            return 'domain'
        if '_max' in p or 'max' in p:
            return 'max'
        if '_min' in p or 'min' in p:
            return 'min'
        return 'other'
    df['strategy'] = df['prompt_name'].apply(get_strategy)
    return df

def build_model_strategy(df: pd.DataFrame) -> pd.DataFrame:
    df2 = _add_strategy_column(df)
    if 'strategy' not in df2.columns:
        return pd.DataFrame()
    return agg_group(df2, ['model_alias', 'strategy'])

def build_model_prompt_format(df: pd.DataFrame) -> pd.DataFrame:
    return agg_group(df, ['model_alias', 'prompt_name', 'table_format'])

def build_model_source(df: pd.DataFrame) -> pd.DataFrame:
    return agg_group(df, ['model_alias', 'source_group'])

def build_model_size(df: pd.DataFrame) -> pd.DataFrame:
    return agg_group(df, ['model_alias', 'table_rows_bin'])

def build_model_type(df: pd.DataFrame) -> pd.DataFrame:
    if 'has_type_info' not in df.columns:
        return pd.DataFrame()
    sub = df[df['has_type_info'] == True].copy()
    if sub.empty:
        return pd.DataFrame()
    frames = []
    for ht in ['column_headers', 'projected_row_headers', 'spanning']:
        col_f1 = f'{ht}_f1'
        if col_f1 not in sub.columns:
            continue
        tmp = agg_group(sub, ['model_alias'], metric_prefix=ht)
        if not tmp.empty:
            tmp.insert(1, 'header_type', ht)
            frames.append(tmp)
    if 'spanning_soft_f1' in sub.columns and pd.to_numeric(sub['spanning_soft_f1'], errors='coerce').notna().any():
        tmp = agg_group(sub, ['model_alias'], metric_prefix='spanning_soft')
        if not tmp.empty:
            tmp.insert(1, 'header_type', 'spanning_soft')
            frames.append(tmp)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def build_pivot(df: pd.DataFrame, index: str, columns: str, values: str='f1_mean') -> pd.DataFrame:
    agg = agg_group(df, [index, columns])
    if agg.empty or values not in agg.columns:
        return pd.DataFrame()
    pivot = agg.pivot(index=index, columns=columns, values=values)
    pivot.columns.name = None
    pivot = pivot.round(3)
    return pivot

def write_summary(df: pd.DataFrame, out_path: Path, run_dirs: List[Path], views: dict=None):
    models = sorted(df['model_alias'].dropna().unique())
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('CROSS-MODEL COMPARISON SUMMARY\n')
        f.write('=' * 80 + '\n')
        f.write(f'Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n')
        f.write(f'Runs analysed: {len(run_dirs)}\n')
        f.write(f'Models: {', '.join(models)}\n')
        f.write(f'Total records: {len(df)}\n\n')
        f.write('OVERALL METRICS BY MODEL\n' + '-' * 40 + '\n')
        overall = build_model_summary(df)
        if not overall.empty:
            cols = ['model_alias', 'n', 'api_success', 'parse_success'] + [k for m, k in DISPLAY_METRICS if k in overall.columns]
            f.write(overall[[c for c in cols if c in overall.columns]].to_string(index=False) + '\n\n')
        f.write('JSON vs HTML — F1 BY MODEL\n' + '-' * 40 + '\n')
        try:
            pivot_fmt = build_pivot(df, 'model_alias', 'table_format', 'f1_mean')
            if not pivot_fmt.empty:
                f.write(pivot_fmt.to_string() + '\n\n')
        except Exception as e:
            f.write(f'  (pivot error: {e})\n\n')
        f.write('BEST PROMPT PER MODEL (by F1)\n' + '-' * 40 + '\n')
        mp = build_model_prompt(df)
        if not mp.empty and 'f1_mean' in mp.columns:
            best = mp.loc[mp.groupby('model_alias')['f1_mean'].idxmax()]
            for _, row in best.iterrows():
                f.write(f'  {row['model_alias']:15s}  prompt={row['prompt_name']:25s}  F1={row.get('f1_mean', 0):.3f}  P={row.get('precision_mean', 0):.3f}  R={row.get('recall_mean', 0):.3f}\n')
            f.write('\n')
        f.write('HEADER TYPE BREAKDOWN (pubtables, F1 mean)\n' + '-' * 40 + '\n')
        mt = build_model_type(df)
        if not mt.empty and 'f1_mean' in mt.columns:
            f.write(mt[['model_alias', 'header_type', 'n', 'f1_mean', 'f1_median']].to_string(index=False) + '\n\n')
        f.write('BY TABLE SIZE (F1 mean)\n' + '-' * 40 + '\n')
        try:
            pivot_size = build_pivot(df, 'model_alias', 'table_rows_bin', 'f1_mean')
            if not pivot_size.empty:
                size_order = ['<=10', '11-25', '>25']
                ordered = [c for c in size_order if c in pivot_size.columns]
                f.write(pivot_size[ordered].to_string() + '\n\n')
        except Exception as e:
            f.write(f'  (pivot error: {e})\n\n')
        if 'completion_capped' in df.columns:
            f.write('TRUNCATION ANALYSIS (completion_capped)\n' + '-' * 40 + '\n')
            capped_df = df[df['completion_capped'] == True]
            uncapped_df = df[df['completion_capped'] == False]
            for model in models:
                cap = capped_df[capped_df['model_alias'] == model]
                unc = uncapped_df[uncapped_df['model_alias'] == model]
                cap_f1 = cap['f1'].mean() if len(cap) else float('nan')
                unc_f1 = unc['f1'].mean() if len(unc) else float('nan')
                f.write(f'  {model:15s}  capped={len(cap)}/{len(df[df['model_alias'] == model])} ({len(cap) / max(1, len(df[df['model_alias'] == model])):.0%})  F1_capped={cap_f1:.3f}  F1_full={unc_f1:.3f}\n')
            f.write('\n')
        strat = views.get('comparison_by_model_strategy', pd.DataFrame()) if views else pd.DataFrame()
        if not strat.empty:
            f.write('STRATEGY BREAKDOWN (domain/max/min)\n' + '-' * 40 + '\n')
            s_cols = [c for c in ['strategy', 'n', 'f1_mean', 'precision_mean', 'recall_mean', 'capped_rate'] if c in strat.columns]
            f.write(strat[s_cols].to_string(index=False) + '\n\n')
        cons = views.get('format_consistency', pd.DataFrame()) if views else pd.DataFrame()
        if not cons.empty and 'consistency_mean' in cons.columns:
            f.write('FORMAT CONSISTENCY (JSON vs HTML Jaccard)\n' + '-' * 40 + '\n')
            f.write(cons.to_string(index=False) + '\n\n')
        if 'predicted_no_headers' in df.columns:
            f.write('PREDICTED NO HEADERS (false negatives on non-empty tables)\n' + '-' * 40 + '\n')
            for model in models:
                sub = df[df['model_alias'] == model]
                no_h = sub[sub['predicted_no_headers'] == True]
                f.write(f'  {model}: {len(no_h)}/{len(sub)} ({len(no_h) / max(1, len(sub)):.1%})\n')
            f.write('\n')
    logging.info(f'Summary → {out_path}')

def save_views(views: Dict[str, pd.DataFrame], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, vdf in views.items():
        if vdf.empty:
            continue
        vdf.to_csv(out_dir / f'{name}.csv', index=False, encoding='utf-8-sig')
        logging.info(f'  Saved {name}.csv ({len(vdf)} rows)')
    try:
        with pd.ExcelWriter(out_dir / 'comparison_all.xlsx', engine='openpyxl') as w:
            for name, vdf in views.items():
                if not vdf.empty:
                    vdf.to_excel(w, sheet_name=name[:31], index=False)
        logging.info('  Saved comparison_all.xlsx')
    except Exception as e:
        logging.warning(f'XLSX save failed: {e}')

def save_flat_responses(df: pd.DataFrame, out_dir: Path):
    slim = df.drop(columns=[c for c in ['raw_response', 'parsed_headers', 'tokens_used', '_run_dir', 'true_headers_raw', 'true_headers_1based', 'true_headers_by_type_raw'] if c in df.columns], errors='ignore')
    slim.to_csv(out_dir / 'all_responses.csv', index=False, encoding='utf-8-sig')
    logging.info(f'  Saved all_responses.csv ({len(slim)} rows)')
    try:
        slim.to_parquet(out_dir / 'all_responses.parquet', index=False)
        logging.info('  Saved all_responses.parquet')
    except Exception:
        pass

def discover_runs(results_dir: Path) -> List[Path]:
    runs = sorted([p for p in results_dir.iterdir() if p.is_dir() and p.name.startswith('run_')])
    logging.info(f'Discovered {len(runs)} run directories in {results_dir}')
    return runs

def main():
    parser = argparse.ArgumentParser(description='Cross-model analysis for table header detection experiment', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('run_dirs', nargs='*', metavar='RUN_DIR', help='One or more run_* directories to compare. If omitted, uses --results-dir to discover all runs.')
    parser.add_argument('--results-dir', default=None, metavar='DIR', help='Root results directory. Auto-discovers all run_* subdirs.')
    parser.add_argument('--output-dir', default='analysis', metavar='DIR', help='Where to write comparison outputs.')
    parser.add_argument('--models', nargs='+', default=None, metavar='ALIAS', help='Filter to specific model aliases (e.g. qwen30b llama8b).')
    parser.add_argument('--merge-strategy', choices=['best', 'latest'], default='best', help="How to collapse duplicate tasks across runs. 'best' = highest-quality result wins (replaces failed/capped originals; can bias upward if successful tasks repeat). 'latest' = most recent result wins (unbiased).")
    parser.add_argument('--metric', default='f1', help='Metric used for per-model CIs and paired comparison.')
    args = parser.parse_args()
    run_dirs: List[Path] = []
    if args.run_dirs:
        for rd in args.run_dirs:
            p = Path(rd)
            if not p.exists():
                logging.warning(f'Run dir not found: {p}')
            else:
                run_dirs.append(p)
    elif args.results_dir:
        run_dirs = discover_runs(Path(args.results_dir))
    else:
        parser.error('Provide run dirs as arguments or use --results-dir.')
    if not run_dirs:
        logging.error('No run directories found.')
        sys.exit(1)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df_raw = load_all_runs(run_dirs)
    df = merge_runs(df_raw, strategy=args.merge_strategy)
    logging.info(f'After merge: {len(df)} unique tasks (from {len(df_raw)} total records across {len(run_dirs)} runs)')
    if args.models:
        df = df[df['model_alias'].isin(args.models)]
        logging.info(f'Filtered to models: {args.models} → {len(df)} records')
    if df.empty:
        logging.error('No records after filtering.')
        sys.exit(1)
    views = {'comparison_by_model': build_model_summary(df), 'comparison_by_model_prompt': build_model_prompt(df), 'comparison_by_model_format': build_model_format(df), 'comparison_by_model_prompt_format': build_model_prompt_format(df), 'comparison_by_model_source': build_model_source(df), 'comparison_by_model_strategy': build_model_strategy(df), 'format_consistency': build_format_consistency(df), 'comparison_by_model_size': build_model_size(df), 'comparison_by_model_type': build_model_type(df), 'model_ci': build_model_ci(df, metric=args.metric), 'model_quality': build_model_quality(df, metric=args.metric), 'paired_model_comparison': build_paired_model_comparison(df, metric=args.metric), 'paired_format_comparison': build_paired_format_comparison(df, metric=args.metric), 'continuation_stats': build_continuation_stats(df)}
    try:
        views['pivot_model_x_prompt_F1'] = build_pivot(df, 'model_alias', 'prompt_name')
        views['pivot_model_x_format_F1'] = build_pivot(df, 'model_alias', 'table_format')
        views['pivot_model_x_size_F1'] = build_pivot(df, 'model_alias', 'table_rows_bin')
    except Exception as e:
        logging.warning(f'Pivot build error: {e}')
    save_views(views, out_dir)
    save_flat_responses(df, out_dir)
    write_summary(df, out_dir / 'summary.txt', run_dirs, views=views)
    print('\n' + '=' * 70)
    print('QUICK OVERVIEW — F1 by model')
    print('=' * 70)
    overall = views['comparison_by_model']
    if not overall.empty and 'f1_mean' in overall.columns:
        for _, row in overall.sort_values('f1_mean', ascending=False).iterrows():
            alias = str(row.get('model_alias', '?'))
            print(f'  {alias:<15}  n={int(row.get('n', 0)):5d}  F1={float(row.get('f1_mean', 0)):.3f}  P={float(row.get('precision_mean', 0)):.3f}  R={float(row.get('recall_mean', 0)):.3f}  Exact={float(row.get('exact_match_mean', 0)):.3f}')
    if 'comparison_by_model_format' in views:
        fmt_df = views['comparison_by_model_format']
        if not fmt_df.empty and 'f1_mean' in fmt_df.columns:
            print('\n' + '-' * 70)
            print('F1 by model × format')
            print('-' * 70)
            for _, row in fmt_df.iterrows():
                alias = str(row.get('model_alias', '?'))
                fmt = str(row.get('table_format', '?'))
                print(f'  {alias:<15}  [{fmt:<4}]  F1={float(row.get('f1_mean', 0)):.3f}  P={float(row.get('precision_mean', 0)):.3f}  R={float(row.get('recall_mean', 0)):.3f}')
    paired = views.get('paired_model_comparison', pd.DataFrame())
    if not paired.empty:
        print('\n' + '-' * 70)
        print(f'PAIRED COMPARISON ({args.metric}, bootstrap 95% CI of difference)')
        print('-' * 70)
        for _, row in paired.iterrows():
            sig = '  *SIGNIFICANT*' if row.get('significant_95') else ''
            dlo, dhi = (row.get('delta_ci_low'), row.get('delta_ci_high'))
            ci = f'[{dlo:+.3f}, {dhi:+.3f}]' if dlo is not None and dhi is not None else '[n/a]'
            print(f'  {str(row['model_a']):<12} vs {str(row['model_b']):<12} n={int(row['n_paired']):4d}  Δ={float(row.get('delta_mean', 0)):+.3f} {ci}{sig}')
    mq = views.get('model_quality', pd.DataFrame())
    if not mq.empty:
        print('\n' + '-' * 70)
        print('HONEST QUALITY CUTS (F1: all / answered / on tables-with-headers)')
        print('-' * 70)
        for _, row in mq.iterrows():
            alias = str(row.get('model_alias', '?'))
            def g(k):
                v = row.get(k)
                return float(v) if v is not None and v == v else float('nan')
            print(f'  {alias:<15}  api={g("api_success_rate"):.0%} parse={g("parse_success_rate"):.0%} '
                  f'trivial={g("trivial_rate"):.0%} | F1 all={g("f1_all"):.3f} '
                  f'answered={g("f1_answered"):.3f} with_headers={g("f1_with_headers"):.3f}')
    print(f'\nFull analysis → {out_dir}/')
if __name__ == '__main__':
    main()
