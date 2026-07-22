import argparse
import json
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N_BOOT = 10000
STRAT = ["zero", "fewshot", "reasoning"]
DS = {"domain": "DOMAIN", "max": "VGR", "min": "NORM"}


def split_prompt(name):
    m = re.match(r"(zero|fewshot|reasoning)_(domain|max|min)$", str(name))
    return (m.group(1), m.group(2)) if m else (None, None)


def load(path, alias):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    df = pd.DataFrame(rows)
    df["model"] = alias
    df[["strategy", "dataset"]] = df["prompt_name"].apply(
        lambda x: pd.Series(split_prompt(x)))
    df = df.dropna(subset=["strategy", "dataset"])
    for c in ["f1", "precision", "recall", "support"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ok"] = (df["status"] == "ok") & df["api_success"].astype(bool)
    df["task"] = (df["source_group"] + "|" + df["source_stem"] + "|"
                  + df["table_index"].astype(str) + "|" + df["prompt_name"]
                  + "|" + df["table_format"])
    return df


def boot_ci(x, n_boot=N_BOOT, alpha=0.05):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    idx = RNG.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return (float(x.mean()),
            float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


def paired_boot(a, b, n_boot=N_BOOT, alpha=0.05):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    d = d[~np.isnan(d)]
    if len(d) == 0:
        return dict(n=0, diff=np.nan, lo=np.nan, hi=np.nan, p=np.nan)
    idx = RNG.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    p_boot = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return dict(n=len(d), diff=float(d.mean()), lo=lo, hi=hi,
                p=float(min(1.0, p_boot)))


def wilcoxon(a, b):
    try:
        from scipy.stats import wilcoxon as w
        d = np.asarray(a, float) - np.asarray(b, float)
        d = d[~np.isnan(d)]
        if len(d) < 10 or np.allclose(d, 0):
            return np.nan
        return float(w(d).pvalue)
    except Exception:
        return np.nan


def cohen_dz(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[~np.isnan(d)]
    s = d.std(ddof=1)
    return float(d.mean() / s) if s > 0 else np.nan


def stars(p):
    if np.isnan(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def section(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def ci_table(df, group, metric="f1", scored_only=True):
    d = df[df["ok"]]
    if scored_only:
        d = d[d["support"] > 0]
    rows = []
    for key, g in d.groupby(group):
        m, lo, hi = boot_ci(g[metric])
        rows.append({group if isinstance(group, str) else "group": key,
                     "n": len(g), "mean": round(m, 4),
                     "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                     "ci_width": round(hi - lo, 4)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True, metavar="ALIAS=PATH")
    ap.add_argument("--out", default="stats")
    ap.add_argument("--metric", default="f1")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    metric = args.metric

    frames = []
    for spec in args.results:
        alias, path = spec.split("=", 1)
        frames.append(load(path, alias))
        print(f"loaded {alias}: {len(frames[-1])} tasks from {path}")
    ALL = pd.concat(frames, ignore_index=True)
    models = list(dict.fromkeys(ALL["model"]))

    section("1. OVERALL WITH 95% BOOTSTRAP CI (scored tasks: ok & support>0)")
    t = ci_table(ALL, "model", metric)
    print(t.to_string(index=False))
    t.to_csv(out / "ci_overall.csv", index=False)

    section("2. CI BY PROMPT CONFIGURATION")
    d = ALL[ALL["ok"] & (ALL["support"] > 0)].copy()
    rows = []
    for (mo, s, ds), g in d.groupby(["model", "strategy", "dataset"]):
        m, lo, hi = boot_ci(g[metric])
        rows.append(dict(model=mo, strategy=s, dataset=DS[ds], n=len(g),
                         mean=round(m, 4), ci_low=round(lo, 4),
                         ci_high=round(hi, 4)))
    t = pd.DataFrame(rows).sort_values(["model", "dataset", "strategy"])
    print(t.to_string(index=False))
    t.to_csv(out / "ci_by_config.csv", index=False)

    section("3. PAIRED: JSON vs HTML (same table, same prompt, both succeeded)")
    rows = []
    for mo, g in d.groupby("model"):
        piv = g.pivot_table(index=["source_group", "source_stem", "table_index",
                                   "prompt_name"],
                            columns="table_format", values=metric)
        piv = piv.dropna(subset=[c for c in ("json", "html") if c in piv])
        if not {"json", "html"} <= set(piv.columns):
            continue
        r = paired_boot(piv["json"], piv["html"])
        rows.append(dict(model=mo, n_pairs=r["n"],
                         json=round(piv["json"].mean(), 4),
                         html=round(piv["html"].mean(), 4),
                         diff=round(r["diff"], 4),
                         ci_low=round(r["lo"], 4), ci_high=round(r["hi"], 4),
                         p_boot=round(r["p"], 5),
                         p_wilcoxon=round(wilcoxon(piv["json"], piv["html"]), 5),
                         d_z=round(cohen_dz(piv["json"], piv["html"]), 3),
                         sig=stars(r["p"])))
    t = pd.DataFrame(rows)
    print(t.to_string(index=False))
    t.to_csv(out / "paired_json_vs_html.csv", index=False)

    section("4. PAIRED: PROMPT STRATEGIES (same table+format, within dataset)")
    rows = []
    for (mo, ds), g in d.groupby(["model", "dataset"]):
        piv = g.pivot_table(index=["source_stem", "table_index", "table_format"],
                            columns="strategy", values=metric)
        for a, b in combinations(STRAT, 2):
            if a not in piv or b not in piv:
                continue
            sub = piv[[a, b]].dropna()
            if len(sub) < 10:
                continue
            r = paired_boot(sub[a], sub[b])
            rows.append(dict(model=mo, dataset=DS[ds], comparison=f"{a} - {b}",
                             n_pairs=r["n"], mean_a=round(sub[a].mean(), 4),
                             mean_b=round(sub[b].mean(), 4),
                             diff=round(r["diff"], 4), ci_low=round(r["lo"], 4),
                             ci_high=round(r["hi"], 4), p_boot=round(r["p"], 5),
                             p_wilcoxon=round(wilcoxon(sub[a], sub[b]), 5),
                             d_z=round(cohen_dz(sub[a], sub[b]), 3),
                             sig=stars(r["p"])))
    t = pd.DataFrame(rows)
    if len(t):
        print(t.to_string(index=False))
        t.to_csv(out / "paired_strategies.csv", index=False)

    if len(models) > 1:
        section("5. PAIRED: MODEL vs MODEL (identical tasks, both succeeded)")
        rows = []
        for a, b in combinations(models, 2):
            da = ALL[(ALL.model == a) & ALL.ok & (ALL.support > 0)][["task", metric]]
            db = ALL[(ALL.model == b) & ALL.ok & (ALL.support > 0)][["task", metric]]
            mg = da.merge(db, on="task", suffixes=("_a", "_b"))
            if len(mg) < 10:
                continue
            r = paired_boot(mg[f"{metric}_a"], mg[f"{metric}_b"])
            rows.append(dict(model_a=a, model_b=b, n_pairs=r["n"],
                             mean_a=round(mg[f"{metric}_a"].mean(), 4),
                             mean_b=round(mg[f"{metric}_b"].mean(), 4),
                             diff=round(r["diff"], 4), ci_low=round(r["lo"], 4),
                             ci_high=round(r["hi"], 4), p_boot=round(r["p"], 5),
                             p_wilcoxon=round(wilcoxon(mg[f"{metric}_a"],
                                                       mg[f"{metric}_b"]), 5),
                             d_z=round(cohen_dz(mg[f"{metric}_a"],
                                                mg[f"{metric}_b"]), 3),
                             sig=stars(r["p"])))
        t = pd.DataFrame(rows)
        if len(t):
            print(t.to_string(index=False))
            t.to_csv(out / "paired_models.csv", index=False)

        section("6. PAIRED MODEL DIFFERENCE BY DATASET")
        rows = []
        for a, b in combinations(models, 2):
            for ds in ["domain", "max", "min"]:
                da = ALL[(ALL.model == a) & ALL.ok & (ALL.support > 0)
                         & (ALL.dataset == ds)][["task", metric]]
                db = ALL[(ALL.model == b) & ALL.ok & (ALL.support > 0)
                         & (ALL.dataset == ds)][["task", metric]]
                mg = da.merge(db, on="task", suffixes=("_a", "_b"))
                if len(mg) < 10:
                    continue
                r = paired_boot(mg[f"{metric}_a"], mg[f"{metric}_b"])
                rows.append(dict(model_a=a, model_b=b, dataset=DS[ds],
                                 n_pairs=r["n"], diff=round(r["diff"], 4),
                                 ci_low=round(r["lo"], 4),
                                 ci_high=round(r["hi"], 4),
                                 p_boot=round(r["p"], 5), sig=stars(r["p"])))
        t = pd.DataFrame(rows)
        if len(t):
            print(t.to_string(index=False))
            t.to_csv(out / "paired_models_by_dataset.csv", index=False)

    section("7. RELIABILITY / COVERAGE")
    rows = []
    for mo, g in ALL.groupby("model"):
        sc = g[g["ok"] & (g["support"] > 0)]
        rows.append(dict(model=mo, tasks=len(g),
                         api_ok=round(g["ok"].mean(), 4),
                         scored=len(sc),
                         trivial_no_headers=int((g["support"] == 0).sum()),
                         skipped=int((g.get("error_type", pd.Series(dtype=str))
                                      == "skipped_too_large").sum())))
    t = pd.DataFrame(rows)
    print(t.to_string(index=False))
    t.to_csv(out / "reliability.csv", index=False)

    print(f"\nCSV files written to {out.resolve()}")
    print("\nNOTE: p_boot is a two-sided bootstrap p-value; p_wilcoxon is the "
          "non-parametric signed-rank test. d_z is Cohen's d for paired data.")


if __name__ == "__main__":
    main()
