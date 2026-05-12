"""Matplotlib plots for Phase 2 report. All figures saved to report/output/figures/*.png."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except Exception:
    plt.style.use("ggplot")

FIGURES_DIR = Path("report/output/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig, name: str) -> Path:
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_exp01_chunking(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["chunk_size"], df["judge_faithfulness_avg"], marker="o", label="Faithfulness (mean)")
    ax.plot(df["chunk_size"], df["judge_relevance_avg"], marker="s", label="Relevance (mean)")
    ax.plot(df["chunk_size"], df["judge_completeness_avg"], marker="^", label="Completeness (mean)")
    if "f_haiku_avg" in df.columns and "f_4omini_avg" in df.columns:
        ax.plot(df["chunk_size"], df["f_haiku_avg"], marker=".", linestyle=":", color="C0", alpha=0.5, label="F (haiku)")
        ax.plot(df["chunk_size"], df["f_4omini_avg"], marker="x", linestyle=":", color="C0", alpha=0.5, label="F (4o-mini)")
    ax2 = ax.twinx()
    ax2.bar(df["chunk_size"], df["num_chunks"], width=40, alpha=0.15, color="gray", label="Chunks")
    ax2.set_ylabel("# chunks in index", color="gray")
    ax.set_xlabel("Chunk size (characters)")
    ax.set_ylabel("Judge score (1-5)")
    ax.set_title("EXP-01: Chunk size vs answer quality (dual judge mean, gpt-4o-mini, top-3)")
    ax.set_ylim(3.5, 5.2)
    ax.legend(loc="lower right", fontsize=8)
    return _save(fig, "exp01_chunking")


def plot_exp02_topk(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["top_k"], df["judge_faithfulness_avg"], marker="o", label="Faithfulness (mean)")
    ax.plot(df["top_k"], df["judge_relevance_avg"], marker="s", label="Relevance (mean)")
    ax.plot(df["top_k"], df["judge_completeness_avg"], marker="^", label="Completeness (mean)")
    if "c_haiku_avg" in df.columns and "c_4omini_avg" in df.columns:
        ax.plot(df["top_k"], df["c_haiku_avg"], marker=".", linestyle=":", color="C2", alpha=0.5, label="C (haiku)")
        ax.plot(df["top_k"], df["c_4omini_avg"], marker="x", linestyle=":", color="C2", alpha=0.5, label="C (4o-mini)")
    ax2 = ax.twinx()
    ax2.plot(df["top_k"], df["avg_input_tokens"], color="C3", marker="x", linestyle="--", label="Avg input tokens")
    ax2.set_ylabel("Avg input tokens", color="C3")
    ax.set_xlabel("top_k")
    ax.set_ylabel("Judge score (1-5)")
    ax.set_title("EXP-02: top-K vs quality (dual judge mean) and prompt size")
    ax.set_ylim(3.8, 5.2)
    ax.legend(loc="lower right", fontsize=8)
    return _save(fig, "exp02_topk")


def plot_exp03_cache_threshold(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["threshold"], df["tpr"], marker="o", label="TPR (good paraphrase hit)")
    ax.plot(df["threshold"], df["fpr"], marker="s", label="FPR (wrong-answer hit)")
    ax.set_xlabel("Cache similarity threshold (cosine)")
    ax.set_ylabel("Rate")
    ax.set_title("EXP-03: cache threshold — TPR / FPR trade-off")
    ax.set_ylim(-0.02, 1.02)
    ax.legend()
    return _save(fig, "exp03_cache_threshold")


def plot_exp04_cost_quality(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    quality = (df["faithfulness_avg"] + df["relevance_avg"] + df["completeness_avg"]) / 3
    x = df["cost_per_request_avg"]
    ax.scatter(x, quality, s=140, c="C0", edgecolors="black")
    for _, r in df.iterrows():
        q = (r["faithfulness_avg"] + r["relevance_avg"] + r["completeness_avg"]) / 3
        ax.annotate(
            r["model"].split("/")[-1],
            (r["cost_per_request_avg"], q),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Cost per request (USD, log scale)")
    ax.set_ylabel("Avg judge score (1-5)")
    ax.set_title("EXP-04: cost vs quality (Pareto)")
    ax.set_ylim(3.5, 5.2)
    return _save(fig, "exp04_pareto")


def plot_exp04_latency(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    names = [m.split("/")[-1] for m in df["model"]]
    x = range(len(names))
    width = 0.35
    ax.bar([i - width / 2 for i in x], df["ttft_p50"], width=width, label="TTFT p50")
    ax.bar([i + width / 2 for i in x], df["latency_p50"], width=width, label="Latency p50")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Milliseconds")
    ax.set_title("EXP-04: TTFT vs total latency (p50, direct OpenRouter)")
    ax.legend()
    return _save(fig, "exp04_latency")


def plot_exp05_load(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(df))
    width = 0.35
    ax.bar([i - width / 2 for i in x], df["p50_ms"], width=width, label="p50 latency")
    ax.bar([i + width / 2 for i in x], df["p95_ms"], width=width, label="p95 latency")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"c={c}" for c in df["concurrent"]])
    ax.set_ylabel("Milliseconds")
    ax.set_title("EXP-05: latency under concurrent load (Fly.io shared-cpu-1x)")
    ax.legend()
    for i, row in df.reset_index(drop=True).iterrows():
        ax.annotate(
            f"ok={int(row['success'])} | fail={int(row['fail_other'] + row['fail_429'])}",
            xy=(i, max(row['p50_ms'], row['p95_ms'])),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="dimgray",
        )
    return _save(fig, "exp05_load")


def plot_exp06_fallback(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    names = [m.split("/")[-1] for m in df["model"]]
    x = range(len(names))
    width = 0.35
    ax.bar([i - width / 2 for i in x], df["n_requests"], width=width, label="Total requests")
    ax.bar([i + width / 2 for i in x], df["fallback_count"], width=width, label="Fallback used")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=10, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("EXP-06: observed fallback frequency per model (prod logs)")
    ax.legend()
    return _save(fig, "exp06_fallback")


def plot_exp07_injection(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    cats = sorted(df["category"].unique())
    outcomes = ["blocked_at_input", "defended_at_output", "attack_succeeded", "other_error"]
    pivot = (
        df.groupby(["category", "outcome"]).size().unstack(fill_value=0).reindex(columns=outcomes, fill_value=0)
    )
    bottom = [0] * len(cats)
    colors = {"blocked_at_input": "#1f77b4", "defended_at_output": "#2ca02c", "attack_succeeded": "#d62728", "other_error": "#7f7f7f"}
    for outcome in outcomes:
        vals = [int(pivot.loc[c, outcome]) if c in pivot.index else 0 for c in cats]
        ax.bar(cats, vals, bottom=bottom, label=outcome, color=colors[outcome])
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_ylabel("Attacks (count)")
    ax.set_xticklabels(cats, rotation=15, ha="right")
    ax.set_title("EXP-07: injection attack outcomes per category (30 attacks)")
    ax.legend(loc="upper right", fontsize=8)
    return _save(fig, "exp07_injection")


def plot_exp09_judge_agreement(df: pd.DataFrame) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    per_exp = df[df["experiment"] != "ALL"].copy()
    # Bar chart: rho per (experiment × metric)
    pivot = per_exp.pivot_table(index="experiment", columns="metric", values="spearman_rho", aggfunc="first")
    pivot = pivot[["faithfulness", "relevance", "completeness"]] if all(c in pivot.columns for c in ["faithfulness", "relevance", "completeness"]) else pivot
    pivot.plot(kind="bar", ax=ax1, edgecolor="black")
    ax1.set_ylabel("Spearman ρ (Haiku vs gpt-4o-mini)")
    ax1.set_title("Inter-rater rank correlation per experiment × metric")
    ax1.set_ylim(-0.1, 1.05)
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.axhline(0.7, color="green", linewidth=0.5, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=8, loc="lower right")
    for tick in ax1.get_xticklabels():
        tick.set_rotation(0)

    # Right: exact-agreement + MAD
    overall = df[df["experiment"] == "ALL"]
    metrics = overall["metric"].tolist()
    agree = overall["agreement_exact"].tolist()
    mad = overall["mean_abs_diff"].tolist()
    x = range(len(metrics))
    width = 0.35
    ax2.bar([i - width / 2 for i in x], agree, width=width, label="Exact agreement", color="#16a34a")
    ax2b = ax2.twinx()
    ax2b.bar([i + width / 2 for i in x], mad, width=width, label="Mean |diff|", color="#dc2626")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(metrics)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Exact agreement (share)", color="#16a34a")
    ax2b.set_ylabel("Mean absolute diff (1-5 pts)", color="#dc2626")
    ax2.set_title("Overall (all 140 pairs) — exact agreement and MAD")
    ax2.legend(loc="upper left", fontsize=8)
    ax2b.legend(loc="upper right", fontsize=8)
    return _save(fig, "exp09_judge_agreement")


def plot_exp08_cost_projection(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 5))
    pivot = df.pivot_table(
        index="volume_per_day", columns="cache_hit_rate", values="cost_per_day_usd", aggfunc="first"
    )
    pivot.plot(kind="bar", ax=ax)
    ax.set_yscale("log")
    ax.set_xlabel("Requests per day")
    ax.set_ylabel("Cost per day (USD, log scale)")
    ax.set_title("EXP-08: projected daily cost vs cache hit rate")
    ax.legend(title="cache hit rate")
    return _save(fig, "exp08_cost_projection")


PLOTTERS = {
    "exp01": plot_exp01_chunking,
    "exp02": plot_exp02_topk,
    "exp03": plot_exp03_cache_threshold,
    "exp04_pareto": plot_exp04_cost_quality,
    "exp04_latency": plot_exp04_latency,
    "exp05": plot_exp05_load,
    "exp06": plot_exp06_fallback,
    "exp07": plot_exp07_injection,
    "exp08": plot_exp08_cost_projection,
    "exp09": plot_exp09_judge_agreement,
}
