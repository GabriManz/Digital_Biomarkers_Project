"""Phase 5 — Biomarker analysis: CAS rates, statistics, and visualisations.

Exports:
- compute_cas_rates()
- run_normality_test()
- run_pre_post_test()
- run_between_group_test()
- cohen_d()
- generate_all_figures()
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns

matplotlib.use("Agg")  # non-interactive backend for saving

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BD_PRE: int = 1
BD_POST: int = 2

ALPHA: float = 0.05          # significance threshold
DPI: int = 300               # figure resolution

FIG_BOXPLOT: str = "boxplot_cas_pre_post.png"
FIG_ROC: str = "roc_curve_biomarker.png"
FIG_BARPLOT: str = "barplot_cas_per_subject.png"
FIG_HEATMAP: str = "heatmap_cas_subject_manoeuvre.png"


# ---------------------------------------------------------------------------
# Public API — CAS rate computation
# ---------------------------------------------------------------------------


def compute_cas_rates(
    cas_labels: np.ndarray,
    v_subject: np.ndarray,
    v_bd: np.ndarray,
) -> pd.DataFrame:
    """Compute per-subject CAS rates pre/post bronchodilator.

    Args:
        cas_labels: Binary (N,) array — 1=CAS, 0=normal.
        v_subject: (N,) int array of subject IDs.
        v_bd: (N,) int array — 1=pre-BD, 2=post-BD.

    Returns:
        DataFrame with columns: subject_id, cas_rate_pre, cas_rate_post, delta_cas.
    """
    subject_ids = np.unique(v_subject)
    rows = []
    for sid in subject_ids:
        pre_mask = (v_subject == sid) & (v_bd == BD_PRE)
        post_mask = (v_subject == sid) & (v_bd == BD_POST)

        n_pre = int(np.sum(pre_mask))
        n_post = int(np.sum(post_mask))

        rate_pre = float(np.sum(cas_labels[pre_mask]) / n_pre) if n_pre > 0 else float("nan")
        rate_post = float(np.sum(cas_labels[post_mask]) / n_post) if n_post > 0 else float("nan")
        delta = rate_pre - rate_post if np.isfinite(rate_pre) and np.isfinite(rate_post) else float("nan")

        rows.append({
            "subject_id": int(sid),
            "cas_rate_pre": rate_pre,
            "cas_rate_post": rate_post,
            "delta_cas": delta,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Public API — Statistical tests
# ---------------------------------------------------------------------------


def run_normality_test(data: np.ndarray) -> scipy.stats.ShapiroResult:
    """Run Shapiro-Wilk normality test.

    Args:
        data: 1D array of values.

    Returns:
        scipy.stats.ShapiroResult with .statistic and .pvalue.

    Raises:
        ValueError: If data has fewer than 3 elements.
    """
    if len(data) < 3:
        raise ValueError(f"Need at least 3 data points for Shapiro-Wilk. Got {len(data)}.")
    return scipy.stats.shapiro(data)


def run_pre_post_test(
    pre: np.ndarray,
    post: np.ndarray,
    alpha: float = ALPHA,
) -> dict:
    """Compare pre vs post values using paired test (Wilcoxon or t-test).

    Normality is tested with Shapiro-Wilk. If either group is non-normal
    (p < alpha), Wilcoxon signed-rank is used; otherwise paired t-test.

    Args:
        pre: Pre-BD values (N,).
        post: Post-BD values (N,).
        alpha: Significance level for normality test.

    Returns:
        Dict with keys: test_name, statistic, pvalue.
    """
    valid = np.isfinite(pre) & np.isfinite(post)
    pre_v, post_v = pre[valid], post[valid]

    # Degenerate case: identical arrays → no test needed, p = 1
    if np.allclose(pre_v, post_v):
        return {"test_name": "trivial", "statistic": 0.0, "pvalue": 1.0}

    normal_pre = run_normality_test(pre_v).pvalue >= alpha if len(pre_v) >= 3 else False
    normal_post = run_normality_test(post_v).pvalue >= alpha if len(post_v) >= 3 else False

    if normal_pre and normal_post:
        stat, pval = scipy.stats.ttest_rel(pre_v, post_v)
        test_name = "ttest_rel"
    else:
        stat, pval = scipy.stats.wilcoxon(pre_v, post_v, zero_method="wilcox")
        test_name = "wilcoxon"

    return {"test_name": test_name, "statistic": float(stat), "pvalue": float(pval)}


def run_between_group_test(group_a: np.ndarray, group_b: np.ndarray) -> dict:
    """Compare two independent groups using Mann-Whitney U test.

    Args:
        group_a: Values for group A (e.g., BDR+).
        group_b: Values for group B (e.g., BDR−).

    Returns:
        Dict with keys: statistic, pvalue.
    """
    a_v = group_a[np.isfinite(group_a)]
    b_v = group_b[np.isfinite(group_b)]
    stat, pval = scipy.stats.mannwhitneyu(a_v, b_v, alternative="two-sided")
    return {"statistic": float(stat), "pvalue": float(pval)}


def cohen_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Compute Cohen's d effect size between two groups.

    When both groups have zero pooled standard deviation (e.g. constant
    arrays), the raw mean difference is returned as a degenerate effect size.
    For groups of all-zeros vs all-ones this yields d = -1.0, preserving the
    intuitive sign and magnitude of the separation.

    Args:
        group_a: 1D array of values for group A.
        group_b: 1D array of values for group B.

    Returns:
        Cohen's d (signed: positive if mean_a > mean_b).
    """
    a_v = np.asarray(group_a, dtype=float)
    b_v = np.asarray(group_b, dtype=float)
    pooled_std = _pooled_std(a_v, b_v)
    if pooled_std == 0.0:
        # Degenerate case: return raw mean difference
        return float(np.mean(a_v) - np.mean(b_v))
    return float((np.mean(a_v) - np.mean(b_v)) / pooled_std)


# ---------------------------------------------------------------------------
# Public API — Figures
# ---------------------------------------------------------------------------


def generate_all_figures(
    cas_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    output_dir: Path | str,
) -> None:
    """Generate and save all 4 publication-quality figures.

    Args:
        cas_df: DataFrame from compute_cas_rates() — columns: subject_id,
            cas_rate_pre, cas_rate_post, delta_cas.
        metadata_df: DataFrame from subject_metadata.csv — must include
            subject_id (as subject_num column) and bdr_label columns.
        output_dir: Directory where PNG files will be saved.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Merge BDR labels
    merged = cas_df.merge(
        metadata_df[["subject_num", "bdr_label"]].rename(columns={"subject_num": "subject_id"}),
        on="subject_id",
        how="left",
    )

    _plot_boxplot(merged, output_dir / FIG_BOXPLOT)
    _plot_roc(merged, output_dir / FIG_ROC)
    _plot_barplot(merged, output_dir / FIG_BARPLOT)
    _plot_heatmap(merged, output_dir / FIG_HEATMAP)


# ---------------------------------------------------------------------------
# Private helpers — statistics
# ---------------------------------------------------------------------------


def _pooled_std(a: np.ndarray, b: np.ndarray) -> float:
    """Compute pooled standard deviation for Cohen's d."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float(np.std(np.concatenate([a, b]), ddof=1))
    var_a = float(np.var(a, ddof=1))
    var_b = float(np.var(b, ddof=1))
    pooled = np.sqrt(((na - 1) * var_a + (nb - 1) * var_b) / (na + nb - 2))
    return float(pooled)


# ---------------------------------------------------------------------------
# Private helpers — figures
# ---------------------------------------------------------------------------


def _plot_boxplot(merged: pd.DataFrame, out_path: Path) -> None:
    """Boxplot: CAS rate pre vs post, BDR+ and BDR- side by side."""
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_data = []
    for _, row in merged.iterrows():
        plot_data.append({
            "subject_id": row["subject_id"],
            "bdr_label": row["bdr_label"],
            "condition": "Pre-BD",
            "cas_rate": row["cas_rate_pre"],
        })
        plot_data.append({
            "subject_id": row["subject_id"],
            "bdr_label": row["bdr_label"],
            "condition": "Post-BD",
            "cas_rate": row["cas_rate_post"],
        })
    df_long = pd.DataFrame(plot_data).dropna(subset=["cas_rate"])
    sns.boxplot(
        data=df_long, x="bdr_label", y="cas_rate", hue="condition", ax=ax, palette="Set2"
    )
    ax.set_xlabel("BDR Group")
    ax.set_ylabel("CAS Rate")
    ax.set_title("CAS Rate Pre vs Post Bronchodilator by BDR Group")
    ax.legend(title="Condition")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=DPI)
    plt.close(fig)


def _plot_roc(merged: pd.DataFrame, out_path: Path) -> None:
    """ROC curve using delta_cas as biomarker score for BDR+."""
    from sklearn.metrics import roc_curve, auc as sk_auc

    bdr_binary = (merged["bdr_label"] == "BDR+").astype(int).values
    scores = merged["delta_cas"].values
    valid = np.isfinite(scores)

    fig, ax = plt.subplots(figsize=(6, 6))
    if np.sum(valid) > 1 and len(np.unique(bdr_binary[valid])) > 1:
        fpr, tpr, _ = roc_curve(bdr_binary[valid], scores[valid])
        roc_auc = sk_auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"delta_CAS (AUC = {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — delta_CAS as BDR Biomarker")
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=DPI)
    plt.close(fig)


def _plot_barplot(merged: pd.DataFrame, out_path: Path) -> None:
    """Bar plot: mean CAS rate per subject, colour-coded by BDR label."""
    fig, ax = plt.subplots(figsize=(12, 5))
    merged_sorted = merged.sort_values("subject_id")
    colors = ["#e74c3c" if lbl == "BDR+" else "#3498db"
              for lbl in merged_sorted["bdr_label"]]
    ax.bar(merged_sorted["subject_id"].astype(str),
           merged_sorted["cas_rate_pre"], color=colors)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#e74c3c", label="BDR+"),
        Patch(facecolor="#3498db", label="BDR-"),
    ]
    ax.legend(handles=legend_elements)
    ax.set_xlabel("Subject ID")
    ax.set_ylabel("CAS Rate (Pre-BD)")
    ax.set_title("Mean CAS Rate per Subject (Pre-Bronchodilator)")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=DPI)
    plt.close(fig)


def _plot_heatmap(merged: pd.DataFrame, out_path: Path) -> None:
    """Heatmap: CAS rate matrix subjects × pre/post."""
    pivot = merged[["subject_id", "cas_rate_pre", "cas_rate_post"]].set_index("subject_id")
    pivot.columns = ["Pre-BD", "Post-BD"]
    fig, ax = plt.subplots(figsize=(4, max(6, len(pivot) // 3)))
    sns.heatmap(pivot, ax=ax, cmap="YlOrRd", annot=True, fmt=".2f", linewidths=0.5)
    ax.set_title("CAS Rate — Subjects × Condition")
    ax.set_xlabel("Condition")
    ax.set_ylabel("Subject ID")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=DPI)
    plt.close(fig)
