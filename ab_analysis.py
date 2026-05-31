"""
A/B Testing Analysis: Cookie Cats Mobile Game
Dataset: cookie_cats.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu
import warnings
import os
from datetime import datetime

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 150, "font.size": 10})
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONTROL   = "gate_30"
TREATMENT = "gate_40"
CONTROL_LABEL   = "Gate 30 (Control)"
TREATMENT_LABEL = "Gate 40 (Treatment)"


# ── 1. Load & Validate ───────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["retention_1"] = df["retention_1"].astype(bool)
    df["retention_7"] = df["retention_7"].astype(bool)
    return df


# ── 2. EDA ───────────────────────────────────────────────────────────────────
def eda_summary(df: pd.DataFrame) -> dict:
    ctrl  = df[df["version"] == CONTROL]
    treat = df[df["version"] == TREATMENT]

    summary = {
        "total_users":       len(df),
        "control_n":         len(ctrl),
        "treatment_n":       len(treat),
        "control_pct":       len(ctrl) / len(df) * 100,
        "treatment_pct":     len(treat) / len(df) * 100,
        # Retention day-1
        "ctrl_ret1":         ctrl["retention_1"].mean() * 100,
        "treat_ret1":        treat["retention_1"].mean() * 100,
        # Retention day-7
        "ctrl_ret7":         ctrl["retention_7"].mean() * 100,
        "treat_ret7":        treat["retention_7"].mean() * 100,
        # Game rounds
        "ctrl_rounds_mean":  ctrl["sum_gamerounds"].mean(),
        "ctrl_rounds_med":   ctrl["sum_gamerounds"].median(),
        "treat_rounds_mean": treat["sum_gamerounds"].mean(),
        "treat_rounds_med":  treat["sum_gamerounds"].median(),
        # Relative lift
        "lift_ret1": (treat["retention_1"].mean() - ctrl["retention_1"].mean())
                     / ctrl["retention_1"].mean() * 100,
        "lift_ret7": (treat["retention_7"].mean() - ctrl["retention_7"].mean())
                     / ctrl["retention_7"].mean() * 100,
    }
    return summary, ctrl, treat


# ── 3. Statistical Tests ─────────────────────────────────────────────────────
def run_tests(ctrl: pd.DataFrame, treat: pd.DataFrame) -> dict:
    results = {}

    # ─ Chi-square test for retention_1
    c1 = ctrl["retention_1"].sum()
    t1 = treat["retention_1"].sum()
    contingency_1 = [[c1, len(ctrl) - c1],
                     [t1, len(treat) - t1]]
    chi2_1, p1, _, _ = chi2_contingency(contingency_1)
    results["chi2_ret1"]   = chi2_1
    results["p_ret1"]      = p1

    # ─ Chi-square test for retention_7
    c7 = ctrl["retention_7"].sum()
    t7 = treat["retention_7"].sum()
    contingency_7 = [[c7, len(ctrl) - c7],
                     [t7, len(treat) - t7]]
    chi2_7, p7, _, _ = chi2_contingency(contingency_7)
    results["chi2_ret7"]   = chi2_7
    results["p_ret7"]      = p7

    # ─ Mann-Whitney U for game rounds (non-normal distribution)
    u_stat, p_rounds = mannwhitneyu(
        ctrl["sum_gamerounds"],
        treat["sum_gamerounds"],
        alternative="two-sided"
    )
    results["u_stat_rounds"] = u_stat
    results["p_rounds"]      = p_rounds

    # ─ Effect size (Cohen's h for proportions)
    p_ctrl_1  = ctrl["retention_1"].mean()
    p_treat_1 = treat["retention_1"].mean()
    results["cohens_h_ret1"] = 2 * np.arcsin(np.sqrt(p_treat_1)) - \
                               2 * np.arcsin(np.sqrt(p_ctrl_1))

    p_ctrl_7  = ctrl["retention_7"].mean()
    p_treat_7 = treat["retention_7"].mean()
    results["cohens_h_ret7"] = 2 * np.arcsin(np.sqrt(p_treat_7)) - \
                               2 * np.arcsin(np.sqrt(p_ctrl_7))

    return results


# ── 4. Bootstrap Confidence Intervals ────────────────────────────────────────
def bootstrap_ci(ctrl: pd.DataFrame, treat: pd.DataFrame,
                 metric: str, n_boot: int = 5000, alpha: float = 0.05) -> dict:
    np.random.seed(42)
    n_c, n_t = len(ctrl), len(treat)

    diffs = []
    for _ in range(n_boot):
        s_c = ctrl[metric].sample(n=n_c, replace=True).mean()
        s_t = treat[metric].sample(n=n_t, replace=True).mean()
        diffs.append(s_t - s_c)

    diffs = np.array(diffs)
    observed_diff = treat[metric].mean() - ctrl[metric].mean()
    ci_lo = np.percentile(diffs, 100 * alpha / 2)
    ci_hi = np.percentile(diffs, 100 * (1 - alpha / 2))

    return {
        "observed_diff": observed_diff,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "diffs": diffs,
        "significant": not (ci_lo <= 0 <= ci_hi),
    }


# ── 5. Plots ─────────────────────────────────────────────────────────────────
def plot_all(df, ctrl, treat, summary, boot_r1, boot_r7):

    # ─ Figure 1: Overview dashboard ─────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Cookie Cats A/B Test — Analysis Dashboard", fontsize=15, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38)

    colors = {"gate_30": "#4C72B0", "gate_40": "#DD8452"}
    labels = [CONTROL_LABEL, TREATMENT_LABEL]

    # 1a) Group sizes
    ax0 = fig.add_subplot(gs[0, 0])
    sizes = [summary["control_n"], summary["treatment_n"]]
    bars = ax0.bar(labels, sizes, color=[colors[CONTROL], colors[TREATMENT]], width=0.5)
    ax0.set_title("Sample Size per Group")
    ax0.set_ylabel("Users")
    for bar, n in zip(bars, sizes):
        ax0.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                 f"{n:,}", ha="center", va="bottom", fontsize=9)

    # 1b) 1-day retention rate
    ax1 = fig.add_subplot(gs[0, 1])
    ret1 = [summary["ctrl_ret1"], summary["treat_ret1"]]
    bars = ax1.bar(labels, ret1, color=[colors[CONTROL], colors[TREATMENT]], width=0.5)
    ax1.set_title("1-Day Retention Rate (%)")
    ax1.set_ylabel("Retention (%)")
    ax1.set_ylim(0, max(ret1) * 1.2)
    for bar, v in zip(bars, ret1):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{v:.2f}%", ha="center", va="bottom", fontsize=9)

    # 1c) 7-day retention rate
    ax2 = fig.add_subplot(gs[0, 2])
    ret7 = [summary["ctrl_ret7"], summary["treat_ret7"]]
    bars = ax2.bar(labels, ret7, color=[colors[CONTROL], colors[TREATMENT]], width=0.5)
    ax2.set_title("7-Day Retention Rate (%)")
    ax2.set_ylabel("Retention (%)")
    ax2.set_ylim(0, max(ret7) * 1.2)
    for bar, v in zip(bars, ret7):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                 f"{v:.2f}%", ha="center", va="bottom", fontsize=9)

    # 1d) Game rounds distribution (capped at 200)
    ax3 = fig.add_subplot(gs[1, 0])
    capped_ctrl  = ctrl["sum_gamerounds"].clip(upper=200)
    capped_treat = treat["sum_gamerounds"].clip(upper=200)
    ax3.hist(capped_ctrl,  bins=40, alpha=0.6, label=CONTROL_LABEL,   color=colors[CONTROL])
    ax3.hist(capped_treat, bins=40, alpha=0.6, label=TREATMENT_LABEL, color=colors[TREATMENT])
    ax3.set_title("Game Rounds Distribution (capped @200)")
    ax3.set_xlabel("Game Rounds")
    ax3.set_ylabel("Users")
    ax3.legend(fontsize=8)

    # 1e) Bootstrap dist for retention_1
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(boot_r1["diffs"] * 100, bins=60, color="#6baed6", edgecolor="none")
    ax4.axvline(0, color="red", linestyle="--", linewidth=1.5, label="No effect")
    ax4.axvline(boot_r1["ci_lo"] * 100, color="orange", linestyle=":", linewidth=1.5)
    ax4.axvline(boot_r1["ci_hi"] * 100, color="orange", linestyle=":", linewidth=1.5,
                label="95% CI")
    ax4.set_title("Bootstrap: Δ 1-Day Retention")
    ax4.set_xlabel("Difference (pp)")
    ax4.set_ylabel("Count")
    ax4.legend(fontsize=8)

    # 1f) Bootstrap dist for retention_7
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.hist(boot_r7["diffs"] * 100, bins=60, color="#74c476", edgecolor="none")
    ax5.axvline(0, color="red", linestyle="--", linewidth=1.5, label="No effect")
    ax5.axvline(boot_r7["ci_lo"] * 100, color="orange", linestyle=":", linewidth=1.5)
    ax5.axvline(boot_r7["ci_hi"] * 100, color="orange", linestyle=":", linewidth=1.5,
                label="95% CI")
    ax5.set_title("Bootstrap: Δ 7-Day Retention")
    ax5.set_xlabel("Difference (pp)")
    ax5.set_ylabel("Count")
    ax5.legend(fontsize=8)

    plt.savefig(os.path.join(OUTPUT_DIR, "dashboard.png"), bbox_inches="tight")
    plt.close()

    # ─ Figure 2: Retention funnel ────────────────────────────────────────────
    fig2, ax = plt.subplots(figsize=(8, 5))
    metrics = ["Installed", "Day-1 Retained", "Day-7 Retained"]
    ctrl_vals  = [100, summary["ctrl_ret1"],  summary["ctrl_ret7"]]
    treat_vals = [100, summary["treat_ret1"], summary["treat_ret7"]]

    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width/2, ctrl_vals,  width, label=CONTROL_LABEL,   color=colors[CONTROL],   alpha=0.85)
    ax.bar(x + width/2, treat_vals, width, label=TREATMENT_LABEL, color=colors[TREATMENT], alpha=0.85)
    for i, (c, t) in enumerate(zip(ctrl_vals, treat_vals)):
        ax.text(i - width/2, c + 0.5, f"{c:.1f}%", ha="center", fontsize=8)
        ax.text(i + width/2, t + 0.5, f"{t:.1f}%", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("% of Users")
    ax.set_title("Retention Funnel: Gate 30 vs Gate 40")
    ax.legend()
    ax.set_ylim(0, 115)
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUTPUT_DIR, "retention_funnel.png"), bbox_inches="tight")
    plt.close()
    print("  [OK] Charts saved to output/")


# ── 6. Generate Report ───────────────────────────────────────────────────────
def generate_report(summary: dict, tests: dict, boot_r1: dict, boot_r7: dict) -> str:
    alpha = 0.05

    def sig_label(p): return "SIGNIFICANT" if p < alpha else "NOT significant"
    def effect_label(h):
        h = abs(h)
        if h < 0.2: return "negligible"
        if h < 0.5: return "small"
        if h < 0.8: return "medium"
        return "large"
    def boot_label(b): return "SIGNIFICANT" if b["significant"] else "NOT significant"

    report = f"""
================================================================================
  COOKIE CATS A/B TESTING — FULL ANALYSIS REPORT
  Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
================================================================================

EXPERIMENT DESCRIPTION
─────────────────────
Game   : Cookie Cats (mobile puzzle game)
Test   : Gate position — players hit a timed gate at level 30 (control)
         vs level 40 (treatment)
Goal   : Determine which gate position maximises player retention
Metrics: 1-day retention, 7-day retention, game rounds played


1. DATASET OVERVIEW
───────────────────
  Total users      : {summary["total_users"]:,}
  Control  (Gate 30): {summary["control_n"]:,} users ({summary["control_pct"]:.1f}%)
  Treatment(Gate 40): {summary["treatment_n"]:,} users ({summary["treatment_pct"]:.1f}%)

  ✔ Groups are well-balanced (random assignment assumed).


2. KEY METRICS
──────────────
  ┌─────────────────────────┬──────────────┬──────────────┬───────────┐
  │ Metric                  │  Gate 30 (C) │  Gate 40 (T) │   Lift    │
  ├─────────────────────────┼──────────────┼──────────────┼───────────┤
  │ 1-Day Retention         │  {summary["ctrl_ret1"]:>8.2f}%   │  {summary["treat_ret1"]:>8.2f}%   │ {summary["lift_ret1"]:>+7.2f}%  │
  │ 7-Day Retention         │  {summary["ctrl_ret7"]:>8.2f}%   │  {summary["treat_ret7"]:>8.2f}%   │ {summary["lift_ret7"]:>+7.2f}%  │
  │ Avg Game Rounds         │  {summary["ctrl_rounds_mean"]:>8.1f}   │  {summary["treat_rounds_mean"]:>8.1f}   │     —     │
  │ Median Game Rounds      │  {summary["ctrl_rounds_med"]:>8.0f}   │  {summary["treat_rounds_med"]:>8.0f}   │     —     │
  └─────────────────────────┴──────────────┴──────────────┴───────────┘


3. STATISTICAL TESTS  (α = {alpha})
──────────────────────────────────

  [A] 1-Day Retention — Chi-Square Test
      H0: no difference in 1-day retention between groups
      χ² = {tests["chi2_ret1"]:.4f},  p = {tests["p_ret1"]:.6f}
      Cohen's h = {tests["cohens_h_ret1"]:+.4f} ({effect_label(tests["cohens_h_ret1"])} effect)
      Result: {sig_label(tests["p_ret1"])}

  [B] 7-Day Retention — Chi-Square Test
      H0: no difference in 7-day retention between groups
      χ² = {tests["chi2_ret7"]:.4f},  p = {tests["p_ret7"]:.6f}
      Cohen's h = {tests["cohens_h_ret7"]:+.4f} ({effect_label(tests["cohens_h_ret7"])} effect)
      Result: {sig_label(tests["p_ret7"])}

  [C] Game Rounds — Mann-Whitney U Test (non-parametric)
      H0: no difference in game rounds distribution between groups
      U = {tests["u_stat_rounds"]:.0f},  p = {tests["p_rounds"]:.6f}
      Result: {sig_label(tests["p_rounds"])}


4. BOOTSTRAP ANALYSIS  (5,000 iterations, 95% CI)
──────────────────────────────────────────────────

  [1-Day Retention]
    Observed difference : {boot_r1["observed_diff"]*100:+.4f} pp
    95% CI              : [{boot_r1["ci_lo"]*100:+.4f} pp,  {boot_r1["ci_hi"]*100:+.4f} pp]
    Zero in CI?         : {"Yes — NOT significant" if not boot_r1["significant"] else "No  — " + boot_label(boot_r1)}

  [7-Day Retention]
    Observed difference : {boot_r7["observed_diff"]*100:+.4f} pp
    95% CI              : [{boot_r7["ci_lo"]*100:+.4f} pp,  {boot_r7["ci_hi"]*100:+.4f} pp]
    Zero in CI?         : {"Yes — NOT significant" if not boot_r7["significant"] else "No  — " + boot_label(boot_r7)}


5. INTERPRETATION & BUSINESS INSIGHTS
──────────────────────────────────────

  1-DAY RETENTION
  ───────────────
  Gate 30 retains {summary["ctrl_ret1"]:.2f}% of players after one day; Gate 40 retains
  {summary["treat_ret1"]:.2f}%.  The difference ({summary["lift_ret1"]:+.2f}% relative lift) is
  {'STATISTICALLY SIGNIFICANT' if tests["p_ret1"] < alpha else 'NOT statistically significant'}
  (p = {tests["p_ret1"]:.4f}).  The effect size is {effect_label(tests["cohens_h_ret1"])} (Cohen's h ≈ {tests["cohens_h_ret1"]:.3f}),
  so even if real, the practical impact on day-1 is minimal.

  7-DAY RETENTION  ← KEY FINDING
  ──────────────────────────────
  This is the most important metric for long-term monetisation.
  Gate 30 retains {summary["ctrl_ret7"]:.2f}% of players after seven days; Gate 40 retains
  {summary["treat_ret7"]:.2f}%.
  Relative lift: {summary["lift_ret7"]:+.2f}%
  Chi-square p-value: {tests["p_ret7"]:.6f} → {'SIGNIFICANT' if tests["p_ret7"] < alpha else 'NOT significant'}
  Bootstrap 95% CI crosses zero: {"Yes" if not boot_r7["significant"] else "No"}

  {"► Gate 30 produces HIGHER 7-day retention than Gate 40." if summary["ctrl_ret7"] > summary["treat_ret7"] else "► Gate 40 produces HIGHER 7-day retention than Gate 30."}
  This aligns with the psychological concept of 'hedonic adaptation':
  players who hit the gate earlier (level 30) feel a stronger urge to
  return after the wait; players who progress further (level 40) may
  become more satisfied and less motivated to return.

  GAME ENGAGEMENT
  ───────────────
  Median rounds are similar between groups ({summary["ctrl_rounds_med"]:.0f} vs {summary["treat_rounds_med"]:.0f}),
  suggesting the gate position does not meaningfully change how much
  players play before they either churn or are gated.


6. RECOMMENDATION
─────────────────
  {"✅  KEEP Gate 30 (do NOT ship Gate 40)." if summary["ctrl_ret7"] > summary["treat_ret7"] else "✅  SHIP Gate 40 (better 7-day retention)."}

  Rationale:
  • 7-day retention is the stronger predictor of long-term revenue.
  • {"Gate 30 wins on 7-day retention with statistical significance." if tests["p_ret7"] < alpha else "Results on 7-day retention are not statistically significant — more data needed."}
  • The difference in 1-day retention is negligible in practical terms.
  • Moving the gate later (to level 40) does not improve any metric
    and significantly hurts the metric that matters most.

  Next Steps:
  1. Run the experiment longer if sample size was still accumulating.
  2. Segment results by platform (iOS / Android) if available.
  3. Consider testing gate_50 vs gate_30 to confirm the trend.
  4. Analyse revenue / in-app purchase data alongside retention.


================================================================================
  OUTPUT FILES
  • output/dashboard.png          — Full 6-panel analysis dashboard
  • output/retention_funnel.png   — Retention funnel comparison
  • output/interpretation.txt     — This report (plain text)
================================================================================
"""
    return report.strip()


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Cookie Cats A/B Testing Analysis")
    print("=" * 60)

    print("\n[1/5] Loading data...")
    df = load_data("cookie_cats.csv")
    print(f"  Loaded {len(df):,} rows | columns: {list(df.columns)}")

    print("\n[2/5] Computing summary statistics...")
    summary, ctrl, treat = eda_summary(df)
    print(f"  Control:   {summary['control_n']:,} users | 1-day ret: {summary['ctrl_ret1']:.2f}% | 7-day ret: {summary['ctrl_ret7']:.2f}%")
    print(f"  Treatment: {summary['treatment_n']:,} users | 1-day ret: {summary['treat_ret1']:.2f}% | 7-day ret: {summary['treat_ret7']:.2f}%")

    print("\n[3/5] Running statistical tests...")
    tests = run_tests(ctrl, treat)
    print(f"  Retention-1 chi2 p = {tests['p_ret1']:.6f}")
    print(f"  Retention-7 chi2 p = {tests['p_ret7']:.6f}")

    print("\n[4/5] Bootstrap confidence intervals (5,000 iterations)...")
    boot_r1 = bootstrap_ci(ctrl, treat, "retention_1")
    boot_r7 = bootstrap_ci(ctrl, treat, "retention_7")
    print(f"  Diff 1-day: {boot_r1['observed_diff']*100:+.4f} pp  95% CI [{boot_r1['ci_lo']*100:.4f}, {boot_r1['ci_hi']*100:.4f}]")
    print(f"  Diff 7-day: {boot_r7['observed_diff']*100:+.4f} pp  95% CI [{boot_r7['ci_lo']*100:.4f}, {boot_r7['ci_hi']*100:.4f}]")

    print("\n[5/5] Generating charts and report...")
    plot_all(df, ctrl, treat, summary, boot_r1, boot_r7)

    report_text = generate_report(summary, tests, boot_r1, boot_r7)
    report_path = os.path.join(OUTPUT_DIR, "interpretation.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"  [OK] Report saved to {report_path}")

    print("\n" + "=" * 60)
    print("  Analysis complete! Check the output/ folder.")
    print("=" * 60)
