# ================================================================
# STEP 4: NON-LINEARITY CHECK — VISUAL + STATISTICAL
# ================================================================
# Checks run:
#   Visual 1 : Raw series + rolling variance
#   Visual 2 : ACF of residuals  (should be white noise)
#   Visual 3 : ACF of squared residuals  (spikes = non-linear!)
#   Visual 4 : Residuals vs fitted  (curve = non-linear mean)
#   Visual 5 : Residual distribution  (skew/fat tails = non-linear)
#
#   Stat 1   : BDS test — general hidden structure (broadest, weight=2)
#   Stat 2   : Ljung-Box on squared residuals — ARCH / volatility clustering (weight=1)
#   Stat 3   : RESET test — non-linearity in the mean (weight=1)
#
# Decision rules (applied in order):
#   1. BDS p < 0.001  → strong override → NON-LINEAR regardless of others
#   2. Weighted vote  → NON-LINEAR if weighted_nl_votes > total_weight / 2
#   3. Tie            → conservative default → NON-LINEAR
#   4. N < 30         → sample too small → fall back to MI scores (Step 2)
# ================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.ar_model            import AutoReg
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools             import add_constant
from statsmodels.stats.diagnostic        import acorr_ljungbox, linear_reset
from statsmodels.graphics.tsaplots       import plot_acf

print("\n" + "=" * 60)
print("STEP 4: NON-LINEARITY CHECK")
print("=" * 60)

# ── Config ──────────────────────────────────────────────────────
_NL_WEIGHTS       = {"BDS": 2, "LjungBox": 1, "RESET": 1}
_BDS_STRONG_THRESH = 0.001     # BDS p below this → hard override to NON-LINEAR
_AR_LAGS          = 2          # AR order for residual extraction
_LB_LAGS          = 4          # Ljung-Box lags on squared residuals
_ACF_LAGS         = 15         # lags shown in ACF plots
_ROLLING_WIN      = None       # rolling variance window — None = auto (N//6)


# ================================================================
# PART A — VISUAL CHECKS
# ================================================================

def _plot_nonlinearity_visuals(s, col_name, ar_resid, ar_ok):
    """
    6-panel diagnostic figure saved to disk.
    Panels:
      [0,0] Raw series + rolling variance overlay
      [0,1] ACF of residuals
      [0,2] ACF of squared residuals   ← key non-linearity indicator
      [1,0] Residuals vs fitted values ← curve = non-linear mean
      [1,1] Rolling variance (separate panel, larger)
      [1,2] Residual histogram         ← skew/fat tails = non-linear
    """
    win = _ROLLING_WIN or max(3, len(s) // 6)

    fig = plt.figure(figsize=(15, 8))
    fig.suptitle(
        f"Step 4 — Non-linearity visual checks : {col_name}",
        fontsize=13, y=1.01
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.35)

    # ── [0,0] Raw series ─────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(s, linewidth=0.9, color="#378ADD", label="series")
    ax.axhline(np.mean(s), color="#E24B4A", linewidth=0.8,
               linestyle="--", alpha=0.7, label="mean")
    ax.set_title("Raw series", fontsize=11)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Value")
    ax.legend(fontsize=8)

    # ── [0,1] ACF of raw residuals ───────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    if ar_ok:
        plot_acf(ar_resid, lags=_ACF_LAGS, ax=ax,
                 color="#378ADD", vlines_kwargs={"colors": "#378ADD"},
                 alpha=0.05, zero=False)
        ax.set_title("ACF — residuals\n(should stay within bands)", fontsize=11)
    else:
        ax.text(0.5, 0.5, "AR fit failed\n(no residuals)", ha="center",
                va="center", transform=ax.transAxes, color="gray", fontsize=10)
        ax.set_title("ACF — residuals", fontsize=11)

    # ── [0,2] ACF of SQUARED residuals ──────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    if ar_ok:
        plot_acf(ar_resid ** 2, lags=_ACF_LAGS, ax=ax,
                 color="#D85A30", vlines_kwargs={"colors": "#D85A30"},
                 alpha=0.05, zero=False)
        ax.set_title("ACF — squared residuals\n(spikes outside bands = NON-LINEAR!)",
                     fontsize=11)
        ax.title.set_color("#D85A30")
    else:
        ax.text(0.5, 0.5, "AR fit failed\n(no residuals)", ha="center",
                va="center", transform=ax.transAxes, color="gray", fontsize=10)
        ax.set_title("ACF — squared residuals", fontsize=11)

    # ── [1,0] Residuals vs fitted ────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    if ar_ok:
        fitted = s[_AR_LAGS:] - ar_resid
        ax.scatter(fitted, ar_resid, alpha=0.5, s=14, color="#7F77DD",
                   label="residuals")
        ax.axhline(0, color="#E24B4A", linewidth=0.8, linestyle="--")
        # fit quadratic trend line to reveal any non-linear mean structure
        z  = np.polyfit(fitted, ar_resid, 2)
        xr = np.linspace(fitted.min(), fitted.max(), 120)
        ax.plot(xr, np.polyval(z, xr), color="#D85A30", linewidth=1.5,
                linestyle="--", label="quadratic fit")
        ax.set_title("Residuals vs fitted\n(curved fit = non-linear mean)", fontsize=11)
        ax.set_xlabel("Fitted values")
        ax.set_ylabel("Residuals")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "AR fit failed", ha="center", va="center",
                transform=ax.transAxes, color="gray", fontsize=10)
        ax.set_title("Residuals vs fitted", fontsize=11)

    # ── [1,1] Rolling variance ───────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    roll_var = pd.Series(s).rolling(win).var().values
    ax.plot(roll_var, linewidth=0.9, color="#1D9E75")
    ax.axhline(np.nanmean(roll_var), color="#E24B4A", linewidth=0.8,
               linestyle="--", alpha=0.7, label="mean variance")
    ax.fill_between(range(len(roll_var)), roll_var,
                    np.nanmean(roll_var), alpha=0.12, color="#1D9E75")
    ax.set_title(f"Rolling variance  (window={win})\n(flat = constant variance → linear)",
                 fontsize=11)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Variance")
    ax.legend(fontsize=8)

    # ── [1,2] Residual histogram ─────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    if ar_ok:
        ax.hist(ar_resid, bins=20, color="#7F77DD", edgecolor="white",
                linewidth=0.4, alpha=0.85, density=True)
        ax.axvline(0, color="#E24B4A", linewidth=0.8, linestyle="--")
        # overlay normal curve for comparison
        mu, sigma = np.mean(ar_resid), np.std(ar_resid)
        xr = np.linspace(ar_resid.min(), ar_resid.max(), 200)
        ax.plot(xr, (1 / (sigma * np.sqrt(2 * np.pi))) *
                np.exp(-0.5 * ((xr - mu) / sigma) ** 2),
                color="#E24B4A", linewidth=1.2, linestyle="--", label="normal")
        skew = float(pd.Series(ar_resid).skew())
        ax.set_title(
            f"Residual distribution  (skew={skew:.2f})\n"
            f"(skew/fat tails = non-linear)", fontsize=11
        )
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "AR fit failed", ha="center", va="center",
                transform=ax.transAxes, color="gray", fontsize=10)
        ax.set_title("Residual distribution", fontsize=11)

    plt.tight_layout()
    fname = f"step4_nonlinearity_visuals_{col_name.replace(' ', '_')}.png"
    plt.savefig(fname, dpi=130, bbox_inches="tight")
    plt.show()
    print(f"  [saved] {fname}\n")


# ================================================================
# PART B — STATISTICAL TESTS + DECISION
# ================================================================

def check_nonlinearity(series, col_name="target", save_plots=True):
    """
    Run visual diagnostics + 3 statistical tests on `series`.
    Returns (is_nonlinear: bool, results: dict).
    """
    import pandas as pd

    s       = np.array(series.dropna() if hasattr(series, "dropna") else series,
                       dtype=float)
    results = {}

    # ── Pre-fit AR(2) — needed for Ljung-Box and residual plots ──
    try:
        ar_model = AutoReg(s, lags=_AR_LAGS).fit()
        ar_resid = ar_model.resid
        ar_ok    = True
    except Exception as e:
        print(f"  ⚠  AR({_AR_LAGS}) fit failed: {e}")
        ar_resid = None
        ar_ok    = False

    # ── Pre-fit OLS — needed for RESET ───────────────────────────
    try:
        X         = add_constant(np.arange(len(s)))
        ols_model = OLS(s, X).fit()
        ols_ok    = True
    except Exception as e:
        print(f"  ⚠  OLS fit failed: {e}")
        ols_ok = False

    # ── Visual checks ─────────────────────────────────────────────
    if save_plots:
        print("  Generating visual diagnostics ...")
        _plot_nonlinearity_visuals(s, col_name, ar_resid, ar_ok)

    print("  Statistical tests:")
    print("  " + "-" * 44)

    # ── Test 1: BDS ───────────────────────────────────────────────
    # Correct import: statsmodels.tsa.stattools (not stats.diagnostic)
    # Returns array of p-values — must call float() to get a scalar
    try:
        from statsmodels.tsa.stattools import bds as _bds
        _, p_arr       = _bds(s, distance=1.5)
        p              = float(p_arr)
        results["BDS"] = {
            "p"        : round(p, 4),
            "nonlinear": p < 0.05,
            "weight"   : _NL_WEIGHTS["BDS"],
        }
        flag = "⚠  non-linear" if p < 0.05 else "✅ linear"
        note = "  ← very strong signal" if p < _BDS_STRONG_THRESH else ""
        print(f"  BDS       : p={p:.4f}  →  {flag}{note}")
    except Exception as e:
        print(f"  BDS       : failed ({e})")

    # ── Test 2: Ljung-Box on squared residuals ────────────────────
    if ar_ok:
        try:
            lb = acorr_ljungbox(ar_resid ** 2, lags=_LB_LAGS, return_df=True)
            p  = float(lb["lb_pvalue"].min())
            results["LjungBox"] = {
                "p"        : round(p, 4),
                "nonlinear": p < 0.05,
                "weight"   : _NL_WEIGHTS["LjungBox"],
            }
            print(f"  Ljung-Box : p={p:.4f}  →  "
                  f"{'⚠  non-linear' if p < 0.05 else '✅ linear'}")
        except Exception as e:
            print(f"  Ljung-Box : failed ({e})")

    # ── Test 3: RESET ─────────────────────────────────────────────
    if ols_ok:
        try:
            reset = linear_reset(ols_model, power=2, use_f=True)
            p     = float(reset.pvalue)
            results["RESET"] = {
                "p"        : round(p, 4),
                "nonlinear": p < 0.05,
                "weight"   : _NL_WEIGHTS["RESET"],
            }
            print(f"  RESET     : p={p:.4f}  →  "
                  f"{'⚠  non-linear' if p < 0.05 else '✅ linear'}")
        except Exception as e:
            print(f"  RESET     : failed ({e})")

    # ── Bail out if every test failed ────────────────────────────
    if not results:
        print("\n  ⚠  All tests failed — defaulting to LINEAR")
        return False, {}

    # ── Weighted vote ─────────────────────────────────────────────
    total_weight = sum(v["weight"] for v in results.values())
    nl_weight    = sum(v["weight"] for v in results.values() if v["nonlinear"])
    simple_votes = sum(1 for v in results.values() if v["nonlinear"])
    total_tests  = len(results)

    # Rule 1: BDS strong override
    bds_res      = results.get("BDS", {})
    bds_override = (bds_res.get("p") is not None and
                    bds_res["p"] < _BDS_STRONG_THRESH)

    if bds_override:
        is_nonlinear  = True
        decision_note = (
            f"BDS override  p={bds_res['p']:.4f} < {_BDS_STRONG_THRESH} "
            f"— signal too strong to dismiss by majority vote"
        )
    elif nl_weight == total_weight / 2:
        # Rule 3: tie → conservative → non-linear
        is_nonlinear  = True
        decision_note = "Tied weighted vote — defaulting to NON-LINEAR (conservative)"
    else:
        # Rule 2: weighted majority
        is_nonlinear  = nl_weight > total_weight / 2
        decision_note = (
            f"Weighted vote  {nl_weight}/{total_weight} weight → "
            f"{'NON-LINEAR' if is_nonlinear else 'LINEAR'}"
        )

    print(f"\n  Tests run      : {total_tests}")
    print(f"  Votes          : {simple_votes}/{total_tests} non-linear")
    print(f"  Weighted votes : {nl_weight}/{total_weight} non-linear weight")
    print(f"  Reasoning      : {decision_note}")
    print(f"  Decision       : "
          f"{'⚠  NON-LINEAR → Transfer Entropy' if is_nonlinear else '✅ LINEAR → Granger'}")

    return is_nonlinear, results


# ================================================================
# RUN
# ================================================================

is_nonlinear, nl_results = check_nonlinearity(
    target_stat,
    col_name   = TARGET,       # uses your TARGET variable from config
    save_plots = True,         # set False to skip plot generation
)

causality_method = "transfer_entropy" if is_nonlinear else "granger"

# ── Small-sample override ────────────────────────────────────────
# Granger and Transfer Entropy need N > 30 (ideally > 50) to be
# statistically reliable. Fall back to MI scores from Step 2.
N_AVAILABLE = len(df_stationary)
if N_AVAILABLE < 30:
    print(f"\n⚠  N={N_AVAILABLE} < 30 — too small for reliable Granger/TE")
    print(f"   Overriding to MI-based selection from Step 2")
    print(f"   Granger requires N > 30 (ideally N > 50)")
    causality_method = "mi_only"
