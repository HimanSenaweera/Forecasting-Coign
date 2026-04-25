# ================================================================
# FULL PIPELINE — Feature Selection + SARIMAX Grid Search
# Coign Portfolio — Gross Credit Losses Forecasting
# ================================================================
# PIPELINE ORDER:
#   Step 1 : Stationarity (ADF + KPSS)
#   Step 2 : Correlation + Mutual Information screening
#   Step 3 : VIF redundancy check
#   Step 4 : Non-linearity check (RESET + Ljung-Box + BDS)
#   Step 5 : Causality test (Granger or Transfer Entropy)
#   Step 6 : SARIMAX rolling grid search on surviving features
# ================================================================

import warnings
import itertools
import datetime

import numpy as np
import pandas as pd
import scipy.stats as scipy_stats

from statsmodels.tsa.stattools        import adfuller, kpss, grangercausalitytests
from statsmodels.tsa.ar_model         import AutoReg
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.diagnostic     import acorr_ljungbox, linear_reset
from statsmodels.tools.tools          import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.preprocessing            import StandardScaler
from sklearn.feature_selection        import mutual_info_regression
from sklearn.metrics                  import (
    mean_absolute_percentage_error,
    mean_absolute_error,
    mean_squared_error,
)

from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

# ── Try importing pyinform (Transfer Entropy) ─────────────────
try:
    from pyinform.transferentropy import transfer_entropy as pyinform_te
    TE_AVAILABLE = True
except ImportError:
    TE_AVAILABLE = False
    print("⚠  pyinform not installed — Transfer Entropy unavailable")
    print("   pip install pyinform  to enable it")


# ================================================================
# USER CONFIG — set these before running
# ================================================================

TARGET      = "Gross Credit Losses"     # target column name in df
ALL_FEATURES = [                         # all candidate feature columns
    "90+ DQ Rate",
    "Payment Rate",
    "Finance Charges Rate",
    "Expected Loss Roll Ave 3M",
    # add more features here
]

# Date ranges (ISO format strings)
FIRST_INPUT_DATE = "2024-01-31"
LAST_INPUT_DATE  = "2025-08-31"
FIRST_TEST_DATE  = "2025-09-30"
LAST_TEST_DATE   = "2026-02-28"

# df = your dataframe (df_merged) — must be defined before running
# df = df_merged.copy()


# ================================================================
# METRIC FUNCTIONS — match your notebook exactly
# ================================================================

def smape(y_true, y_pred):
    return float(
        np.mean(
            2 * np.abs(y_true - y_pred) /
            (np.abs(y_true) + np.abs(y_pred))
        ) * 100
    )

def mape_pct(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred) * 100

def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def print_metrics(label, y_true, y_pred):
    m  = y_true != 0
    yt = y_true[m]
    yp = y_pred[m]
    print(f"  {label}")
    print(f"    MAPE  : {mape_pct(yt, yp):.2f}%")
    print(f"    SMAPE : {smape(yt, yp):.2f}%")
    print(f"    MAE   : {mae(yt, yp):>14,.0f}")
    print(f"    RMSE  : {rmse(yt, yp):>14,.0f}")


# ================================================================
# STEP 1: STATIONARITY CHECK
# ================================================================

print("=" * 60)
print("STEP 1: STATIONARITY CHECK")
print("=" * 60)


def check_stationarity(series, name=""):
    try:
        s = series.dropna()
        if len(s) < 10:
            return False
        adf_p  = adfuller(s)[1]
        kpss_p = kpss(s, regression="c", nlags="auto")[1]
        return (adf_p < 0.05) and (kpss_p > 0.05)
    except:
        return False


def make_stationary(series, col_name, max_diffs=2):
    s = series.copy()
    for d in range(max_diffs + 1):
        if check_stationarity(s.dropna(), col_name):
            return s, d
        if d < max_diffs:
            s = s.diff()
    print(f"  ⚠  {col_name}: not stationary after {max_diffs} diffs — kept at d=1")
    return series.diff(), 1


df_stationary  = df.copy()
non_stationary = []
diff_order     = {}

for col in ALL_FEATURES:
    if check_stationarity(df[col].dropna(), col):
        diff_order[col] = 0
    else:
        transformed, n_diffs = make_stationary(df[col], col)
        df_stationary[col]   = transformed
        diff_order[col]      = n_diffs
        non_stationary.append(col)

        is_now_stat = check_stationarity(transformed.dropna(), col)
        status      = "✅" if is_now_stat else "⚠  still non-stationary"
        if n_diffs == 2:
            print(f"  {status} {col} needed d={n_diffs}")

# Target differenced separately — for Steps 2-5 only
target_stat, _ = make_stationary(df[TARGET].copy(), TARGET)

# Align lengths
df_stationary = df_stationary.dropna().reset_index(drop=True)
target_stat   = target_stat.dropna().reset_index(drop=True)
min_len       = min(len(df_stationary), len(target_stat))
df_stationary = df_stationary.iloc[:min_len].reset_index(drop=True)
target_stat   = target_stat.iloc[:min_len].reset_index(drop=True)

d_max = max(diff_order.values()) if diff_order else 0

d0 = sum(1 for v in diff_order.values() if v == 0)
d1 = sum(1 for v in diff_order.values() if v == 1)
d2 = sum(1 for v in diff_order.values() if v == 2)

print(f"Already stationary  (d=0) : {d0}")
print(f"Stationary after 1 diff   : {d1}")
print(f"Stationary after 2 diffs  : {d2}")
print(f"Rows after differencing   : {len(df_stationary)}")
print(f"d_max across all features : {d_max}")


# ================================================================
# STEP 2: CORRELATION + MUTUAL INFORMATION SCREENING
# ================================================================

print("\n" + "=" * 60)
print("STEP 2: CORRELATION + MUTUAL INFORMATION SCREENING")
print("=" * 60)

LAGS        = [0, 1, 2, 3]
CORR_THRESH = 0.25
MI_THRESH   = 0.05

screening_results = []

for col in ALL_FEATURES:
    max_corr    = 0
    max_mi      = 0
    best_lag_c  = 0
    best_lag_mi = 0

    for lag in LAGS:
        feature_lagged = df_stationary[col].shift(lag)
        combined = pd.concat(
            [feature_lagged, target_stat], axis=1
        ).dropna()
        combined.columns = ["feature", "target"]

        if len(combined) < 10:
            continue

        corr = abs(combined["feature"].corr(combined["target"]))
        if not np.isnan(corr) and corr > max_corr:
            max_corr   = corr
            best_lag_c = lag

        X  = combined["feature"].values.reshape(-1, 1)
        y  = combined["target"].values
        mi = mutual_info_regression(X, y, random_state=42)[0]
        if mi > max_mi:
            max_mi      = mi
            best_lag_mi = lag

    screening_results.append({
        "feature"     : col,
        "max_corr"    : round(max_corr,    4),
        "best_lag_corr": best_lag_c,
        "max_mi"      : round(max_mi,      4),
        "best_lag_mi" : best_lag_mi,
    })

screening_df   = pd.DataFrame(screening_results)
step2_features = screening_df[
    (screening_df["max_corr"] >= CORR_THRESH) |
    (screening_df["max_mi"]   >= MI_THRESH)
]["feature"].tolist()

dropped = len(ALL_FEATURES) - len(step2_features)
print(f"Features before screening  : {len(ALL_FEATURES)}")
print(f"Features after screening   : {len(step2_features)}")
print(f"Features dropped           : {dropped}")
print(f"\nAll features ranked by correlation:")
print(
    screening_df
    .sort_values("max_corr", ascending=False)
    .to_string(index=False)
)


# ================================================================
# STEP 3: VIF REDUNDANCY CHECK
# ================================================================

print("\n" + "=" * 60)
print("STEP 3: VIF REDUNDANCY CHECK")
print("=" * 60)

VIF_THRESH = 5.0


def compute_vif(feature_df):
    scaler    = StandardScaler()
    scaled    = scaler.fit_transform(feature_df.fillna(0))
    scaled_df = pd.DataFrame(scaled, columns=feature_df.columns)
    vif_data  = pd.DataFrame({
        "feature": scaled_df.columns,
        "VIF"    : [
            variance_inflation_factor(scaled_df.values, i)
            for i in range(scaled_df.shape[1])
        ],
    })
    return vif_data.sort_values("VIF", ascending=False).reset_index(drop=True)


remaining_features = step2_features.copy()
mi_scores          = screening_df.set_index("feature")["max_mi"]

while True:
    if len(remaining_features) <= 2:
        break

    feat_df = df_stationary[remaining_features].fillna(0)
    vif_df  = compute_vif(feat_df)
    max_vif = vif_df["VIF"].max()

    if max_vif <= VIF_THRESH:
        break

    top_vif    = vif_df[vif_df["VIF"] > VIF_THRESH]["feature"].tolist()
    worst_feat = min(top_vif, key=lambda f: mi_scores.get(f, 0))
    remaining_features.remove(worst_feat)
    print(f"  Removed (VIF={max_vif:.1f}, lowest MI): {worst_feat}")

step3_features = remaining_features
print(f"\nFeatures after VIF check   : {len(step3_features)}")
print(f"Remaining                  : {step3_features}")

# Print final VIF table
if step3_features:
    final_vif = compute_vif(df_stationary[step3_features].fillna(0))
    print(f"\nFinal VIF scores:")
    print(final_vif.to_string(index=False))


# ================================================================
# STEP 4: NON-LINEARITY CHECK — MAJORITY VOTE
# ================================================================

print("\n" + "=" * 60)
print("STEP 4: NON-LINEARITY CHECK")
print("=" * 60)


def check_nonlinearity(series):
    s       = series.dropna().values
    results = {}

    # Fit AR(2) for Ljung-Box
    try:
        ar_model = AutoReg(s, lags=2).fit()
        ar_resid = ar_model.resid
        ar_ok    = True
    except Exception as e:
        print(f"  ⚠  AR(2) failed: {e}")
        ar_ok = False

    # Fit OLS for RESET
    try:
        X         = add_constant(np.arange(len(s)))
        ols_model = OLS(s, X).fit()
        ols_ok    = True
    except Exception as e:
        print(f"  ⚠  OLS failed: {e}")
        ols_ok = False

    # Test 1: BDS
    try:
        from statsmodels.stats.diagnostic import bds
        _, p           = bds(s, distance=1.5)
        results["BDS"] = {"p": round(p, 4), "nonlinear": p < 0.05}
        print(f"  BDS       : p={p:.4f}  →  "
              f"{'⚠  non-linear' if p < 0.05 else '✅ linear'}")
    except Exception as e:
        print(f"  BDS       : failed ({e})")

    # Test 2: Ljung-Box on squared residuals
    if ar_ok:
        try:
            lb = acorr_ljungbox(ar_resid ** 2, lags=4, return_df=True)
            p  = lb["lb_pvalue"].min()
            results["LjungBox"] = {"p": round(p, 4), "nonlinear": p < 0.05}
            print(f"  Ljung-Box : p={p:.4f}  →  "
                  f"{'⚠  non-linear' if p < 0.05 else '✅ linear'}")
        except Exception as e:
            print(f"  Ljung-Box : failed ({e})")

    # Test 3: RESET
    if ols_ok:
        try:
            reset = linear_reset(ols_model, power=2, use_f=True)
            p     = reset.pvalue
            results["RESET"] = {"p": round(p, 4), "nonlinear": p < 0.05}
            print(f"  RESET     : p={p:.4f}  →  "
                  f"{'⚠  non-linear' if p < 0.05 else '✅ linear'}")
        except Exception as e:
            print(f"  RESET     : failed ({e})")

    if not results:
        print("  ⚠  All tests failed — defaulting to LINEAR")
        return False, {}

    votes        = sum(v["nonlinear"] for v in results.values())
    total        = len(results)
    is_nonlinear = votes > total / 2

    print(f"\n  Tests run    : {total}")
    print(f"  Non-linear   : {votes}/{total} votes")
    print(f"  Decision     : "
          f"{'⚠  NON-LINEAR → Transfer Entropy' if is_nonlinear else '✅ LINEAR → Granger'}")

    return is_nonlinear, results


is_nonlinear, nl_results = check_nonlinearity(target_stat)
causality_method = "transfer_entropy" if is_nonlinear else "granger"

# Small sample override
N_AVAILABLE = len(df_stationary)
if N_AVAILABLE < 30:
    print(f"\n⚠  N={N_AVAILABLE} < 30 — too small for reliable Granger/TE")
    print(f"   Overriding to MI-based selection from Step 2")
    print(f"   Granger requires N > 30 (ideally N > 50)")
    causality_method = "mi_only"


# ================================================================
# STEP 5: CAUSALITY TEST
# ================================================================

print("\n" + "=" * 60)
print(f"STEP 5: CAUSALITY TEST ({causality_method.upper()})")
print("=" * 60)


def run_granger(step3_features, df_stationary, target_stat,
                max_lag=3, p_thresh=0.05):

    print(f"Base features to test : {len(step3_features)}")
    causal_features   = []
    causality_results = []

    for col in step3_features:
        try:
            combined = pd.concat(
                [df_stationary[col], target_stat], axis=1
            ).dropna()
            combined.columns = ["feature", "target"]

            if len(combined) < 15:
                print(f"  Skipped {col}: insufficient rows ({len(combined)})")
                continue

            # Lag 0: Pearson correlation p-value
            _, corr_p   = scipy_stats.pearsonr(
                combined["feature"], combined["target"]
            )
            lag_pvalues = {0: round(corr_p, 4)}

            # Lags 1..max_lag: Granger F-test
            test_result = grangercausalitytests(
                combined[["target", "feature"]],
                maxlag  = max_lag,
                verbose = False,
            )
            for lag in range(1, max_lag + 1):
                lag_pvalues[lag] = round(
                    test_result[lag][0]["ssr_ftest"][1], 4
                )

            best_lag = min(lag_pvalues, key=lag_pvalues.get)
            best_p   = lag_pvalues[best_lag]

            print(f"\n  {col}")
            for lag, p in lag_pvalues.items():
                sig    = "✅" if p < p_thresh else "❌"
                marker = " ← best" if lag == best_lag else ""
                print(f"    lag {lag}: p={p:.4f} {sig}{marker}")

            causality_results.append({
                "feature" : col,
                "best_lag": best_lag,
                "best_p"  : best_p,
                "causal"  : best_p < p_thresh,
            })

            if best_p < p_thresh:
                causal_features.append({
                    "feature" : col,
                    "best_lag": best_lag,
                })

        except Exception as e:
            print(f"  Skipped {col}: {e}")

    return causal_features, pd.DataFrame(causality_results)


def run_transfer_entropy(step3_features, df_stationary, target_stat,
                         max_lag=3, te_thresh=0.01):

    if not TE_AVAILABLE:
        print("⚠  pyinform unavailable — falling back to Granger")
        return run_granger(step3_features, df_stationary, target_stat, max_lag)

    print(f"Base features to test : {len(step3_features)}")

    def discretize(series, bins=10):
        s            = series.dropna().values
        s_min, s_max = s.min(), s.max()
        if s_max == s_min:
            return np.zeros(len(s), dtype=int)
        binned = (
            (s - s_min) / (s_max - s_min) * (bins - 1)
        ).astype(int)
        return np.clip(binned, 0, bins - 1)

    causal_features   = []
    causality_results = []

    for col in step3_features:
        try:
            lag_te_scores = {}

            for lag in range(0, max_lag + 1):
                feat_lagged = df_stationary[col].shift(lag)
                combined    = pd.concat(
                    [feat_lagged, target_stat], axis=1
                ).dropna()

                if len(combined) < 15:
                    continue

                feat_disc      = discretize(combined.iloc[:, 0])
                tgt_disc       = discretize(combined.iloc[:, 1])
                te_score       = pyinform_te(feat_disc, tgt_disc, k=1)
                lag_te_scores[lag] = round(float(te_score), 6)

            if not lag_te_scores:
                continue

            best_lag = max(lag_te_scores, key=lag_te_scores.get)
            best_te  = lag_te_scores[best_lag]

            print(f"\n  {col}")
            for lag, score in lag_te_scores.items():
                sig    = "✅" if score > te_thresh else "❌"
                marker = " ← best" if lag == best_lag else ""
                print(f"    lag {lag}: TE={score:.6f} {sig}{marker}")

            causality_results.append({
                "feature" : col,
                "best_lag": best_lag,
                "best_te" : best_te,
                "causal"  : best_te > te_thresh,
            })

            if best_te > te_thresh:
                causal_features.append({
                    "feature" : col,
                    "best_lag": best_lag,
                })

        except Exception as e:
            print(f"  Skipped {col}: {e}")

    return causal_features, pd.DataFrame(causality_results)


def run_mi_selection(step3_features, screening_df, mi_thresh=MI_THRESH):
    """
    Fallback for N < 30 — use MI scores from Step 2 directly.
    No stationarity or linearity assumption required.
    """
    print(f"  Using MI scores from Step 2 (N too small for Granger)")
    causal_features   = []
    causality_results = []

    for col in step3_features:
        row     = screening_df[screening_df["feature"] == col]
        if row.empty:
            continue
        best_mi  = row["max_mi"].values[0]
        best_lag = row["best_lag_mi"].values[0]
        causal   = best_mi >= mi_thresh

        sig    = "✅" if causal else "❌"
        print(f"  {col}: MI={best_mi:.4f} at lag={best_lag}  {sig}")

        causality_results.append({
            "feature" : col,
            "best_lag": best_lag,
            "best_mi" : best_mi,
            "causal"  : causal,
        })

        if causal:
            causal_features.append({
                "feature" : col,
                "best_lag": best_lag,
            })

    return causal_features, pd.DataFrame(causality_results)


# Run the correct method
if causality_method == "granger":
    causal_features, causality_df = run_granger(
        step3_features, df_stationary, target_stat,
        max_lag=3, p_thresh=0.05,
    )
elif causality_method == "transfer_entropy":
    causal_features, causality_df = run_transfer_entropy(
        step3_features, df_stationary, target_stat,
        max_lag=3, te_thresh=0.01,
    )
else:
    # mi_only — small sample fallback
    causal_features, causality_df = run_mi_selection(
        step3_features, screening_df,
    )

# Summary
print("\n" + "=" * 60)
print("CAUSALITY RESULTS SUMMARY")
print("=" * 60)
if not causality_df.empty:
    sort_col = ("best_p"  if causality_method == "granger"
                else "best_te" if causality_method == "transfer_entropy"
                else "best_mi")
    sort_asc = causality_method == "granger"
    print(causality_df.sort_values(sort_col, ascending=sort_asc).to_string(index=False))

print(f"\nFeatures passing causality : {len(causal_features)}")
for item in causal_features:
    print(f"  {item['feature']:45s} → best lag = {item['best_lag']}")

# Fall back to step3 features if causality test dropped everything
if len(causal_features) == 0:
    print("\n⚠  No features passed causality test")
    print("   Falling back to Step 3 features for grid search")
    causal_features = [
        {"feature": f, "best_lag": int(screening_df[screening_df["feature"]==f]["best_lag_mi"].values[0])}
        for f in step3_features
    ]
    print(f"   Using: {[c['feature'] for c in causal_features]}")

# Extract final feature list + suggested lags from causality
final_features     = [c["feature"]  for c in causal_features]
suggested_lag_dict = {c["feature"]: c["best_lag"] for c in causal_features}


# ================================================================
# STEP 6: SARIMAX ROLLING GRID SEARCH
# ================================================================

print("\n" + "=" * 60)
print("STEP 6: SARIMAX ROLLING GRID SEARCH")
print("=" * 60)

# ── Build lag search space ─────────────────────────────────────
# Centre around suggested lags from Step 5 but test ±1 around them
def build_lag_range(suggested_lag, max_lag=3):
    candidates = set()
    for delta in (-1, 0, 1):
        l = suggested_lag + delta
        if 0 <= l <= max_lag:
            candidates.add(l)
    return tuple(sorted(candidates))

lag_ranges = {
    feat: build_lag_range(suggested_lag_dict[feat])
    for feat in final_features
}

print(f"\nFinal features entering grid search : {len(final_features)}")
for feat in final_features:
    print(f"  {feat:45s} lag search: {lag_ranges[feat]}")

# ── ARIMA order search space ───────────────────────────────────
# d_range informed by Step 1 ADF results
P_RANGE  = (1, 2, 3)
D_RANGE  = tuple(sorted({0, min(d_max, 1)}))  # 0 and/or 1 from ADF
Q_RANGE  = (0, 1)
TREND_OPTIONS = ("n",)
TARGET_LAG    = 1       # Gross_Credit_Losses lag — always included in exog
TOP_K         = 20
RANK_BY       = "smape" # "smape" | "mape" | "mae" | "rmse"
MIN_TRAIN     = 10


def full_grid_search_rolling(
    base_df,
    features,
    target_col,
    train_range,
    test_range,
    lag_ranges,
    p_range     = (1, 2, 3),
    d_range     = (0, 1),
    q_range     = (0, 1),
    P_range     = (0,),
    D_range     = (0,),
    Q_range     = (0,),
    m_range     = (0,),
    trend_options = ("n",),
    target_lag  = 1,
    rank_by     = "smape",
    top_k       = 20,
    min_train   = 10,
):
    # ── Build ALL lag columns once ─────────────────────────────
    work = base_df.copy()
    work.index = pd.to_datetime(work.index).date

    # collect all unique lags needed across features
    all_lags = set()
    for lags in lag_ranges.values():
        all_lags.update(lags)

    for feat in features:
        for L in all_lags:
            col = f"{feat}__lag{L}"
            work[col] = work[feat].shift(L) if L > 0 else work[feat]

    target_lag_col = f"{target_col}__lag{target_lag}"
    work[target_lag_col] = work[target_col].shift(target_lag)

    t0, t1 = map(datetime.date.fromisoformat, train_range)
    s0, s1 = map(datetime.date.fromisoformat, test_range)

    # ── Build search spaces ────────────────────────────────────
    # Per-feature lag options → cartesian product
    per_feature_lags = [lag_ranges[f] for f in features]
    lag_combos       = list(itertools.product(*per_feature_lags))

    order_combos    = list(itertools.product(p_range, d_range, q_range))
    seasonal_combos = list(itertools.product(P_range, D_range, Q_range, m_range))

    total = (len(lag_combos) * len(order_combos) *
             len(seasonal_combos) * len(trend_options))

    print(f"\n  Lag combos      : {len(lag_combos)}")
    print(f"  Order combos    : {len(order_combos)}")
    print(f"  Seasonal combos : {len(seasonal_combos)}")
    print(f"  Trend options   : {len(trend_options)}")
    print(f"  Total fits      : {total:,}  (each × len(test) refits)")
    print(f"  d_range         : {d_range}  ← informed by ADF Step 1")

    results = []
    best    = {rank_by: np.inf}
    pbar    = tqdm(total=total, desc="grid search")

    for lag_combo in lag_combos:

        feat_exog_cols = [
            f"{feat}__lag{L}"
            for feat, L in zip(features, lag_combo)
        ]
        exog_cols = feat_exog_cols + [target_lag_col]

        # slice clean data for this lag combo
        clean = work.dropna(subset=exog_cols + [target_col])
        idx   = pd.Index(clean.index)
        train = clean[(idx >= t0) & (idx <= t1)]
        test  = clean[(idx >= s0) & (idx <= s1)]

        if len(train) < min_train or len(test) == 0:
            pbar.update(
                len(order_combos) * len(seasonal_combos) * len(trend_options)
            )
            continue

        for order in order_combos:
            for seasonal_order in seasonal_combos:
                for trend in trend_options:
                    pbar.update(1)
                    try:
                        predictions = []
                        history_y   = train[target_col].astype(float).copy()
                        history_df  = train.copy()

                        for i in range(len(test)):

                            # refit on growing history
                            model = SARIMAX(
                                history_y,
                                exog=history_df[exog_cols].astype(float),
                                order=order,
                                seasonal_order=seasonal_order,
                                trend=trend,
                                enforce_stationarity=False,
                                enforce_invertibility=False,
                            ).fit(disp=False)

                            # build new_row exog from HISTORY ONLY
                            # never from test actuals — mirrors your notebook
                            new_row = test.iloc[[i]].copy()

                            for col in feat_exog_cols:
                                new_row[col] = history_df[col].iloc[-1]

                            # target lag updates with latest predicted value
                            new_row[target_lag_col] = history_y.iloc[-1]

                            # one-step forecast
                            pred = model.get_forecast(
                                steps=1,
                                exog=new_row[exog_cols].astype(float),
                            ).predicted_mean.iloc[0]

                            predictions.append(pred)

                            # update history
                            history_y = pd.concat([
                                history_y,
                                pd.Series([pred], index=[test.index[i]])
                            ])
                            new_row[target_col]     = pred
                            new_row[target_lag_col] = history_y.iloc[-2]
                            history_df = pd.concat([history_df, new_row])

                        # score
                        actual = test[target_col].astype(float).values
                        fc     = np.array(predictions)
                        m      = actual != 0
                        if not m.any():
                            continue

                        yt = actual[m]
                        yp = fc[m]

                        row = {
                            "smape"         : smape(yt, yp),
                            "mape"          : mape_pct(yt, yp),
                            "mae"           : mae(yt, yp),
                            "rmse"          : rmse(yt, yp),
                            "order"         : order,
                            "seasonal_order": seasonal_order,
                            "trend"         : trend,
                            "lags"          : dict(zip(features, lag_combo)),
                            "exog_cols"     : exog_cols,
                            "n_train"       : len(train),
                            "n_test"        : len(test),
                        }
                        results.append(row)

                        if row[rank_by] < best.get(rank_by, np.inf):
                            best = {
                                **row,
                                "forecast"  : fc,
                                "actual"    : actual,
                                "test_index": test.index,
                            }

                    except Exception as e:
                        results.append({
                            "smape"         : np.nan,
                            "mape"          : np.nan,
                            "mae"           : np.nan,
                            "rmse"          : np.nan,
                            "order"         : order,
                            "seasonal_order": seasonal_order,
                            "trend"         : trend,
                            "lags"          : dict(zip(features, lag_combo)),
                            "error"         : str(e)[:120],
                        })

    pbar.close()

    out = pd.DataFrame(results)

    if out.empty or out[rank_by].isna().all():
        print("⚠  All combinations failed. Sample errors:")
        if "error" in out.columns:
            print(out["error"].value_counts().head(5))
        return out, best

    out = (out.sort_values(rank_by, na_position="last")
              .reset_index(drop=True))

    print(f"\n{'='*60}")
    print(f"  GRID SEARCH RESULTS")
    print(f"{'='*60}")
    print(f"  Best {rank_by.upper():<6} : {best.get(rank_by,   np.nan):.4f}")
    print(f"  MAPE         : {best.get('mape',  np.nan):.4f}%")
    print(f"  SMAPE        : {best.get('smape', np.nan):.4f}%")
    print(f"  MAE          : {best.get('mae',   np.nan):>12,.0f}")
    print(f"  RMSE         : {best.get('rmse',  np.nan):>12,.0f}")
    print(f"  Order        : {best.get('order')}")
    print(f"  Seasonal     : {best.get('seasonal_order')}")
    print(f"  Trend        : {best.get('trend')}")
    print(f"  Lags         : {best.get('lags')}")
    print(f"  N train      : {best.get('n_train')}")
    print(f"  N test       : {best.get('n_test')}")

    return out.head(top_k), best


# ── Run grid search ────────────────────────────────────────────
top_results, best_model = full_grid_search_rolling(
    base_df       = df,                  # original df_merged — raw, unmodified
    features      = final_features,
    target_col    = TARGET,
    train_range   = (FIRST_INPUT_DATE, LAST_INPUT_DATE),
    test_range    = (FIRST_TEST_DATE,  LAST_TEST_DATE),
    lag_ranges    = lag_ranges,          # per-feature lag options from Step 5
    p_range       = P_RANGE,
    d_range       = D_RANGE,            # informed by Step 1 ADF
    q_range       = Q_RANGE,
    trend_options = TREND_OPTIONS,
    target_lag    = TARGET_LAG,
    rank_by       = RANK_BY,
    top_k         = TOP_K,
    min_train     = MIN_TRAIN,
)

print(f"\nTop {TOP_K} results:")
print(
    top_results[["smape", "mape", "mae", "rmse",
                 "order", "seasonal_order", "trend", "lags"]]
    .to_string(index=True)
)


# ================================================================
# FINAL PIPELINE SUMMARY
# ================================================================

print("\n" + "=" * 60)
print("FULL PIPELINE SUMMARY")
print("=" * 60)
print(f"Step 1 — After stationarity    : {len(ALL_FEATURES)} features checked")
print(f"         d_max found           : {d_max}")
print(f"         d_range for SARIMAX   : {D_RANGE}")
print(f"Step 2 — After corr + MI       : {len(step2_features)} features")
print(f"Step 3 — After VIF             : {len(step3_features)} features")
print(f"Step 4 — Linearity             : {'Non-linear → TE' if is_nonlinear else 'Linear → Granger'}")
if N_AVAILABLE < 30:
    print(f"         ⚠  N={N_AVAILABLE} overrode to MI selection")
print(f"Step 5 — After causality ({causality_method})")
print(f"         Causal features       : {len(final_features)}")
for feat in final_features:
    print(f"           → {feat:45s} suggested lag={suggested_lag_dict[feat]}")
print(f"Step 6 — Grid search winner")
print(f"         Best {RANK_BY.upper():<10}        : {best_model.get(RANK_BY, np.nan):.4f}")
print(f"         Best order            : {best_model.get('order')}")
print(f"         Best lags             : {best_model.get('lags')}")
print(f"\n{'='*60}")
print(f"USE THIS CONFIGURATION FOR INFERENCE:")
print(f"{'='*60}")
print(f"  features      = {final_features}")
print(f"  best_lags     = {best_model.get('lags')}")
print(f"  order         = {best_model.get('order')}")
print(f"  seasonal_order= {best_model.get('seasonal_order')}")
print(f"  trend         = '{best_model.get('trend')}'")
print(f"  target_lag    = {TARGET_LAG}")
