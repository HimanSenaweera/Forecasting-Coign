# ============================================================
# FEATURE SELECTION PIPELINE — FULL FIXED CODE
# 69 features → filtered set → ready for itertools loop
# ============================================================

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
import itertools
import warnings
warnings.filterwarnings('ignore')

TARGET       = 'Gross Credit Losses'
DROP_COLUMNS = ['DATE', TARGET]
ALL_FEATURES = [col for col in df.columns if col not in DROP_COLUMNS]

# ============================================================
# STEP 1: STATIONARITY CHECK — Transform non-stationary features
# ============================================================

def check_stationarity(series, name=""):
    """
    Returns True if series is stationary using both ADF and KPSS.
    Stationary = ADF rejects unit root AND KPSS fails to reject stationarity.
    """
    try:
        s      = series.dropna()
        if len(s) < 10:
            return False
        adf_p  = adfuller(s)[1]
        kpss_p = kpss(s, regression='c', nlags='auto')[1]
        return (adf_p < 0.05) and (kpss_p > 0.05)
    except:
        return False


def make_stationary(series, col_name, max_diffs=2):
    """
    Differences series until stationary, up to max_diffs times.
    Returns (transformed_series, n_diffs_applied).
    """
    s = series.copy()
    for d in range(max_diffs + 1):
        if check_stationarity(s.dropna(), col_name):
            return s, d
        if d < max_diffs:
            s = s.diff()
    # Still not stationary — return best effort (first diff)
    print(f"  ⚠️  {col_name}: not stationary after {max_diffs} diffs — kept at d=1")
    return series.diff(), 1


print("=" * 60)
print("STEP 1: STATIONARITY CHECK")
print("=" * 60)

df_stationary  = df.copy()
non_stationary = []
diff_order     = {}

for col in ALL_FEATURES:
    if check_stationarity(df[col].dropna(), col):
        diff_order[col] = 0   # already stationary, no change
    else:
        transformed, n_diffs    = make_stationary(df[col], col)
        df_stationary[col]      = transformed
        diff_order[col]         = n_diffs
        non_stationary.append(col)

        # ✅ Verify it actually worked
        is_now_stat = check_stationarity(transformed.dropna(), col)
        status      = "✅" if is_now_stat else "⚠️  still non-stationary"
        if n_diffs == 2:
            print(f"  {status} {col} needed d={n_diffs}")

# Target — differenced separately for testing only
target_stat         = df[TARGET].copy()
target_transformed, target_diffs = make_stationary(target_stat, TARGET)
target_stat         = target_transformed

# Drop NaN rows introduced by differencing
df_stationary = df_stationary.dropna().reset_index(drop=True)
target_stat   = target_stat.dropna().reset_index(drop=True)

# Align lengths
min_len       = min(len(df_stationary), len(target_stat))
df_stationary = df_stationary.iloc[:min_len].reset_index(drop=True)
target_stat   = target_stat.iloc[:min_len].reset_index(drop=True)

d0 = sum(1 for v in diff_order.values() if v == 0)
d1 = sum(1 for v in diff_order.values() if v == 1)
d2 = sum(1 for v in diff_order.values() if v == 2)

print(f"Already stationary  (d=0) : {d0}")
print(f"Stationary after 1 diff   : {d1}")
print(f"Stationary after 2 diffs  : {d2}")
print(f"Rows after differencing   : {len(df_stationary)}")


# ============================================================
# STEP 2: CORRELATION + MUTUAL INFORMATION (multi-lag)
# ============================================================

print("\n" + "=" * 60)
print("STEP 2: CORRELATION + MUTUAL INFORMATION SCREENING")
print("=" * 60)

LAGS        = [0, 1, 2, 3]
CORR_THRESH = 0.25
MI_THRESH   = 0.05

screening_results = []

for col in ALL_FEATURES:
    max_corr = 0
    max_mi   = 0

    for lag in LAGS:
        feature_lagged = df_stationary[col].shift(lag)
        combined       = pd.concat(
            [feature_lagged, target_stat], axis=1
        ).dropna()
        combined.columns = ['feature', 'target']

        if len(combined) < 10:
            continue

        # Correlation
        corr     = abs(combined['feature'].corr(combined['target']))
        max_corr = max(max_corr, corr if not np.isnan(corr) else 0)

        # Mutual Information
        X      = combined['feature'].values.reshape(-1, 1)
        y      = combined['target'].values
        mi     = mutual_info_regression(X, y, random_state=42)[0]
        max_mi = max(max_mi, mi)

    screening_results.append({
        'feature' : col,
        'max_corr': round(max_corr, 4),
        'max_mi'  : round(max_mi,   4)
    })

screening_df   = pd.DataFrame(screening_results)

# Keep if EITHER correlation OR MI passes threshold
step2_features = screening_df[
    (screening_df['max_corr'] >= CORR_THRESH) |
    (screening_df['max_mi']   >= MI_THRESH)
]['feature'].tolist()

print(f"Features before screening : {len(ALL_FEATURES)}")
print(f"Features after screening  : {len(step2_features)}")
print(f"\nTop 15 by correlation:")
print(
    screening_df
    .sort_values('max_corr', ascending=False)
    .head(15)
    .to_string(index=False)
)


# ============================================================
# STEP 3: VIF — REDUNDANCY / MULTICOLLINEARITY CHECK
# ============================================================

print("\n" + "=" * 60)
print("STEP 3: VIF REDUNDANCY CHECK")
print("=" * 60)

VIF_THRESH = 5.0

def compute_vif(feature_df):
    scaler    = StandardScaler()
    scaled    = scaler.fit_transform(feature_df.fillna(0))
    scaled_df = pd.DataFrame(scaled, columns=feature_df.columns)
    vif_data  = pd.DataFrame({
        'feature': scaled_df.columns,
        'VIF'    : [
            variance_inflation_factor(scaled_df.values, i)
            for i in range(scaled_df.shape[1])
        ]
    })
    return vif_data.sort_values('VIF', ascending=False).reset_index(drop=True)


# Iteratively remove highest VIF until all below threshold
remaining_features = step2_features.copy()

while True:
    if len(remaining_features) <= 2:
        break
    feat_df = df_stationary[remaining_features].fillna(0)
    vif_df  = compute_vif(feat_df)
    max_vif = vif_df['VIF'].max()

    if max_vif <= VIF_THRESH:
        break

    # Among high-VIF features, drop the one with lowest MI score
    worst_feat = vif_df.iloc[0]['feature']

    # Prefer keeping higher MI score — check MI from Step 2
    mi_scores  = screening_df.set_index('feature')['max_mi']
    top_vif    = vif_df[vif_df['VIF'] > VIF_THRESH]['feature'].tolist()
    if top_vif:
        worst_feat = min(top_vif, key=lambda f: mi_scores.get(f, 0))

    remaining_features.remove(worst_feat)
    print(f"  Removed (VIF={max_vif:.1f}, low MI): {worst_feat}")

step3_features = remaining_features
print(f"\nFeatures after VIF check : {len(step3_features)}")
print(f"Remaining features       : {step3_features}")


# ============================================================
# STEP 4: NON-LINEARITY CHECK — BDS TEST ON TARGET
# ============================================================

print("\n" + "=" * 60)
print("STEP 4: NON-LINEARITY CHECK")
print("=" * 60)

def check_nonlinearity(series):
    """
    Proxy non-linearity test:
    Fits linear AR(2), checks if squared fitted values
    correlate with residuals → sign of non-linear structure.
    """
    from statsmodels.tsa.ar_model import AutoReg
    from scipy import stats

    s = series.dropna().values
    try:
        model      = AutoReg(s, lags=2).fit()
        resid      = model.resid
        fitted_sq  = model.fittedvalues ** 2
        min_len    = min(len(resid), len(fitted_sq))
        _, p       = stats.pearsonr(resid[-min_len:], fitted_sq[-min_len:])
        return p < 0.05, round(p, 4)
    except Exception as e:
        print(f"  Non-linearity test failed: {e}")
        return False, 1.0

# Try BDS first, fall back to AR residual test
try:
    from statsmodels.tsa.stattools import bds
    bds_stat, bds_p  = bds(target_stat.values, distance=1.5)
    is_nonlinear     = bds_p < 0.05
    print(f"BDS Statistic  : {bds_stat:.4f}")
    print(f"BDS p-value    : {bds_p:.4f}")
except ImportError:
    try:
        from statsmodels.stats.stattools import bds
        bds_stat, bds_p = bds(target_stat.values, distance=1.5)
        is_nonlinear    = bds_p < 0.05
        print(f"BDS p-value    : {bds_p:.4f}")
    except:
        print("BDS unavailable — using AR residual proxy test")
        is_nonlinear, bds_p = check_nonlinearity(target_stat)
        print(f"Proxy p-value  : {bds_p:.4f}")

causality_method = "transfer_entropy" if is_nonlinear else "granger"
print(f"Non-linear     : {is_nonlinear}")
print(f"→ Using        : {causality_method.upper()}")


# ============================================================
# STEP 5: CAUSALITY TEST
# ============================================================

print("\n" + "=" * 60)
print(f"STEP 5: CAUSALITY TEST ({causality_method.upper()})")
print("=" * 60)

CAUSALITY_P_THRESH = 0.10
MAX_LAG            = 2
causal_features    = []
causality_results  = []

if causality_method == "granger":
    for col in step3_features:
        try:
            combined = pd.concat(
                [df_stationary[col], target_stat], axis=1
            ).dropna()
            combined.columns = ['feature', 'target']

            if len(combined) < 15:
                print(f"  Skipped {col}: insufficient rows ({len(combined)})")
                continue

            test_result = grangercausalitytests(
                combined[['target', 'feature']],
                maxlag=MAX_LAG,
                verbose=False
            )

            # Minimum p-value across all lags
            min_p = min([
                test_result[lag][0]['ssr_ftest'][1]
                for lag in range(1, MAX_LAG + 1)
            ])

            causality_results.append({
                'feature': col,
                'min_p'  : round(min_p, 4),
                'causal' : min_p < CAUSALITY_P_THRESH
            })

            if min_p < CAUSALITY_P_THRESH:
                causal_features.append(col)

        except Exception as e:
            print(f"  Skipped {col}: {e}")

else:
    # Transfer Entropy fallback
    print("⚠️  Transfer Entropy requires: pip install pyinform")
    print("   Falling back to Granger as approximation...")
    causal_features = step3_features

if causality_results:
    causality_df = pd.DataFrame(causality_results).sort_values('min_p')
    print(f"\nCausality results:")
    print(causality_df.to_string(index=False))

print(f"\nFeatures passing causality : {len(causal_features)}")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PIPELINE SUMMARY")
print("=" * 60)
print(f"Step 1 — After stationarity transform : {len(ALL_FEATURES)}")
print(f"Step 2 — After correlation + MI       : {len(step2_features)}")
print(f"Step 3 — After VIF redundancy check   : {len(step3_features)}")
print(f"Step 5 — After causality test         : {len(causal_features)}")
print(f"\nFINAL FILTERED FEATURES:")
for i, f in enumerate(causal_features, 1):
    print(f"  {i:2d}. {f}")


# ============================================================
# READY FOR YOUR BRUTE FORCE LOOP
# ============================================================

ALL_FEATS            = causal_features
feature_combinations = []

for r in range(1, min(4, len(ALL_FEATS) + 1)):  # max 3 covariates
    feature_combinations.extend(itertools.combinations(ALL_FEATS, r))

print(f"\nTotal combinations to test : {len(feature_combinations)}")
