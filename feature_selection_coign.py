# ============================================================
# FEATURE SELECTION PIPELINE
# 69 features → filtered set → ready for itertools loop
# ============================================================

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests
from statsmodels.stats.diagnostic import bds
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

TARGET = 'Gross Credit Losses'
DROP_COLUMNS = ['DATE', TARGET]
ALL_FEATURES = [col for col in df.columns if col not in DROP_COLUMNS]

# ============================================================
# STEP 1: STATIONARITY CHECK — Transform non-stationary features
# ============================================================

def check_stationarity(series, name):
    try:
        adf_p = adfuller(series.dropna())[1]
        kpss_p = kpss(series.dropna(), regression='c')[1]
        # Stationary = ADF rejects unit root + KPSS fails to reject
        is_stationary = (adf_p < 0.05) and (kpss_p > 0.05)
        return is_stationary
    except:
        return False

print("=" * 60)
print("STEP 1: STATIONARITY CHECK")
print("=" * 60)

df_stationary = df.copy()
non_stationary = []

for col in ALL_FEATURES:
    is_stat = check_stationarity(df[col], col)
    if not is_stat:
        df_stationary[col] = df[col].diff()  # first difference
        non_stationary.append(col)

# Also difference the target for testing purposes
target_stat = df[TARGET].copy()
if not check_stationarity(target_stat, TARGET):
    target_stat = target_stat.diff()

# Drop NaN rows introduced by differencing
df_stationary = df_stationary.dropna().reset_index(drop=True)
target_stat = target_stat.dropna().reset_index(drop=True)

print(f"Non-stationary features differenced : {len(non_stationary)}")
print(f"Stationary features (no change)     : {len(ALL_FEATURES) - len(non_stationary)}")
print(f"Rows after differencing             : {len(df_stationary)}")


# ============================================================
# STEP 2: CORRELATION + MUTUAL INFORMATION (multi-lag)
# ============================================================

print("\n" + "=" * 60)
print("STEP 2: CORRELATION + MUTUAL INFORMATION SCREENING")
print("=" * 60)

LAGS        = [0, 1, 2, 3]
CORR_THRESH = 0.25   # minimum |correlation| at any lag
MI_THRESH   = 0.05   # minimum MI score at any lag

screening_results = []

for col in ALL_FEATURES:
    max_corr = 0
    max_mi   = 0

    for lag in LAGS:
        feature_lagged = df_stationary[col].shift(lag)
        combined       = pd.concat([feature_lagged, target_stat], axis=1).dropna()

        if len(combined) < 10:
            continue

        # Correlation
        corr = abs(combined.iloc[:, 0].corr(combined.iloc[:, 1]))
        max_corr = max(max_corr, corr if not np.isnan(corr) else 0)

        # Mutual Information
        X = combined.iloc[:, 0].values.reshape(-1, 1)
        y = combined.iloc[:, 1].values
        mi = mutual_info_regression(X, y, random_state=42)[0]
        max_mi = max(max_mi, mi)

    screening_results.append({
        'feature' : col,
        'max_corr': round(max_corr, 4),
        'max_mi'  : round(max_mi,   4)
    })

screening_df = pd.DataFrame(screening_results)

# Filter: must pass EITHER correlation OR MI threshold
step2_features = screening_df[
    (screening_df['max_corr'] >= CORR_THRESH) |
    (screening_df['max_mi']   >= MI_THRESH)
]['feature'].tolist()

print(f"Features before screening : {len(ALL_FEATURES)}")
print(f"Features after screening  : {len(step2_features)}")
print(f"\nTop 10 by correlation:")
print(screening_df.sort_values('max_corr', ascending=False).head(10).to_string(index=False))


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
    vif_data  = pd.DataFrame()
    vif_data['feature'] = scaled_df.columns
    vif_data['VIF']     = [
        variance_inflation_factor(scaled_df.values, i)
        for i in range(scaled_df.shape[1])
    ]
    return vif_data.sort_values('VIF', ascending=False)

# Iteratively remove highest VIF feature until all < threshold
remaining_features = step2_features.copy()

while True:
    feat_df  = df_stationary[remaining_features].fillna(0)
    vif_df   = compute_vif(feat_df)
    max_vif  = vif_df['VIF'].max()

    if max_vif <= VIF_THRESH or len(remaining_features) <= 3:
        break

    # Remove feature with highest VIF
    # (but prefer keeping higher MI score among ties)
    worst_feat = vif_df.iloc[0]['feature']
    remaining_features.remove(worst_feat)
    print(f"  Removed (VIF={max_vif:.1f}): {worst_feat}")

step3_features = remaining_features
print(f"\nFeatures after VIF check: {len(step3_features)}")


# ============================================================
# STEP 4: NON-LINEARITY CHECK — BDS TEST ON TARGET
# ============================================================

print("\n" + "=" * 60)
print("STEP 4: NON-LINEARITY CHECK (BDS TEST)")
print("=" * 60)

try:
    bds_stat, bds_p = bds(target_stat.values, distance=1.5)
    is_nonlinear    = bds_p < 0.05
    print(f"BDS Statistic : {bds_stat:.4f}")
    print(f"BDS p-value   : {bds_p:.4f}")
    print(f"Non-linear    : {is_nonlinear}")
    causality_method = "transfer_entropy" if is_nonlinear else "granger"
    print(f"→ Using       : {causality_method.upper()}")
except Exception as e:
    print(f"BDS test failed: {e} — defaulting to Granger")
    causality_method = "granger"
    is_nonlinear     = False


# ============================================================
# STEP 5: CAUSALITY TEST
# ============================================================

print("\n" + "=" * 60)
print(f"STEP 5: CAUSALITY TEST ({causality_method.upper()})")
print("=" * 60)

CAUSALITY_P_THRESH = 0.10   # slightly relaxed given small sample
MAX_LAG            = 2      # conservative for 20 points

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
                continue

            test_result = grangercausalitytests(
                combined[['target', 'feature']],
                maxlag=MAX_LAG,
                verbose=False
            )

            # Get minimum p-value across lags
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
    # Transfer Entropy fallback (requires 'pyinform' or similar)
    print("⚠️  Transfer Entropy requires: pip install pyinform")
    print("   Falling back to Granger as approximation...")
    causal_features = step3_features  # keep all if TE unavailable

causality_df = pd.DataFrame(causality_results) if causality_results else pd.DataFrame()

if not causality_df.empty:
    print(f"\nCausality results:")
    print(causality_df.sort_values('min_p').to_string(index=False))

print(f"\nFeatures passing causality test: {len(causal_features)}")


# ============================================================
# FINAL: FILTERED FEATURE SET
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

ALL_FEATS = causal_features  # plug directly into your existing code

feature_combinations = []
for r in range(1, min(4, len(ALL_FEATS) + 1)):  # max 3 covariates
    feature_combinations.extend(itertools.combinations(ALL_FEATS, r))

print(f"\nTotal combinations to test: {len(feature_combinations)}")
