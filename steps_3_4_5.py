# ============================================================
# STEP 4: NON-LINEARITY CHECK
# ============================================================

print("\n" + "=" * 60)
print("STEP 4: NON-LINEARITY CHECK")
print("=" * 60)

def check_nonlinearity_bds(series):
    """BDS test — primary method"""
    try:
        from statsmodels.tsa.stattools import bds
        stat, p = bds(series.dropna().values, distance=1.5)
        return p < 0.05, round(p, 4), "BDS"
    except:
        pass
    try:
        from statsmodels.stats.stattools import bds
        stat, p = bds(series.dropna().values, distance=1.5)
        return p < 0.05, round(p, 4), "BDS"
    except:
        pass
    return None, None, None

def check_nonlinearity_proxy(series):
    """
    AR residual proxy — fallback if BDS unavailable
    Fits AR(2), checks if squared fitted values 
    correlate with residuals
    """
    from statsmodels.tsa.ar_model import AutoReg
    from scipy import stats

    s = series.dropna().values
    try:
        model     = AutoReg(s, lags=2).fit()
        resid     = model.resid
        fitted_sq = model.fittedvalues ** 2
        min_len   = min(len(resid), len(fitted_sq))
        _, p      = stats.pearsonr(resid[-min_len:], fitted_sq[-min_len:])
        return p < 0.05, round(p, 4), "AR_proxy"
    except Exception as e:
        print(f"  Proxy test failed: {e}")
        return False, 1.0, "AR_proxy"

# Run BDS first, fall back to proxy
is_nonlinear, test_p, test_method = check_nonlinearity_bds(target_stat)

if is_nonlinear is None:
    print("BDS unavailable — using AR residual proxy")
    is_nonlinear, test_p, test_method = check_nonlinearity_proxy(target_stat)

causality_method = "transfer_entropy" if is_nonlinear else "granger"

print(f"Test method    : {test_method}")
print(f"p-value        : {test_p}")
print(f"Non-linear     : {is_nonlinear}")
print(f"→ Using        : {causality_method.upper()}")


# ============================================================
# STEP 5A: GRANGER CAUSALITY (if linear)
# ============================================================

def run_granger(step3_features, df_stationary, target_stat,
                max_lag=3, p_thresh=0.05):
    
    from scipy import stats as scipy_stats

    BASE_FEATURES = [
        col for col in step3_features
        if not any(col.endswith(f'_lag{i}') for i in range(1, 10))
    ]

    print(f"Base features to test : {len(BASE_FEATURES)}")

    causal_features   = []
    causality_results = []

    for col in BASE_FEATURES:
        try:
            combined = pd.concat(
                [df_stationary[col], target_stat], axis=1
            ).dropna()
            combined.columns = ['feature', 'target']

            if len(combined) < 15:
                print(f"  Skipped {col}: insufficient rows")
                continue

            # Lag 0: correlation p-value as proxy
            _, corr_p       = scipy_stats.pearsonr(
                combined['feature'], combined['target']
            )
            lag_pvalues     = {0: round(corr_p, 4)}

            # Lag 1,2,3: Granger
            test_result = grangercausalitytests(
                combined[['target', 'feature']],
                maxlag=max_lag,
                verbose=False
            )
            for lag in range(1, max_lag + 1):
                lag_pvalues[lag] = round(
                    test_result[lag][0]['ssr_ftest'][1], 4
                )

            best_lag = min(lag_pvalues, key=lag_pvalues.get)
            best_p   = lag_pvalues[best_lag]

            # Print all lags
            print(f"\n  {col}")
            for lag, p in lag_pvalues.items():
                sig    = "✅" if p < p_thresh else "❌"
                marker = " ← best" if lag == best_lag else ""
                print(f"    lag {lag}: p={p:.4f} {sig}{marker}")

            causality_results.append({
                'feature' : col,
                'best_lag': best_lag,
                'best_p'  : best_p,
                'causal'  : best_p < p_thresh,
            })

            if best_p < p_thresh:
                causal_features.append({
                    'feature' : col,
                    'best_lag': best_lag
                })

        except Exception as e:
            print(f"  Skipped {col}: {e}")

    return causal_features, pd.DataFrame(causality_results)


# ============================================================
# STEP 5B: TRANSFER ENTROPY (if non-linear)
# ============================================================

def run_transfer_entropy(step3_features, df_stationary, target_stat,
                         max_lag=3, te_thresh=0.01):
    """
    Transfer Entropy — non-linear equivalent of Granger.
    Tests each feature at lag 0,1,2,3 against target.
    Keeps feature at best (highest TE) lag.
    """
    try:
        from pyinform import transfer_entropy as te
        TE_AVAILABLE = True
    except ImportError:
        TE_AVAILABLE = False
        print("⚠️  pyinform not installed: pip install pyinform")
        print("   Falling back to Granger as approximation")

    if not TE_AVAILABLE:
        print("   Running Granger instead...")
        return run_granger(
            step3_features, df_stationary, 
            target_stat, max_lag=max_lag
        )

    BASE_FEATURES = [
        col for col in step3_features
        if not any(col.endswith(f'_lag{i}') for i in range(1, 10))
    ]

    print(f"Base features to test : {len(BASE_FEATURES)}")

    causal_features   = []
    causality_results = []

    # Discretize target for TE (requires integer bins)
    def discretize(series, bins=10):
        s = series.dropna().values
        s_min, s_max = s.min(), s.max()
        if s_max == s_min:
            return np.zeros(len(s), dtype=int)
        binned = ((s - s_min) / (s_max - s_min) * (bins - 1)).astype(int)
        return np.clip(binned, 0, bins - 1)

    target_disc = discretize(target_stat)

    for col in BASE_FEATURES:
        try:
            feat_series = df_stationary[col].dropna()

            lag_te_scores = {}

            for lag in range(0, max_lag + 1):
                # Shift feature by lag
                feat_lagged = df_stationary[col].shift(lag)
                combined    = pd.concat(
                    [feat_lagged, target_stat], axis=1
                ).dropna()

                if len(combined) < 15:
                    continue

                feat_disc   = discretize(combined.iloc[:, 0])
                tgt_disc    = discretize(combined.iloc[:, 1])

                # TE: how much does feature reduce uncertainty in target
                te_score         = te(feat_disc, tgt_disc, k=1)
                lag_te_scores[lag] = round(float(te_score), 6)

            if not lag_te_scores:
                continue

            # Best lag = highest TE score
            best_lag = max(lag_te_scores, key=lag_te_scores.get)
            best_te  = lag_te_scores[best_lag]

            # Print all lags
            print(f"\n  {col}")
            for lag, score in lag_te_scores.items():
                sig    = "✅" if score > te_thresh else "❌"
                marker = " ← best" if lag == best_lag else ""
                print(f"    lag {lag}: TE={score:.6f} {sig}{marker}")

            causality_results.append({
                'feature' : col,
                'best_lag': best_lag,
                'best_te' : best_te,
                'causal'  : best_te > te_thresh,
            })

            if best_te > te_thresh:
                causal_features.append({
                    'feature' : col,
                    'best_lag': best_lag
                })

        except Exception as e:
            print(f"  Skipped {col}: {e}")

    return causal_features, pd.DataFrame(causality_results)


# ============================================================
# RUN STEP 5 BASED ON STEP 4 RESULT
# ============================================================

print("\n" + "=" * 60)
print(f"STEP 5: CAUSALITY TEST ({causality_method.upper()})")
print("=" * 60)

if causality_method == "granger":
    causal_features, causality_df = run_granger(
        step3_features,
        df_stationary,
        target_stat,
        max_lag  = 3,
        p_thresh = 0.05
    )
else:
    causal_features, causality_df = run_transfer_entropy(
        step3_features,
        df_stationary,
        target_stat,
        max_lag    = 3,
        te_thresh  = 0.01
    )

# Summary
print("\n" + "=" * 60)
print(causality_df.sort_values(
    'best_p' if causality_method == 'granger' else 'best_te',
    ascending = causality_method == 'granger'
).to_string(index=False))

print(f"\nFeatures passing causality : {len(causal_features)}")
for item in causal_features:
    print(f"  {item['feature']:45s} → best lag = {item['best_lag']}")


# ============================================================
# BUILD PAST COVARIATES FROM CAUSALITY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("BUILDING PAST COVARIATES")
print("=" * 60)

past_covariates = {}

for item in causal_features:
    col      = item['feature']
    best_lag = item['best_lag']

    feat_series  = pd.Series(train_df[col].values)
    feat_lagged  = feat_series.shift(best_lag)
    feat_aligned = feat_lagged.iloc[target_index]

    key                  = f"{col}_lag{best_lag}" if best_lag > 0 else col
    past_covariates[key] = feat_aligned.fillna(0).tolist()

# Verify lengths
target_len = len(train_series_smoothed)
print(f"Expected length : {target_len}")
all_ok = True
for k, v in past_covariates.items():
    ok     = len(v) == target_len
    status = "✅" if ok else "❌"
    if not ok:
        all_ok = False
    print(f"  {status} {k:45s} length={len(v)}")

if all_ok:
    print("\n✅ All covariates aligned — ready for Chronos")
else:
    print("\n❌ Length mismatch — check target_index alignment")
