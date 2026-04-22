# ============================================================
# STEP 1: STATIONARITY CHECK
# ============================================================

print("=" * 60)
print("STEP 1: STATIONARITY CHECK")
print("=" * 60)

def check_stationarity(series, name=""):
    try:
        s = series.dropna()
        if len(s) < 10:
            return False
        adf_p  = adfuller(s)[1]
        kpss_p = kpss(s, regression='c', nlags='auto')[1]
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
    print(f"  ⚠️  {col_name}: not stationary after {max_diffs} diffs — kept at d=1")
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
        status      = "✅" if is_now_stat else "⚠️  still non-stationary"
        if n_diffs == 2:
            print(f"  {status} {col} needed d={n_diffs}")

# Target differenced separately for testing only
target_stat              = df[TARGET].copy()
target_transformed, _    = make_stationary(target_stat, TARGET)
target_stat              = target_transformed

# Drop NaNs + align lengths
df_stationary = df_stationary.dropna().reset_index(drop=True)
target_stat   = target_stat.dropna().reset_index(drop=True)
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
# STEP 2: CORRELATION + MUTUAL INFORMATION
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

        corr     = abs(combined['feature'].corr(combined['target']))
        max_corr = max(max_corr, corr if not np.isnan(corr) else 0)

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
step2_features = screening_df[
    (screening_df['max_corr'] >= CORR_THRESH) |
    (screening_df['max_mi']   >= MI_THRESH)
]['feature'].tolist()

dropped = len(ALL_FEATURES) - len(step2_features)
print(f"Features before screening  : {len(ALL_FEATURES)}")
print(f"Features after screening   : {len(step2_features)}")
print(f"Features dropped           : {dropped}")
print(f"\nTop 15 by correlation:")
print(
    screening_df
    .sort_values('max_corr', ascending=False)
    .head(15)
    .to_string(index=False)
)


# ============================================================
# STEP 3: VIF REDUNDANCY CHECK
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

remaining_features = step2_features.copy()
mi_scores          = screening_df.set_index('feature')['max_mi']

while True:
    if len(remaining_features) <= 2:
        break

    feat_df = df_stationary[remaining_features].fillna(0)
    vif_df  = compute_vif(feat_df)
    max_vif = vif_df['VIF'].max()

    if max_vif <= VIF_THRESH:
        break

    top_vif    = vif_df[vif_df['VIF'] > VIF_THRESH]['feature'].tolist()
    worst_feat = min(top_vif, key=lambda f: mi_scores.get(f, 0))
    remaining_features.remove(worst_feat)
    print(f"  Removed (VIF={max_vif:.1f}, low MI): {worst_feat}")

step3_features = remaining_features
print(f"\nFeatures after VIF check   : {len(step3_features)}")
print(f"Remaining                  : {step3_features}")


# ============================================================
# STEP 4: NON-LINEARITY CHECK — MAJORITY VOTE
# ============================================================

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
        print(f"  ⚠️  AR(2) failed: {e}")
        ar_ok    = False

    # Fit OLS for RESET
    try:
        X         = add_constant(np.arange(len(s)))
        ols_model = OLS(s, X).fit()
        ols_ok    = True
    except Exception as e:
        print(f"  ⚠️  OLS failed: {e}")
        ols_ok    = False

    # Test 1: BDS
    try:
        _, p           = bds(s, distance=1.5)
        results['BDS'] = {'p': round(p, 4), 'nonlinear': p < 0.05}
        print(f"  BDS          : p={p:.4f}  →  "
              f"{'⚠️  non-linear' if p < 0.05 else '✅ linear'}")
    except Exception as e:
        print(f"  BDS          : failed ({e})")

    # Test 2: Ljung-Box on squared residuals
    if ar_ok:
        try:
            lb  = acorr_ljungbox(ar_resid**2, lags=4, return_df=True)
            p   = lb['lb_pvalue'].min()
            results['LjungBox'] = {'p': round(p, 4), 'nonlinear': p < 0.05}
            print(f"  Ljung-Box    : p={p:.4f}  →  "
                  f"{'⚠️  non-linear' if p < 0.05 else '✅ linear'}")
        except Exception as e:
            print(f"  Ljung-Box    : failed ({e})")

    # Test 3: RESET (OLS)
    if ols_ok:
        try:
            reset = linear_reset(ols_model, power=2, use_f=True)
            p     = reset.pvalue
            results['RESET'] = {'p': round(p, 4), 'nonlinear': p < 0.05}
            print(f"  RESET        : p={p:.4f}  →  "
                  f"{'⚠️  non-linear' if p < 0.05 else '✅ linear'}")
        except Exception as e:
            print(f"  RESET        : failed ({e})")

    # Majority vote
    if not results:
        print("  ⚠️  All tests failed — defaulting to LINEAR")
        return False, {}

    votes        = sum(v['nonlinear'] for v in results.values())
    total        = len(results)
    is_nonlinear = votes > total / 2

    print(f"\n  Tests run    : {total}")
    print(f"  Non-linear   : {votes}/{total} votes")
    print(f"  Decision     : "
          f"{'⚠️  NON-LINEAR → Transfer Entropy' if is_nonlinear else '✅ LINEAR → Granger'}")

    return is_nonlinear, results

is_nonlinear, nl_results = check_nonlinearity(target_stat)
causality_method = "transfer_entropy" if is_nonlinear else "granger"


# ============================================================
# STEP 5: CAUSALITY TEST
# ============================================================

print("\n" + "=" * 60)
print(f"STEP 5: CAUSALITY TEST ({causality_method.upper()})")
print("=" * 60)

def run_granger(step3_features, df_stationary, target_stat,
                max_lag=3, p_thresh=0.05):

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
                print(f"  Skipped {col}: insufficient rows ({len(combined)})")
                continue

            # Lag 0: correlation p-value
            _, corr_p   = scipy_stats.pearsonr(
                combined['feature'], combined['target']
            )
            lag_pvalues = {0: round(corr_p, 4)}

            # Lag 1,2,3: Granger
            test_result = grangercausalitytests(
                combined[['target', 'feature']],
                maxlag  = max_lag,
                verbose = False
            )
            for lag in range(1, max_lag + 1):
                lag_pvalues[lag] = round(
                    test_result[lag][0]['ssr_ftest'][1], 4
                )

            best_lag = min(lag_pvalues, key=lag_pvalues.get)
            best_p   = lag_pvalues[best_lag]

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


def run_transfer_entropy(step3_features, df_stationary, target_stat,
                         max_lag=3, te_thresh=0.01):

    if not TE_AVAILABLE:
        print("⚠️  pyinform unavailable — falling back to Granger")
        return run_granger(
            step3_features, df_stationary,
            target_stat,    max_lag=max_lag
        )

    BASE_FEATURES = [
        col for col in step3_features
        if not any(col.endswith(f'_lag{i}') for i in range(1, 10))
    ]
    print(f"Base features to test : {len(BASE_FEATURES)}")

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

    for col in BASE_FEATURES:
        try:
            lag_te_scores = {}

            for lag in range(0, max_lag + 1):
                feat_lagged = df_stationary[col].shift(lag)
                combined    = pd.concat(
                    [feat_lagged, target_stat], axis=1
                ).dropna()

                if len(combined) < 15:
                    continue

                feat_disc        = discretize(combined.iloc[:, 0])
                tgt_disc         = discretize(combined.iloc[:, 1])
                te_score         = pyinform_te(feat_disc, tgt_disc, k=1)
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


# Run correct method
if causality_method == "granger":
    causal_features, causality_df = run_granger(
        step3_features, df_stationary,
        target_stat,
        max_lag  = 3,
        p_thresh = 0.05
    )
else:
    causal_features, causality_df = run_transfer_entropy(
        step3_features, df_stationary,
        target_stat,
        max_lag   = 3,
        te_thresh = 0.01
    )

# Summary
print("\n" + "=" * 60)
print("CAUSALITY RESULTS")
print("=" * 60)
sort_col = 'best_p' if causality_method == 'granger' else 'best_te'
sort_asc = causality_method == 'granger'
print(causality_df.sort_values(sort_col, ascending=sort_asc).to_string(index=False))
print(f"\nFeatures passing causality : {len(causal_features)}")
for item in causal_features:
    print(f"  {item['feature']:45s} → best lag = {item['best_lag']}")


# ============================================================
# BUILD PAST COVARIATES
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
print(f"Expected length : {target_len}\n")
all_ok = True

for k, v in past_covariates.items():
    ok     = len(v) == target_len
    status = "✅" if ok else "❌"
    if not ok:
        all_ok = False
    print(f"  {status} {k:45s} length={len(v)}")

print(f"\n{'✅ All covariates ready for Chronos' if all_ok else '❌ Fix length mismatches'}")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PIPELINE SUMMARY")
print("=" * 60)
print(f"Step 1 — After stationarity    : {len(ALL_FEATURES)}")
print(f"Step 2 — After corr + MI       : {len(step2_features)}")
print(f"Step 3 — After VIF             : {len(step3_features)}")
print(f"Step 4 — Linearity             : {'Non-linear → TE' if is_nonlinear else 'Linear → Granger'}")
print(f"Step 5 — After causality       : {len(causal_features)}")
print(f"\nFINAL COVARIATES FOR CHRONOS:")
for k in past_covariates.keys():
    print(f"  → {k}")
