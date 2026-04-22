# ============================================================
# STEP 4: NON-LINEARITY CHECK — MAJORITY VOTE
# ============================================================

print("\n" + "=" * 60)
print("STEP 4: NON-LINEARITY CHECK")
print("=" * 60)

def check_nonlinearity(series):
    """
    Majority vote across 3 tests:
    1. BDS        — detects ANY non-linear structure
    2. Ljung-Box  — detects ARCH/volatility clustering
    3. RESET      — detects smooth non-linearity
    
    Returns (is_nonlinear, results_dict)
    """
    from statsmodels.tsa.ar_model    import AutoReg
    from statsmodels.stats.diagnostic import acorr_ljungbox

    s       = series.dropna().values
    results = {}

    # Fit base AR(2) model — used by Ljung-Box and RESET
    try:
        ar_model = AutoReg(s, lags=2).fit()
        ar_resid = ar_model.resid
        ar_ok    = True
    except Exception as e:
        print(f"  ⚠️  AR(2) fit failed: {e}")
        ar_ok    = False

    # ── Test 1: BDS ─────────────────────────────────────────
    # Most powerful general non-linearity test
    # Tests if residuals are i.i.d — rejection = non-linear structure
    bds_done = False
    for bds_import in [
        "from statsmodels.tsa.stattools   import bds",
        "from statsmodels.stats.stattools import bds",
    ]:
        try:
            exec(bds_import, globals())
            _, p             = bds(s, distance=1.5)
            results['BDS']   = {'p': round(p, 4), 'nonlinear': p < 0.05}
            print(f"  BDS          : p={p:.4f}  →  "
                  f"{'⚠️  non-linear' if p < 0.05 else '✅ linear'}")
            bds_done = True
            break
        except:
            continue
    if not bds_done:
        print("  BDS          : unavailable")

    # ── Test 2: Ljung-Box on Squared Residuals ───────────────
    # Tests if variance is autocorrelated (ARCH effects)
    # → common in financial/credit time series
    if ar_ok:
        try:
            lb_result          = acorr_ljungbox(
                ar_resid ** 2, 
                lags      = 4,
                return_df = True
            )
            p                  = lb_result['lb_pvalue'].min()
            results['LjungBox'] = {'p': round(p, 4), 'nonlinear': p < 0.05}
            print(f"  Ljung-Box    : p={p:.4f}  →  "
                  f"{'⚠️  non-linear' if p < 0.05 else '✅ linear'}")
        except Exception as e:
            print(f"  Ljung-Box    : failed ({e})")
    else:
        print("  Ljung-Box    : skipped (AR fit failed)")

    # ── Test 3: RESET Test ───────────────────────────────────
    # Tests if powers of fitted values improve the model
    # → detects smooth non-linearity missed by BDS/LjungBox
    if ar_ok:
        try:
            from statsmodels.stats.diagnostic import linear_reset
            reset              = linear_reset(ar_model, power=2, use_f=True)
            p                  = reset.pvalue
            results['RESET']   = {'p': round(p, 4), 'nonlinear': p < 0.05}
            print(f"  RESET        : p={p:.4f}  →  "
                  f"{'⚠️  non-linear' if p < 0.05 else '✅ linear'}")
        except Exception as e:
            print(f"  RESET        : failed ({e})")
    else:
        print("  RESET        : skipped (AR fit failed)")

    # ── Majority Vote ────────────────────────────────────────
    if not results:
        print("\n  ⚠️  All tests failed — defaulting to LINEAR")
        return False, {}

    votes        = sum(v['nonlinear'] for v in results.values())
    total        = len(results)
    is_nonlinear = votes > total / 2

    print(f"\n  Tests run    : {total}")
    print(f"  Non-linear   : {votes}/{total} votes")
    print(f"  Decision     : "
          f"{'⚠️  NON-LINEAR' if is_nonlinear else '✅ LINEAR'}")
    print(f"  → Using      : "
          f"{'TRANSFER ENTROPY' if is_nonlinear else 'GRANGER'}")

    return is_nonlinear, results


# Run the check
is_nonlinear, nl_results = check_nonlinearity(target_stat)
causality_method = "transfer_entropy" if is_nonlinear else "granger"


# ============================================================
# STEP 5: CAUSALITY TEST
# ============================================================

print("\n" + "=" * 60)
print(f"STEP 5: CAUSALITY TEST ({causality_method.upper()})")
print("=" * 60)


# ── Step 5A: Granger (linear) ────────────────────────────────

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
                print(f"  Skipped {col}: insufficient rows ({len(combined)})")
                continue

            # Lag 0: correlation p-value as proxy
            _, corr_p     = scipy_stats.pearsonr(
                combined['feature'], combined['target']
            )
            lag_pvalues   = {0: round(corr_p, 4)}

            # Lag 1,2,3: Granger F-test p-values
            test_result   = grangercausalitytests(
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


# ── Step 5B: Transfer Entropy (non-linear) ───────────────────

def run_transfer_entropy(step3_features, df_stationary, target_stat,
                         max_lag=3, te_thresh=0.01):

    try:
        from pyinform import transfer_entropy as pyinform_te
        TE_AVAILABLE = True
        print("✅ pyinform available — running Transfer Entropy")
    except ImportError:
        TE_AVAILABLE = False
        print("⚠️  pyinform not installed: pip install pyinform")
        print("   Falling back to Granger as approximation")

    if not TE_AVAILABLE:
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
        """Discretize continuous series into integer bins for TE"""
        s              = series.dropna().values
        s_min, s_max   = s.min(), s.max()
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


# ── Run correct method ───────────────────────────────────────

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
        max_lag   = 3,
        te_thresh = 0.01
    )


# ── Summary ──────────────────────────────────────────────────

print("\n" + "=" * 60)
print("CAUSALITY RESULTS")
print("=" * 60)

sort_col = 'best_p' if causality_method == 'granger' else 'best_te'
sort_asc = causality_method == 'granger'

print(causality_df.sort_values(
    sort_col, ascending=sort_asc
).to_string(index=False))

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

print(f"\n{'✅ All covariates ready for Chronos' if all_ok else '❌ Fix length mismatches before proceeding'}")


# ============================================================
# FINAL PIPELINE SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PIPELINE SUMMARY")
print("=" * 60)
print(f"Step 1 — After stationarity     : {len(ALL_FEATURES)}")
print(f"Step 2 — After corr + MI        : {len(step2_features)}")
print(f"Step 3 — After VIF              : {len(step3_features)}")
print(f"Step 4 — Non-linearity          : {'Non-linear → TE' if is_nonlinear else 'Linear → Granger'}")
print(f"Step 5 — After causality test   : {len(causal_features)}")
print(f"\nFINAL COVARIATES FOR CHRONOS:")
for k in past_covariates.keys():
    print(f"  → {k}")
