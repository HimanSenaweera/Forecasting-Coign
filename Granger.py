# ============================================================
# STEP 5: GRANGER CAUSALITY — BASE FEATURES + BEST LAG (0,1,2,3)
# ============================================================

from scipy import stats as scipy_stats

# Base features only — strip pre-created lag columns
BASE_FEATURES = [
    col for col in step3_features
    if not any(col.endswith(f'_lag{i}') for i in range(1, 10))
]

print(f"Base features to test : {len(BASE_FEATURES)}")

CAUSALITY_P_THRESH = 0.05
MAX_LAG            = 3      # Granger tests lag 1,2,3
causal_features    = []     # list of dicts {feature, best_lag}
causality_results  = []

for col in BASE_FEATURES:
    try:
        combined = pd.concat(
            [df_stationary[col], target_stat], axis=1
        ).dropna()
        combined.columns = ['feature', 'target']

        if len(combined) < 15:
            print(f"  Skipped {col}: insufficient rows ({len(combined)})")
            continue

        # ── Lag 0: contemporaneous correlation as proxy ──────
        corr, corr_p = scipy_stats.pearsonr(
            combined['feature'], 
            combined['target']
        )
        lag_pvalues = {0: round(corr_p, 4)}

        # ── Lag 1,2,3: Granger causality test ────────────────
        test_result = grangercausalitytests(
            combined[['target', 'feature']],
            maxlag=MAX_LAG,
            verbose=False
        )
        for lag in range(1, MAX_LAG + 1):
            lag_pvalues[lag] = round(
                test_result[lag][0]['ssr_ftest'][1], 4
            )

        # ── Find best lag ─────────────────────────────────────
        best_lag = min(lag_pvalues, key=lag_pvalues.get)
        best_p   = lag_pvalues[best_lag]

        # ── Print all lags for this feature ───────────────────
        print(f"\n{col}")
        for lag, p in lag_pvalues.items():
            sig    = "✅" if p < CAUSALITY_P_THRESH else "❌"
            marker = " ← best" if lag == best_lag else ""
            print(f"  lag {lag}: p={p:.4f} {sig}{marker}")

        causality_results.append({
            'feature' : col,
            'best_lag': best_lag,
            'best_p'  : best_p,
            'causal'  : best_p < CAUSALITY_P_THRESH,
            'all_lags': lag_pvalues
        })

        # ── Keep feature + lag info together ──────────────────
        if best_p < CAUSALITY_P_THRESH:
            causal_features.append({
                'feature' : col,
                'best_lag': best_lag
            })

    except Exception as e:
        print(f"  Skipped {col}: {e}")

# ── Summary ───────────────────────────────────────────────
print("\n" + "=" * 60)
causality_df = pd.DataFrame(causality_results).sort_values('best_p')
print(causality_df[['feature','best_lag','best_p','causal']].to_string(index=False))
print(f"\nFeatures passing causality : {len(causal_features)}")
for item in causal_features:
    print(f"  {item['feature']:40s} → best lag = {item['best_lag']}")
