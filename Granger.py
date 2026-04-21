# ============================================================
# STEP 5: GRANGER CAUSALITY — BASE FEATURES + BEST LAG
# ============================================================

# Separate base features from lag features
BASE_FEATURES = [
    col for col in step3_features
    if not any(col.endswith(f'_lag{i}') for i in range(1, 10))
]

print(f"Base features to test: {len(BASE_FEATURES)}")

CAUSALITY_P_THRESH = 0.05
MAX_LAG            = 4      # test lags 1,2,3,4
causal_features    = []
causality_results  = []

for col in BASE_FEATURES:
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

        # Get p-value AND best lag separately
        lag_pvalues = {
            lag: test_result[lag][0]['ssr_ftest'][1]
            for lag in range(1, MAX_LAG + 1)
        }

        best_lag  = min(lag_pvalues, key=lag_pvalues.get)
        best_p    = lag_pvalues[best_lag]

        causality_results.append({
            'feature' : col,
            'best_lag': best_lag,   # ← which lag matters most
            'best_p'  : round(best_p, 4),
            'causal'  : best_p < CAUSALITY_P_THRESH,
            'all_lags': lag_pvalues
        })

        if best_p < CAUSALITY_P_THRESH:
            # Add the correct lag version of the feature
            if best_lag == 0:
                causal_features.append(col)
            else:
                lag_col = f"{col}_lag{best_lag}"
                # Use lag col if exists, else use base
                if lag_col in df.columns:
                    causal_features.append(lag_col)
                else:
                    causal_features.append(col)

    except Exception as e:
        print(f"  Skipped {col}: {e}")

# Results
causality_df = pd.DataFrame(causality_results).sort_values('best_p')
print(causality_df[['feature','best_lag','best_p','causal']].to_string(index=False))
print(f"\nCausal features with best lag: {causal_features}")
