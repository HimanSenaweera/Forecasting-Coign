# ============================================================
# BUILD PAST COVARIATES WITH LAGS
# ============================================================

# Define your features and their best lag from Granger
feature_lag_map = {
    'Chargeoff Accounts' : 1,
    'Inactive Accounts'  : 1,
    'Finance Charges'    : 1,
    'Payment Rate'       : 1,
    '30+ DQ Rate'        : 1,
    # add more features and their best lag here
}

past_covariates = {}

for col, lag in feature_lag_map.items():
    feat_series  = pd.Series(train_df[col].values)
    feat_lagged  = feat_series.shift(lag)
    feat_aligned = feat_lagged.iloc[target_index]
    feat_aligned = feat_aligned.fillna(0)
    
    key                  = f"{col}_lag{lag}" if lag > 0 else col
    past_covariates[key] = feat_aligned.tolist()
    print(f"Added: {key} (length={len(past_covariates[key])})")
