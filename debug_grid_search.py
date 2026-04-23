# Run this BEFORE the full grid search to diagnose
def debug_grid_search(base_df, features, target_col, train_range, test_range,
                      lags=(0,1,2,3), target_lag=1,
                      order=(3,0,1), seasonal_order=(0,0,0,0), trend="n"):

    work = base_df.copy()
    work.index = pd.to_datetime(work.index).date

    print("=== Step 1: Column check ===")
    for feat in features:
        if feat not in work.columns:
            print(f"  ❌ MISSING: '{feat}'")
        else:
            print(f"  ✓ found  : '{feat}'  | NaNs: {work[feat].isna().sum()}")

    if target_col not in work.columns:
        print(f"  ❌ TARGET MISSING: '{target_col}'")
    else:
        print(f"  ✓ target : '{target_col}' | NaNs: {work[target_col].isna().sum()}")

    print("\n=== Step 2: Build lag columns ===")
    for feat in features:
        for L in lags:
            col = f"{feat}__lag{L}"
            work[col] = work[feat].shift(L) if L > 0 else work[feat]
            print(f"  built: {col} | NaNs: {work[col].isna().sum()}")

    target_lag_col = f"{target_col}__lag{target_lag}"
    work[target_lag_col] = work[target_col].shift(target_lag)
    print(f"  built: {target_lag_col} | NaNs: {work[target_lag_col].isna().sum()}")

    print("\n=== Step 3: Train/test split ===")
    t0, t1 = map(datetime.date.fromisoformat, train_range)
    s0, s1 = map(datetime.date.fromisoformat, test_range)

    lag_combo   = tuple([lags[0]] * len(features))   # simplest combo
    exog_cols   = [f"{feat}__lag{lag_combo[0]}" for feat in features] + [target_lag_col]

    print(f"  exog_cols: {exog_cols}")

    clean = work.dropna(subset=exog_cols + [target_col])
    idx   = pd.Index(clean.index)
    train = clean[(idx >= t0) & (idx <= t1)]
    test  = clean[(idx >= s0) & (idx <= s1)]

    print(f"  work rows  : {len(work)}")
    print(f"  clean rows : {len(clean)}  (after dropna)")
    print(f"  train rows : {len(train)}  ({t0} → {t1})")
    print(f"  test rows  : {len(test)}   ({s0} → {s1})")

    if len(train) == 0:
        print("  ❌ Train is empty — check train_range dates match your index")
        return
    if len(test) == 0:
        print("  ❌ Test is empty — check test_range dates match your index")
        return

    print("\n=== Step 4: Try one SARIMAX fit ===")
    try:
        model = SARIMAX(
            train[target_col].astype(float),
            exog=train[exog_cols].astype(float),
            order=order,
            seasonal_order=seasonal_order,
            trend=trend,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        print(f"  ✓ Model fit OK | AIC: {model.aic:.2f}")

        new_row = test.iloc[[0]].copy()
        for col in exog_cols:
            new_row[col] = train[col].iloc[-1]   # from history, not test

        pred = model.get_forecast(
            steps=1,
            exog=new_row[exog_cols].astype(float),
        ).predicted_mean.iloc[0]
        print(f"  ✓ First forecast: {pred:.4f}")
        print(f"  ✓ First actual  : {test[target_col].iloc[0]:.4f}")

    except Exception as e:
        print(f"  ❌ FAILED: {e}")

    print("\n=== Step 5: Index sample ===")
    print(f"  work.index dtype : {type(work.index[0])}")
    print(f"  work.index[:3]   : {work.index[:3].tolist()}")
    print(f"  t0 type          : {type(t0)}, value: {t0}")

debug_grid_search(
    base_df     = df_merged,
    features    = features,
    target_col  = TARGET_METRIC,
    train_range = (first_input_date, last_input_date),
    test_range  = (first_test_date,  last_test_date),
)
