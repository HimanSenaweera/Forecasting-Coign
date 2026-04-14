results = []

for r in range(1, 4):
    for combo in combinations(optional, r):
        
        exog_test = anchor + list(combo)
        
        # Each combo gets its own clean dataset
        train_combo, test_combo = build_dataset_for_combo(
            base_df, exog_test
        )
        
        history_y_temp  = train_combo[TARGET_METRIC].astype(float).copy()
        history_df_temp = train_combo.copy()
        
        try:
            # ── KEY FIX: find best order for THIS combo ──
            best_order = get_best_order(
                history_y_temp,
                history_df_temp,
                exog_test
            )
            print(f"\nCombo : {exog_test}")
            print(f"Order : {best_order}")
            print(f"Train rows: {len(train_combo)}")
            
            preds = []
            
            for i in range(len(test_combo)):
                
                model = SARIMAX(
                    history_y_temp,
                    exog=history_df_temp[exog_test],
                    order=best_order,        # ← uses combo-specific order
                    trend='n',               # ← matches auto_arima trend=None
                    enforce_stationarity=False,
                    enforce_invertibility=False
                ).fit(disp=False)
                
                new_row = test_combo.iloc[[i]].copy()
                
                # Update lag features dynamically
                for feat in exog_test:
                    if feat in LAG_FEATURE_CONFIG:
                        source_col, lag_n = LAG_FEATURE_CONFIG[feat]
                        if lag_n == 1:
                            new_row[feat] = history_y_temp.iloc[-1] \
                                if source_col == TARGET_METRIC \
                                else history_df_temp[source_col].iloc[-1]
                        else:
                            new_row[feat] = history_df_temp[source_col].iloc[-lag_n]
                
                exog_step = new_row[exog_test]
                pred = model.get_forecast(
                    steps=1, exog=exog_step
                ).predicted_mean.iloc[0]
                
                preds.append(pred)
                
                # Walk forward update
                history_y_temp = pd.concat([
                    history_y_temp,
                    pd.Series([pred], index=[test_combo.index[i]])
                ])
                new_row[TARGET_METRIC] = pred
                history_df_temp = pd.concat([history_df_temp, new_row])
            
            actuals   = test_combo[TARGET_METRIC].values
            preds_arr = np.array(preds)
            mape = np.mean(
                np.abs((actuals - preds_arr) / actuals)
            ) * 100
            
            results.append({
                'features':   exog_test,
                'order':      best_order,   # ← track order too
                'mape':       mape,
                'train_rows': len(train_combo),
            })
            print(f"MAPE: {mape:.2f}%")
            
        except Exception as e:
            print(f"Failed for {exog_test}: {e}")
            continue

# Results
results_df = pd.DataFrame(results).sort_values('mape')
print("\nTop 5 Combinations:")
print(results_df[['features', 'order', 'mape', 'train_rows']].head())
