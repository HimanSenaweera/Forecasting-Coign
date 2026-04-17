for i in range(len(test_df)):

    # ── Model A ──
    model_A = SARIMAX(
        history_y_A,
        exog=history_df_A[exog_cols_A],
        order=order_A,
        trend='n',
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)

    new_row_A = test_df.iloc[[i]].copy()
    new_row_A["Gross_Credit_Losses_lag1"] = history_y_A.iloc[-1]
    new_row_A["Payment_Rate_lag1"] = history_df_A["Payment_Rate_lag1"].iloc[-1]
    new_row_A["90+ DQ Rate"] = history_df_A["90+ DQ Rate"].iloc[-1]
    new_row_A["Finance Charges Rate"] = history_df_A["Finance Charges Rate"].iloc[-1]

    pred_A = model_A.get_forecast(
        steps=1,
        exog=new_row_A[exog_cols_A]
    ).predicted_mean.iloc[0]

    # ── Model B ──
    model_B = SARIMAX(
        history_y_B,
        exog=history_df_B[exog_cols_B],
        order=order_B,
        trend='n',
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)

    new_row_B = test_df.iloc[[i]].copy()
    new_row_B["Gross_Credit_Losses_lag1"] = history_y_B.iloc[-1]
    new_row_B["Payment_Rate_lag1"] = history_df_B["Payment_Rate_lag1"].iloc[-1]
    new_row_B["120+ DQ Rate"] = history_df_B["120+ DQ Rate"].iloc[-1]
    new_row_B["Finance Charges Rate"] = history_df_B["Finance Charges Rate"].iloc[-1]

    pred_B = model_B.get_forecast(
        steps=1,
        exog=new_row_B[exog_cols_B]
    ).predicted_mean.iloc[0]

    # ── Ensemble ──
    w_A = 1 / 26.66
    w_B = 1 / 20.44
    pred_ensemble = (
        (w_A * pred_A + w_B * pred_B) /
        (w_A + w_B)
    )

    predictions_A.append(pred_A)
    predictions_B.append(pred_B)
    predictions_ensemble.append(pred_ensemble)

    # ── Update A history explicitly ──
    history_y_A = pd.concat([
        history_y_A,
        pd.Series([pred_A], index=[test_df.index[i]])
    ])
    new_row_A[TARGET_METRIC] = pred_A
    history_df_A = pd.concat([history_df_A, new_row_A])

    # ── Update B history explicitly ──
    history_y_B = pd.concat([
        history_y_B,
        pd.Series([pred_B], index=[test_df.index[i]])
    ])
    new_row_B[TARGET_METRIC] = pred_B
    history_df_B = pd.concat([history_df_B, new_row_B])
