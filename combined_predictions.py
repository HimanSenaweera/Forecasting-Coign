# ── Run BOTH models in the same 
#    walk-forward loop ──

# Model A config
exog_cols_A = [
    'Gross_Credit_Losses_lag1',
    'Payment_Rate_lag1',
    '90+ DQ Rate',
    'Finance Charges Rate'
]
order_A = (2, 1, 1)

# Model B config
exog_cols_B = [
    'Gross_Credit_Losses_lag1',
    'Payment_Rate_lag1',
    '120+ DQ Rate',
    'Finance Charges Rate'
]
order_B = (2, 1, 2)

predictions_A = []
predictions_B = []
predictions_ensemble = []

# Separate histories for each model
history_y_A  = train_df[TARGET_METRIC].astype(float).copy()
history_df_A = train_df.copy()
history_y_B  = train_df[TARGET_METRIC].astype(float).copy()
history_df_B = train_df.copy()

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
    # Option 1: Simple average
    pred_ensemble = (pred_A + pred_B) / 2

    # Option 2: Weighted — give more weight
    # to Model B (better MAPE)
    w_A = 1 / 26.66  # inverse MAPE weight
    w_B = 1 / 20.44
    pred_ensemble = (
        (w_A * pred_A + w_B * pred_B) / 
        (w_A + w_B)
    )

    predictions_A.append(pred_A)
    predictions_B.append(pred_B)
    predictions_ensemble.append(pred_ensemble)

    # ── Update histories separately ──
    for pred, hy, hdf, nr, cols in [
        (pred_A, history_y_A, history_df_A, new_row_A, exog_cols_A),
        (pred_B, history_y_B, history_df_B, new_row_B, exog_cols_B),
    ]:
        hy = pd.concat([
            hy, pd.Series([pred], index=[test_df.index[i]])
        ])
        nr[TARGET_METRIC] = pred
        hdf = pd.concat([hdf, nr])

# ── Compare all three ──
actuals = test_df[TARGET_METRIC].values

for name, preds in [
    ('Model A (2,1,1)', predictions_A),
    ('Model B (2,1,2)', predictions_B),
    ('Ensemble',        predictions_ensemble)
]:
    mape = np.mean(
        np.abs((actuals - np.array(preds)) / actuals)
    ) * 100
    print(f"{name}: MAPE = {mape:.2f}%")
