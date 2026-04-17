# ── Build ensemble pred_df same way as single model ──

# Step 1: Build pred_df
arimax_fc_ens = pd.Series(
    predictions_ensemble, index=test_df.index
)
pred_df_sarima_ens = build_pred_df(arimax_fc_ens.values, test_y)
pred_df_sarima_ens.loc[
    pred_df_sarima_ens['FORECAST_TYPE'] == 'Prediction',
    'FORECAST_TYPE'
] = 'Statistical Model prediction'

# Step 2: Take absolute values
pred_df_sarima_ens["METRIC_VALUE"] = (
    pred_df_sarima_ens["METRIC_VALUE"].abs()
)

# Step 3: Make copy + add month_year
pred_df_sarima_ens_copy = pred_df_sarima_ens.copy()
pred_df_sarima_ens_copy["METRIC_VALUE"] = (
    pred_df_sarima_ens_copy["METRIC_VALUE"]
)
pred_df_sarima_ens_copy['DATE'] = pd.to_datetime(
    pred_df_sarima_ens_copy['DATE']
)
pred_df_sarima_ens_copy['month_year'] = (
    pred_df_sarima_ens_copy['DATE'].dt.strftime('%b-%y')
)

# Do the same for Model A and Model B:
for fc, preds, name in [
    ('pred_df_sarima_A_copy', predictions_A, 'Model A'),
    ('pred_df_sarima_B_copy', predictions_B, 'Model B'),
]:
    fc_series = pd.Series(preds, index=test_df.index)
    pred_df = build_pred_df(fc_series.values, test_y)
    pred_df.loc[
        pred_df['FORECAST_TYPE'] == 'Prediction',
        'FORECAST_TYPE'
    ] = 'Statistical Model prediction'
    pred_df["METRIC_VALUE"] = pred_df["METRIC_VALUE"].abs()
    pred_df_copy = pred_df.copy()
    pred_df_copy['DATE'] = pd.to_datetime(pred_df_copy['DATE'])
    pred_df_copy['month_year'] = (
        pred_df_copy['DATE'].dt.strftime('%b-%y')
    )

# Step 4: Plot all three + evaluate
plot_results_ensemble(
    pred_df_sarima_A_copy,
    pred_df_sarima_B_copy,
    pred_df_sarima_ens_copy,
    'GROSS CREDIT LOSSES ($)'
)

# Step 5: Evaluate each
evaluate_pred_df(pred_df_sarima_A,   "Statistical Model prediction")
evaluate_pred_df(pred_df_sarima_B,   "Statistical Model prediction")
evaluate_pred_df(pred_df_sarima_ens, "Statistical Model prediction")
