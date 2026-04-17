# ══════════════════════════════════════════════
# ENSEMBLE: Model A (2,1,1) + Model B (2,1,2)
# ══════════════════════════════════════════════

# ── Model configs ──
exog_cols_A = [
    'Gross_Credit_Losses_lag1',
    'Payment_Rate_lag1',
    '90+ DQ Rate',
    'Finance Charges Rate'
]
order_A = (2, 1, 1)

exog_cols_B = [
    'Gross_Credit_Losses_lag1',
    'Payment_Rate_lag1',
    '120+ DQ Rate',
    'Finance Charges Rate'
]
order_B = (2, 1, 2)

# ── Initialize ──
train_y = train_df[TARGET_METRIC].astype(float)
test_y  = test_df[TARGET_METRIC].astype(float)

predictions_A        = []
predictions_B        = []
predictions_ensemble = []

history_y_A  = train_df[TARGET_METRIC].astype(float).copy()
history_df_A = train_df.copy()
history_y_B  = train_df[TARGET_METRIC].astype(float).copy()
history_df_B = train_df.copy()

# ── Walk-forward loop ──
for i in range(len(test_df)):

    # ────────────────────────────
    # Model A
    # ────────────────────────────
    model_A = SARIMAX(
        history_y_A,
        exog=history_df_A[exog_cols_A],
        order=order_A,
        seasonal_order=(0, 0, 0, 0),
        trend='n',
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)

    new_row_A = test_df.iloc[[i]].copy()
    new_row_A["Gross_Credit_Losses_lag1"] = history_y_A.iloc[-1]
    new_row_A["Payment_Rate_lag1"]        = history_df_A["Payment_Rate_lag1"].iloc[-1]
    new_row_A["90+ DQ Rate"]              = history_df_A["90+ DQ Rate"].iloc[-1]
    new_row_A["Finance Charges Rate"]     = history_df_A["Finance Charges Rate"].iloc[-1]

    pred_A = model_A.get_forecast(
        steps=1,
        exog=new_row_A[exog_cols_A]
    ).predicted_mean.iloc[0]

    predictions_A.append(pred_A)

    # Update Model A history
    history_y_A = pd.concat([
        history_y_A,
        pd.Series([pred_A], index=[test_df.index[i]])
    ])
    new_row_A[TARGET_METRIC] = pred_A
    history_df_A = pd.concat([history_df_A, new_row_A])

    # ────────────────────────────
    # Model B
    # ────────────────────────────
    model_B = SARIMAX(
        history_y_B,
        exog=history_df_B[exog_cols_B],
        order=order_B,
        seasonal_order=(0, 0, 0, 0),
        trend='n',
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)

    new_row_B = test_df.iloc[[i]].copy()
    new_row_B["Gross_Credit_Losses_lag1"] = history_y_B.iloc[-1]
    new_row_B["Payment_Rate_lag1"]        = history_df_B["Payment_Rate_lag1"].iloc[-1]
    new_row_B["120+ DQ Rate"]             = history_df_B["120+ DQ Rate"].iloc[-1]
    new_row_B["Finance Charges Rate"]     = history_df_B["Finance Charges Rate"].iloc[-1]

    pred_B = model_B.get_forecast(
        steps=1,
        exog=new_row_B[exog_cols_B]
    ).predicted_mean.iloc[0]

    predictions_B.append(pred_B)

    # Update Model B history
    history_y_B = pd.concat([
        history_y_B,
        pd.Series([pred_B], index=[test_df.index[i]])
    ])
    new_row_B[TARGET_METRIC] = pred_B
    history_df_B = pd.concat([history_df_B, new_row_B])

    # ────────────────────────────
    # Ensemble — inverse MAPE weighted
    # ────────────────────────────
    w_A = 1 / 26.66
    w_B = 1 / 20.44
    pred_ensemble = (
        (w_A * pred_A + w_B * pred_B) /
        (w_A + w_B)
    )
    predictions_ensemble.append(pred_ensemble)


# ══════════════════════════════════════════════
# BUILD PRED DFs
# ══════════════════════════════════════════════

def build_and_process(predictions, test_y, test_df):
    fc = pd.Series(predictions, index=test_df.index)
    pred_df = build_pred_df(fc.values, test_y)
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
    return pred_df, pred_df_copy

pred_df_A,   pred_df_A_copy   = build_and_process(predictions_A,        test_y, test_df)
pred_df_B,   pred_df_B_copy   = build_and_process(predictions_B,        test_y, test_df)
pred_df_ens, pred_df_ens_copy = build_and_process(predictions_ensemble, test_y, test_df)


# ══════════════════════════════════════════════
# PLOT FUNCTION
# ══════════════════════════════════════════════

def plot_results_ensemble(pred_df_A_copy, pred_df_B_copy, 
                          pred_df_ens_copy, metric: str) -> None:

    palette = {
        'Actual'                      : '#A2CEED',
        'Model A (2,1,1) — 26.66%'   : 'green',
        'Model B (2,1,2) — 20.44%'   : 'orange',
        'Ensemble'                    : 'red',
    }

    # Rename forecast types
    df_A = pred_df_A_copy.copy()
    df_A.loc[
        df_A['FORECAST_TYPE'] == 'Statistical Model prediction',
        'FORECAST_TYPE'
    ] = 'Model A (2,1,1) — 26.66%'

    df_B = pred_df_B_copy.copy()
    df_B.loc[
        df_B['FORECAST_TYPE'] == 'Statistical Model prediction',
        'FORECAST_TYPE'
    ] = 'Model B (2,1,2) — 20.44%'

    df_ens = pred_df_ens_copy.copy()
    df_ens.loc[
        df_ens['FORECAST_TYPE'] == 'Statistical Model prediction',
        'FORECAST_TYPE'
    ] = 'Ensemble'

    # Combine — Actual only once
    pred_df_combined = pd.concat([
        df_A[df_A['FORECAST_TYPE'] == 'Actual'],
        df_A[df_A['FORECAST_TYPE'] == 'Model A (2,1,1) — 26.66%'],
        df_B[df_B['FORECAST_TYPE'] == 'Model B (2,1,2) — 20.44%'],
        df_ens[df_ens['FORECAST_TYPE'] == 'Ensemble'],
    ])

    order    = ['Actual', 'Model A (2,1,1) — 26.66%',
                'Model B (2,1,2) — 20.44%', 'Ensemble']
    existing = [k for k in order
                if k in pred_df_combined['FORECAST_TYPE'].unique()]

    fig, ax = plt.subplots(figsize=(20, 6))
    ax.set_title(metric, fontsize=14, fontweight='bold')

    for label in existing:
        subset = pred_df_combined[
            pred_df_combined['FORECAST_TYPE'] == label
        ].sort_values('DATE')

        ax.plot(
            range(len(subset)),
            subset['METRIC_VALUE'].values,
            label=label,
            color=palette.get(label, None),
            linewidth=2.5 if label == 'Actual' else 2,
            linestyle='-' if label == 'Actual' else '--'
        )

    x_labels = (
        pred_df_combined[
            pred_df_combined['FORECAST_TYPE'] == existing[0]
        ].sort_values('DATE')['month_year']
        .astype(str).tolist()
    )

    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=0)
    ax.set_xlabel('DATE')
    ax.legend(loc='upper center', 
              bbox_to_anchor=(0.5, -0.25), ncol=4)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f'{x:,.2f}')
    )
    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════
# PLOT + EVALUATE
# ══════════════════════════════════════════════

plot_results_ensemble(
    pred_df_A_copy,
    pred_df_B_copy,
    pred_df_ens_copy,
    'GROSS CREDIT LOSSES ($)'
)

print("=" * 54)
print("Model A (2,1,1):")
evaluate_pred_df(pred_df_A,   "Statistical Model prediction")

print("=" * 54)
print("Model B (2,1,2):")
evaluate_pred_df(pred_df_B,   "Statistical Model prediction")

print("=" * 54)
print("Ensemble (weighted):")
evaluate_pred_df(pred_df_ens, "Statistical Model prediction")
