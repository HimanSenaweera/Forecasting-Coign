def plot_results_ensemble(pred_df_A, pred_df_B, pred_df_ensemble, metric: str) -> None:
    
    palette = {
        'Actual'                     : '#A2CEED',
        'Statistical Model prediction': 'red',
        'Model A (2,1,1)'            : 'orange',
        'Model B (2,1,2)'            : 'green',
        'Ensemble'                   : 'purple',
    }
    
    order = ['Actual', 'Statistical Model prediction', 
             'Model A (2,1,1)', 'Model B (2,1,2)', 'Ensemble']

    # ── Build combined dataframe ──
    # Rename forecast types to distinguish models
    df_A = pred_df_A.copy()
    df_A.loc[df_A['FORECAST_TYPE'] == 'Statistical Model prediction', 
             'FORECAST_TYPE'] = 'Model A (2,1,1)'

    df_B = pred_df_B.copy()
    df_B.loc[df_B['FORECAST_TYPE'] == 'Statistical Model prediction', 
             'FORECAST_TYPE'] = 'Model B (2,1,2)'

    df_ens = pred_df_ensemble.copy()
    df_ens.loc[df_ens['FORECAST_TYPE'] == 'Statistical Model prediction', 
               'FORECAST_TYPE'] = 'Ensemble'

    # Combine all — Actual only needed once
    pred_df_combined = pd.concat([
        pred_df_A[pred_df_A['FORECAST_TYPE'] == 'Actual'],  # Actual once
        df_A[df_A['FORECAST_TYPE'] == 'Model A (2,1,1)'],
        df_B[df_B['FORECAST_TYPE'] == 'Model B (2,1,2)'],
        df_ens[df_ens['FORECAST_TYPE'] == 'Ensemble'],
    ])

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
            linewidth=2,
            linestyle='--' if label != 'Actual' else '-'
        )

    x_labels = (
        pred_df_combined[pred_df_combined['FORECAST_TYPE'] == existing[0]]
        .sort_values('DATE')['month_year']
        .astype(str).tolist()
    )

    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=0)
    ax.set_xlabel('DATE')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=5)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f'{x:,.2f}')
    )
    plt.tight_layout()
    plt.show()


# ── Call it ──
plot_results_ensemble(
    pred_df_sarima_A,      # from Model A walk-forward
    pred_df_sarima_B,      # from Model B walk-forward
    pred_df_sarima_ens,    # from ensemble walk-forward
    'GROSS CREDIT LOSSES ($)'
)
