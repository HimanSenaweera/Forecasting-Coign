from pmdarima import auto_arima

def get_best_order(history_y, history_df, exog_cols):
    """
    Finds best ARIMA order for THIS specific
    feature combination using auto_arima.
    """
    model = auto_arima(
        history_y,
        exogenous=history_df[exog_cols],
        seasonal=False,
        stepwise=True,
        suppress_warnings=True,
        error_action='ignore',
        trend=None,
    )
    return model.order  # returns (p, d, q)
