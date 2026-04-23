import itertools, datetime, warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")


# ── exact metric functions from your notebook ────────────────────────
def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(
        np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100
    )

def mape_pct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return mean_absolute_percentage_error(y_true, y_pred) * 100

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return mean_absolute_error(y_true, y_pred)

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(mean_squared_error(y_true, y_pred))


def full_grid_search_rolling(
    base_df,
    features,               # raw feature names — NOT lagged yet
    target_col,
    train_range,            # (first_input_date, last_input_date) ISO strings
    test_range,             # (first_test_date,  last_test_date)  ISO strings
    lags=(0, 1, 2, 3),
    target_lag=1,           # lag applied to target col when used as exog
    p_range=(0, 1, 2, 3),
    d_range=(0, 1),
    q_range=(0, 1, 2),
    P_range=(0,),
    D_range=(0,),
    Q_range=(0,),
    m_range=(0,),
    trend_options=("n",),
    rank_by="smape",        # "smape" | "mape" | "mae" | "rmse"
    top_k=20,
    min_train=20,
):
    # ── 1. Build ALL lag columns once on work ────────────────────────
    work = base_df.copy()
    work.index = pd.to_datetime(work.index).date

    for feat in features:
        for L in lags:
            work[f"{feat}__lag{L}"] = work[feat].shift(L) if L > 0 else work[feat]

    # target lag column (e.g. Gross_Credit_Losses__lag1)
    target_lag_col = f"{target_col}__lag{target_lag}"
    work[target_lag_col] = work[target_col].shift(target_lag)

    t0, t1 = map(datetime.date.fromisoformat, train_range)
    s0, s1 = map(datetime.date.fromisoformat, test_range)

    # ── 2. Search spaces ─────────────────────────────────────────────
    lag_combos      = list(itertools.product(lags, repeat=len(features)))
    order_combos    = list(itertools.product(p_range, d_range, q_range))
    seasonal_combos = list(itertools.product(P_range, D_range, Q_range, m_range))

    total = (len(lag_combos) * len(order_combos) *
             len(seasonal_combos) * len(trend_options))
    print(f"Lag combos      : {len(lag_combos)}")
    print(f"Order combos    : {len(order_combos)}")
    print(f"Seasonal combos : {len(seasonal_combos)}")
    print(f"Trend options   : {len(trend_options)}")
    print(f"Total outer     : {total:,}  (each × len(test) refits)\n")

    results = []
    best = {rank_by: np.inf}
    pbar = tqdm(total=total, desc="grid search")

    for lag_combo in lag_combos:

        # columns picked for this lag combo (feature exog only)
        feat_exog_cols = [f"{feat}__lag{L}" for feat, L in zip(features, lag_combo)]
        # always include the target lag column
        exog_cols = feat_exog_cols + [target_lag_col]

        # ── slice clean data for this lag combo ──────────────────────
        clean = work.dropna(subset=exog_cols + [target_col])
        idx   = pd.Index(clean.index)
        train = clean[(idx >= t0) & (idx <= t1)]
        test  = clean[(idx >= s0) & (idx <= s1)]

        if len(train) < min_train or len(test) == 0:
            pbar.update(len(order_combos) * len(seasonal_combos) * len(trend_options))
            continue

        for order in order_combos:
            for seasonal_order in seasonal_combos:
                for trend in trend_options:
                    pbar.update(1)
                    try:
                        predictions = []

                        # ── rolling history — mirrors your notebook ───
                        history_y  = train[target_col].astype(float).copy()
                        history_df = train.copy()

                        for i in range(len(test)):

                            # refit on growing history
                            arimax_model = SARIMAX(
                                history_y,
                                exog=history_df[exog_cols].astype(float),
                                order=order,
                                seasonal_order=seasonal_order,
                                trend=trend,
                                enforce_stationarity=False,
                                enforce_invertibility=False,
                            ).fit(disp=False)

                            # ── build new_row exog from HISTORY only ──
                            # never touch test actuals — mirrors your notebook:
                            # new_row[col] = history_df[col].iloc[-1]
                            new_row = test.iloc[[i]].copy()

                            for col in feat_exog_cols:
                                # take last known value from history
                                new_row[col] = history_df[col].iloc[-1]

                            # target lag = last predicted/actual value in history_y
                            new_row[target_lag_col] = history_y.iloc[-1]

                            # ── one-step forecast ─────────────────────
                            pred = arimax_model.get_forecast(
                                steps=1,
                                exog=new_row[exog_cols].astype(float),
                            ).predicted_mean.iloc[0]

                            predictions.append(pred)

                            # ── update history with prediction ────────
                            history_y = pd.concat([
                                history_y,
                                pd.Series([pred], index=[test.index[i]])
                            ])

                            # update target col in new_row then append to history_df
                            new_row[target_col]    = pred
                            new_row[target_lag_col] = history_y.iloc[-2]  # previous pred
                            history_df = pd.concat([history_df, new_row])

                        # ── score ─────────────────────────────────────
                        actual = test[target_col].astype(float).values
                        fc     = np.array(predictions)
                        m      = actual != 0

                        if not m.any():
                            continue

                        yt = actual[m]
                        yp = fc[m]

                        row = {
                            "smape":          smape(yt, yp),
                            "mape":           mape_pct(yt, yp),
                            "mae":            mae(yt, yp),
                            "rmse":           rmse(yt, yp),
                            "order":          order,
                            "seasonal_order": seasonal_order,
                            "trend":          trend,
                            "lags":           dict(zip(features, lag_combo)),
                            "exog_cols":      exog_cols,
                            "n_train":        len(train),
                            "n_test":         len(test),
                        }
                        results.append(row)

                        if row[rank_by] < best.get(rank_by, np.inf):
                            best = {
                                **row,
                                "forecast":   fc,
                                "actual":     actual,
                                "test_index": test.index,
                            }

                    except Exception as e:
                        results.append({
                            "smape":          np.nan,
                            "mape":           np.nan,
                            "mae":            np.nan,
                            "rmse":           np.nan,
                            "order":          order,
                            "seasonal_order": seasonal_order,
                            "trend":          trend,
                            "lags":           dict(zip(features, lag_combo)),
                            "error":          str(e)[:120],
                        })

    pbar.close()

    out = pd.DataFrame(results)

    if out.empty or out[rank_by].isna().all():
        print("⚠ All combinations failed. Sample errors:")
        if "error" in out.columns:
            print(out["error"].value_counts().head(5))
        return out, best

    out = (out.sort_values(rank_by, na_position="last")
              .reset_index(drop=True))

    print(f"\n✓ Best {rank_by.upper()} : {best.get(rank_by, np.nan):.4f}")
    print(f"  MAPE           : {best.get('mape',  np.nan):.4f}%")
    print(f"  SMAPE          : {best.get('smape', np.nan):.4f}%")
    print(f"  MAE            : {best.get('mae',   np.nan):>14,.0f}")
    print(f"  RMSE           : {best.get('rmse',  np.nan):>14,.0f}")
    print(f"  Order          : {best.get('order')}")
    print(f"  Seasonal       : {best.get('seasonal_order')}")
    print(f"  Trend          : {best.get('trend')}")
    print(f"  Lags           : {best.get('lags')}")

    return out.head(top_k), best

features = [
    "90+ DQ Rate",
    "Payment Rate",
    "Finance Charges Rate",
    "Expected Loss Roll Ave 3M",
]

top, best = full_grid_search_rolling(
    base_df     = df_merged,
    features    = features,
    target_col  = TARGET_METRIC,
    train_range = (first_input_date, last_input_date),
    test_range  = (first_test_date,  last_test_date),
    lags        = (0, 1, 2, 3),
    target_lag  = 1,
    p_range     = (1, 2, 3),
    d_range     = (0,),
    q_range     = (0, 1),
    trend_options = ("n",),
    rank_by     = "smape",
    top_k       = 20,
)

top[["smape", "mape", "mae", "rmse", "order", "seasonal_order", "trend", "lags"]]
