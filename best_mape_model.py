import itertools, datetime, warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_percentage_error
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")


def full_grid_search(
    base_df,
    features,
    target_col,
    train_range,
    test_range,
    # ── lag search space ─────────────────────────────
    lags=(0, 1, 2, 3),
    target_lag=None,           # fixed lag for target in exog; None = exclude
    # ── ARIMA order search space ─────────────────────
    p_range=(0, 1, 2, 3),
    d_range=(0, 1),
    q_range=(0, 1, 2),
    # ── Seasonal order search space ──────────────────
    P_range=(0,),              # keep (0,) to skip seasonal search
    D_range=(0,),
    Q_range=(0,),
    m_range=(0,),              # 0 = no seasonality; try 12 for monthly
    # ── Misc ─────────────────────────────────────────
    trend_options=("n",),      # ("n", "c") to also try with constant
    top_k=20,
    min_train=20,
):
    """
    Grid search over:
      • every (feature -> lag) combination   [4^N combos for N features]
      • every (p, d, q) ARIMA order
      • every (P, D, Q, m) seasonal order
      • trend options

    Ranks all valid fits by test-set MAPE.
    Returns (top_k_df, best_dict).
    """

    # ── 1. Build all lag columns up front ────────────
    work = base_df.copy()
    work.index = pd.to_datetime(work.index).date

    for feat in features:
        for L in lags:
            col = f"{feat}__lag{L}"
            work[col] = work[feat].shift(L) if L > 0 else work[feat]

    if target_lag is not None:
        work[f"__TARGET__lag{target_lag}"] = work[target_col].shift(target_lag)

    t0, t1 = map(datetime.date.fromisoformat, train_range)
    s0, s1 = map(datetime.date.fromisoformat, test_range)

    # ── 2. Build search spaces ────────────────────────
    lag_combos      = list(itertools.product(lags, repeat=len(features)))
    order_combos    = list(itertools.product(p_range, d_range, q_range))
    seasonal_combos = list(itertools.product(P_range, D_range, Q_range, m_range))

    total = len(lag_combos) * len(order_combos) * len(seasonal_combos) * len(trend_options)
    print(f"Total combinations: {len(lag_combos)} lag × "
          f"{len(order_combos)} order × "
          f"{len(seasonal_combos)} seasonal × "
          f"{len(trend_options)} trend = {total:,}")

    results = []
    best = {"mape": np.inf}

    pbar = tqdm(total=total, desc="grid search")

    for lag_combo in lag_combos:
        # build exog column list for this lag combo
        exog_cols = [f"{feat}__lag{L}" for feat, L in zip(features, lag_combo)]
        if target_lag is not None:
            exog_cols.append(f"__TARGET__lag{target_lag}")

        clean = work.dropna(subset=exog_cols + [target_col])
        idx   = pd.Index(clean.index)
        train = clean[(idx >= t0) & (idx <= t1)]
        test  = clean[(idx >= s0) & (idx <= s1)]

        # skip this lag combo entirely if data is insufficient
        if len(train) < min_train or len(test) == 0:
            pbar.update(len(order_combos) * len(seasonal_combos) * len(trend_options))
            continue

        train_y = train[target_col].astype(float)
        test_y  = test[target_col].astype(float)
        train_x = train[exog_cols].astype(float)
        test_x  = test[exog_cols].astype(float)

        for order in order_combos:
            for seasonal_order in seasonal_combos:
                for trend in trend_options:
                    pbar.update(1)
                    try:
                        model = SARIMAX(
                            train_y,
                            exog=train_x,
                            order=order,
                            seasonal_order=seasonal_order,
                            trend=trend,
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                        ).fit(disp=False)

                        fc = model.get_forecast(
                            steps=len(test),
                            exog=test_x,
                        ).predicted_mean.values

                        actual = test_y.values
                        m = actual != 0
                        if not m.any():
                            pbar.update(0)
                            continue

                        mape = mean_absolute_percentage_error(actual[m], fc[m])

                        row = {
                            "mape":           mape,
                            "aic":            model.aic,
                            "bic":            model.bic,
                            "order":          order,
                            "seasonal_order": seasonal_order,
                            "trend":          trend,
                            "lags":           dict(zip(features, lag_combo)),
                            "exog_cols":      exog_cols,
                            "n_train":        len(train),
                            "n_test":         len(test),
                        }
                        results.append(row)

                        if mape < best["mape"]:
                            best = {
                                **row,
                                "model":      model,
                                "forecast":   fc,
                                "actual":     actual,
                                "test_index": test.index,
                            }

                    except Exception as e:
                        results.append({
                            "mape": np.nan,
                            "order": order,
                            "seasonal_order": seasonal_order,
                            "trend": trend,
                            "lags": dict(zip(features, lag_combo)),
                            "error": str(e)[:120],
                        })

    pbar.close()

    out = (pd.DataFrame(results)
             .sort_values("mape", na_position="last")
             .reset_index(drop=True))

    print(f"\n✓ Best MAPE : {best['mape']:.4f}")
    print(f"  Order     : {best.get('order')}")
    print(f"  Seasonal  : {best.get('seasonal_order')}")
    print(f"  Trend     : {best.get('trend')}")
    print(f"  Lags      : {best.get('lags')}")

    return out.head(top_k), best
