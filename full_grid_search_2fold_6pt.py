import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from statsmodels.tsa.statespace.sarimax import SARIMAX


# ─────────────────────────────────────────────
# Metric Helpers
# ─────────────────────────────────────────────

def smape(actual, predicted):
    denom = (np.abs(actual) + np.abs(predicted)) / 2
    return np.mean(np.abs(actual - predicted) / denom) * 100

def mape_pct(actual, predicted):
    return np.mean(np.abs((actual - predicted) / actual)) * 100

def mae(actual, predicted):
    return np.mean(np.abs(actual - predicted))

def rmse(actual, predicted):
    return np.sqrt(np.mean((actual - predicted) ** 2))


# ─────────────────────────────────────────────
# NaN Fill Methods
# ─────────────────────────────────────────────

def apply_fill(series, method):
    """
    Apply a named fill strategy to a pandas Series.

    Methods
    -------
    mean        : fill with column mean  (original behaviour)
    median      : fill with column median  (robust to outliers)
    ffill       : forward fill  (carry last known value forward)
    bfill       : backward fill  (pull next known value back)
    interpolate : linear interpolation between known values
    ffill+bfill : forward fill first, then backward fill any
                  remaining NaNs at the start of the series
    """
    if method == "mean":
        return series.fillna(series.mean())
    elif method == "median":
        return series.fillna(series.median())
    elif method == "ffill":
        return series.ffill().fillna(series.mean())   # fallback for leading NaNs
    elif method == "bfill":
        return series.bfill().fillna(series.mean())   # fallback for trailing NaNs
    elif method == "interpolate":
        return series.interpolate(method="linear").fillna(series.mean())
    elif method == "ffill+bfill":
        return series.ffill().bfill().fillna(series.mean())
    else:
        raise ValueError(f"Unknown fill method: {method}")


# ─────────────────────────────────────────────
# Fold Generator
# ─────────────────────────────────────────────

def generate_2fold_6pt(all_dates, test_size=6):
    """
    Generates 2 non-overlapping walk-forward folds,
    each with a 6-point test window.

    With 26 points and test_size=6:
        Fold 1: Train [1–14]  → Test [15–20]
        Fold 2: Train [1–20]  → Test [21–26]
    """
    n           = len(all_dates)
    total_test  = test_size * 2
    total_train = n - total_test

    if total_train < 1:
        raise ValueError(
            f"Not enough data: need at least {total_test + 1} points, got {n}."
        )

    folds = []
    for i in range(2):
        train_end_idx  = total_train + i * test_size - 1
        test_start_idx = train_end_idx + 1
        test_end_idx   = test_start_idx + test_size - 1

        if test_end_idx >= n:
            print(f"  ⚠ Fold {i+1} skipped — not enough data")
            break

        folds.append({
            "fold":       i + 1,
            "train_end":  all_dates[train_end_idx],
            "test_start": all_dates[test_start_idx],
            "test_end":   all_dates[test_end_idx],
            "n_train":    train_end_idx + 1,
            "n_test":     test_size,
        })

    return folds


# ─────────────────────────────────────────────
# Rolling One-Step Forecast
# ─────────────────────────────────────────────

def rolling_forecast(train_df, test_df, target_col, target_lag_col,
                     exog_cols, feat_exog_cols,
                     order, seasonal_order, trend):
    """
    Refits SARIMAX one step at a time through test_df,
    growing history with each prediction.
    """
    history_y   = train_df[target_col].astype(float).copy()
    history_df  = train_df.copy()
    predictions = []

    for i in range(len(test_df)):
        model = SARIMAX(
            history_y,
            exog=history_df[exog_cols].astype(float),
            order=order,
            seasonal_order=seasonal_order,
            trend=trend,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)

        new_row = test_df.iloc[[i]].copy()

        for col in feat_exog_cols:
            new_row[col] = history_df[col].iloc[-1]

        new_row[target_lag_col] = history_y.iloc[-1]

        pred = model.get_forecast(
            steps=1,
            exog=new_row[exog_cols].astype(float),
        ).predicted_mean.iloc[0]

        predictions.append(pred)

        history_y = pd.concat([
            history_y,
            pd.Series([pred], index=[test_df.index[i]])
        ])
        new_row[target_col] = pred
        history_df = pd.concat([history_df, new_row])

    actual = test_df[target_col].astype(float).values
    return actual, np.array(predictions)


# ─────────────────────────────────────────────
# Main Grid Search Function
# ─────────────────────────────────────────────

def full_grid_search_2fold_6pt(
    base_df,
    features,
    target_col,
    lags=(0, 1, 2, 3),
    target_lags=(0, 1, 2, 3),        # ← NOW a tuple, searches all values
    fill_methods=("mean", "median", "ffill", "bfill", "interpolate", "ffill+bfill"),
    p_range=(0, 1, 2, 3),
    d_range=(0,),
    q_range=(0, 1),
    P_range=(0,),
    D_range=(0,),
    Q_range=(0,),
    m_range=(0,),
    trend_options=("n",),
    rank_by="avg_mape",
    top_k=20,
    test_size=6,
):
    """
    2-Fold Walk-Forward CV Grid Search — extended version.

    Changes vs original
    ───────────────────
    1. target_lags  : now a TUPLE — searches all target lag values
                      e.g. (0, 1, 2, 3) instead of a fixed target_lag=1
                      Best lag is reported in results as 'target_lag'

    2. fill_methods : tries multiple NaN imputation strategies per combo
                      and keeps the one that gives the lowest avg MAPE
                      Methods: mean, median, ffill, bfill,
                               interpolate, ffill+bfill

    Everything else (fold structure, rolling forecast, ranking) is
    identical to the original.
    """

    # ── Build all feature lag columns once ────────────────────────────
    base_work = base_df.copy()
    base_work.index = pd.DatetimeIndex(base_work.index).date

    for feat in features:
        for L in lags:
            base_work[f"{feat}__lag{L}L"] = (
                base_work[feat].shift(L) if L > 0 else base_work[feat]
            )

    # Build target lag columns for all target_lags upfront
    for tl in target_lags:
        col = f"{target_col}__lag{tl}L"
        base_work[col] = base_work[target_col].shift(tl) if tl > 0 else base_work[target_col]

    # ── Generate 2 folds ──────────────────────────────────────────────
    all_dates = list(base_work.index)
    folds     = generate_2fold_6pt(all_dates, test_size=test_size)

    if not folds:
        raise ValueError("No valid folds generated. Check dataset size vs test_size.")

    print(f"\n{'─'*65}")
    print(f"  2-Fold Walk-Forward CV  |  {test_size}-point test windows")
    print(f"{'─'*65}")
    for f in folds:
        print(f"  Fold {f['fold']}: "
              f"Train [start → {f['train_end']}] ({f['n_train']} pts) | "
              f"Test  [{f['test_start']} → {f['test_end']}] ({f['n_test']} pts)")
    print(f"{'─'*65}\n")

    # ── Search space ──────────────────────────────────────────────────
    lag_combos      = list(itertools.product(lags, repeat=len(features)))
    order_combos    = list(itertools.product(p_range, d_range, q_range))
    seasonal_combos = list(itertools.product(P_range, D_range, Q_range, m_range))

    total = (
        len(lag_combos)
        * len(target_lags)
        * len(fill_methods)
        * len(order_combos)
        * len(seasonal_combos)
        * len(trend_options)
    )

    print(f"  Lag combos      : {len(lag_combos)}")
    print(f"  Target lags     : {list(target_lags)}")
    print(f"  Fill methods    : {list(fill_methods)}")
    print(f"  Order combos    : {len(order_combos)}")
    print(f"  Seasonal combos : {len(seasonal_combos)}")
    print(f"  Trend options   : {len(trend_options)}")
    print(f"  Total combos    : {total}  (each × {len(folds)} folds)\n")

    results = []
    best    = {rank_by: np.inf}
    pbar    = tqdm(total=total, desc="2-fold grid search")

    for lag_combo in lag_combos:
        feat_exog_cols = [f"{feat}__lag{L}L" for feat, L in zip(features, lag_combo)]

        for target_lag in target_lags:
            target_lag_col = f"{target_col}__lag{target_lag}L"
            exog_cols      = feat_exog_cols + [target_lag_col]
            all_cols       = exog_cols + [target_col]

            for fill_method in fill_methods:

                # ── Apply chosen fill method ───────────────────────────
                work = base_work.copy()
                for col in all_cols:
                    work[col] = apply_fill(work[col], fill_method)

                clean = work.copy()

                for order in order_combos:
                    for seasonal_order in seasonal_combos:
                        for trend in trend_options:
                            pbar.update(1)

                            fold_mapes   = []
                            fold_smapes  = []
                            fold_maes    = []
                            fold_rmses   = []

                            for fold in folds:
                                try:
                                    train = clean[clean.index <= fold["train_end"]]
                                    test  = clean[
                                        (clean.index >= fold["test_start"]) &
                                        (clean.index <= fold["test_end"])
                                    ]

                                    if len(train) < 10 or len(test) == 0:
                                        continue

                                    actual, predictions = rolling_forecast(
                                        train, test,
                                        target_col, target_lag_col,
                                        exog_cols, feat_exog_cols,
                                        order, seasonal_order, trend,
                                    )

                                    m = actual != 0
                                    if not m.any():
                                        continue

                                    fold_mapes.append(mape_pct(actual[m], predictions[m]))
                                    fold_smapes.append(smape(actual[m], predictions[m]))
                                    fold_maes.append(mae(actual, predictions))
                                    fold_rmses.append(rmse(actual, predictions))

                                except Exception:
                                    continue

                            if len(fold_mapes) == 0:
                                continue

                            fold1_mape = fold_mapes[0] if len(fold_mapes) > 0 else np.nan
                            fold2_mape = fold_mapes[1] if len(fold_mapes) > 1 else np.nan

                            row = {
                                "avg_mape":       np.mean(fold_mapes),
                                "avg_smape":      np.mean(fold_smapes),
                                "avg_mae":        np.mean(fold_maes),
                                "avg_rmse":       np.mean(fold_rmses),
                                "mape_std":       np.std(fold_mapes),
                                "fold1_mape":     fold1_mape,
                                "fold2_mape":     fold2_mape,
                                "order":          order,
                                "seasonal_order": seasonal_order,
                                "trend":          trend,
                                "lags":           dict(zip(features, lag_combo)),
                                "target_lag":     target_lag,     # ← new
                                "fill_method":    fill_method,    # ← new
                                "exog_cols":      exog_cols,
                                "n_folds_used":   len(fold_mapes),
                            }
                            results.append(row)

                            if row[rank_by] < best.get(rank_by, np.inf):
                                best = {**row}

    pbar.close()

    out = pd.DataFrame(results)
    if out.empty or out[rank_by].isna().all():
        print("⚠ All combinations failed.")
        return out, best

    out = out.sort_values(rank_by, na_position="last").reset_index(drop=True)

    print(f"\n{'─'*65}")
    print(f"  Best combination  (ranked by {rank_by})")
    print(f"{'─'*65}")
    print(f"  Avg  MAPE        : {best.get('avg_mape',  np.nan):.4f}%")
    print(f"  MAPE std         : {best.get('mape_std',  np.nan):.4f}%  ← lower = more stable")
    print(f"  Fold 1 MAPE      : {best.get('fold1_mape',np.nan):.4f}%")
    print(f"  Fold 2 MAPE      : {best.get('fold2_mape',np.nan):.4f}%")
    print(f"  Avg  SMAPE       : {best.get('avg_smape', np.nan):.4f}%")
    print(f"  Avg  MAE         : {best.get('avg_mae',   np.nan):.4f}")
    print(f"  Avg  RMSE        : {best.get('avg_rmse',  np.nan):.4f}")
    print(f"  Order            : {best.get('order')}")
    print(f"  Seasonal         : {best.get('seasonal_order')}")
    print(f"  Trend            : {best.get('trend')}")
    print(f"  Lags             : {best.get('lags')}")
    print(f"  Target lag       : {best.get('target_lag')}")    # ← new
    print(f"  Fill method      : {best.get('fill_method')}")   # ← new
    print(f"  Folds used       : {best.get('n_folds_used')}")
    print(f"{'─'*65}\n")

    return out.head(top_k), best


# ─────────────────────────────────────────────
# Example Call
# ─────────────────────────────────────────────

"""
features = [
    "90+ Total DQ Balance",
    "Expected Loss",
]

top, best = full_grid_search_2fold_6pt(
    base_df      = df_merged,
    features     = features,
    target_col   = TARGET_METRIC,
    lags         = (0, 1, 2, 3),
    target_lags  = (0, 1, 2, 3),          # ← searches all target lags
    fill_methods = ("mean", "median", "ffill", "bfill", "interpolate", "ffill+bfill"),
    p_range      = (0, 1, 2, 3),
    d_range      = (0, 1, 2),
    q_range      = (0, 1, 2, 3),
    P_range      = (0,),
    D_range      = (0,),
    Q_range      = (0,),
    m_range      = (0,),
    trend_options= ("n", "c"),
    rank_by      = "avg_mape",
    top_k        = 20,
    test_size    = 6,
)

# View top results including new columns
top[["avg_mape", "mape_std", "fold1_mape", "fold2_mape",
     "order", "trend", "lags", "target_lag", "fill_method"]]
"""
