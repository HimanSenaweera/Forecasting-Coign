import warnings
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# Metric Helpers
# ─────────────────────────────────────────────

def smape(actual, predicted):
    denom = (np.abs(actual) + np.abs(predicted)) / 2
    return np.mean(np.abs(actual - predicted) / denom) * 100

def mape_pct(actual, predicted):
    mask = actual != 0
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100

def mae(actual, predicted):
    return np.mean(np.abs(actual - predicted))

def rmse(actual, predicted):
    return np.sqrt(np.mean((actual - predicted) ** 2))


# ─────────────────────────────────────────────
# Fold Generator  (identical to your grid search)
# ─────────────────────────────────────────────

def generate_2fold_6pt(all_dates, test_size=6):
    n           = len(all_dates)
    total_test  = test_size * 2
    total_train = n - total_test

    if total_train < 1:
        raise ValueError(f"Not enough data: need ≥{total_test+1} pts, got {n}.")

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
# Result Printer
# ─────────────────────────────────────────────

def _print_results(model_name, fold_mapes, fold_smapes, fold_maes, fold_rmses, extra_info=None):
    print(f"\n{'─'*60}")
    print(f"  {model_name}")
    print(f"{'─'*60}")
    print(f"  Avg  MAPE   : {np.mean(fold_mapes):.4f}%")
    print(f"  MAPE std    : {np.std(fold_mapes):.4f}%  ← lower = more stable")
    print(f"  Fold 1 MAPE : {fold_mapes[0]:.4f}%")
    print(f"  Fold 2 MAPE : {fold_mapes[1] if len(fold_mapes) > 1 else float('nan'):.4f}%")
    print(f"  Avg  SMAPE  : {np.mean(fold_smapes):.4f}%")
    print(f"  Avg  MAE    : {np.mean(fold_maes):.4f}")
    print(f"  Avg  RMSE   : {np.mean(fold_rmses):.4f}")
    if extra_info:
        for k, v in extra_info.items():
            print(f"  {k:<13}: {v}")
    print(f"{'─'*60}\n")


# ═══════════════════════════════════════════════════════════════
# MODEL 1  —  Naïve
# ═══════════════════════════════════════════════════════════════

def run_naive(base_df, target_col, test_size=6):
    print(f"\n{'═'*62}")
    print(f"  MODEL 1 → Naïve Baseline")
    print(f"{'═'*62}")

    work       = base_df[[target_col]].copy()
    work.index = pd.DatetimeIndex(work.index).date
    folds      = generate_2fold_6pt(list(work.index), test_size)

    fold_mapes, fold_smapes, fold_maes, fold_rmses = [], [], [], []

    for fold in folds:
        train = work[work.index <= fold["train_end"]][target_col].astype(float)
        test  = work[
            (work.index >= fold["test_start"]) &
            (work.index <= fold["test_end"])
        ][target_col].astype(float)

        history = list(train)
        preds   = []
        for true_val in test:
            preds.append(history[-1])
            history.append(true_val)

        actual      = test.values
        predictions = np.array(preds)

        fold_mapes.append(mape_pct(actual, predictions))
        fold_smapes.append(smape(actual, predictions))
        fold_maes.append(mae(actual, predictions))
        fold_rmses.append(rmse(actual, predictions))

    _print_results("Naïve (last-value)", fold_mapes, fold_smapes, fold_maes, fold_rmses)


# ═══════════════════════════════════════════════════════════════
# MODEL 2  —  Pure ARIMA  (grid search over p, d, q + trend)
# ═══════════════════════════════════════════════════════════════

def run_pure_arima(
    base_df,
    target_col,
    p_range=(0, 1, 2, 3),
    d_range=(0, 1),
    q_range=(0, 1, 2),
    trend_options=("n", "c"),
    test_size=6,
    top_k=10,
):
    print(f"\n{'═'*62}")
    print(f"  MODEL 2 → Pure ARIMA Grid Search  (no covariates)")
    print(f"{'═'*62}")

    work       = base_df[[target_col]].copy()
    work.index = pd.DatetimeIndex(work.index).date
    folds      = generate_2fold_6pt(list(work.index), test_size)

    order_combos = list(itertools.product(p_range, d_range, q_range))
    total        = len(order_combos) * len(trend_options)

    print(f"  p range     : {p_range}")
    print(f"  d range     : {d_range}")
    print(f"  q range     : {q_range}")
    print(f"  Trend opts  : {trend_options}")
    print(f"  Total combos: {total}  (each × {len(folds)} folds)\n")

    results = []
    best    = {"avg_mape": np.inf}
    pbar    = tqdm(total=total, desc="ARIMA grid search")

    for order in order_combos:
        for trend in trend_options:
            pbar.update(1)

            fold_mapes, fold_smapes, fold_maes, fold_rmses = [], [], [], []

            for fold in folds:
                try:
                    train_y = work[work.index <= fold["train_end"]][target_col].astype(float)
                    test_df = work[
                        (work.index >= fold["test_start"]) &
                        (work.index <= fold["test_end"])
                    ]

                    if len(train_y) < 6 or len(test_df) == 0:
                        continue

                    history = train_y.copy()
                    preds   = []

                    for i in range(len(test_df)):
                        m = SARIMAX(
                            history,
                            order=order,
                            seasonal_order=(0, 0, 0, 0),
                            trend=trend,
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                        ).fit(disp=False)

                        pred = m.get_forecast(steps=1).predicted_mean.iloc[0]
                        preds.append(pred)

                        true_val = test_df[target_col].astype(float).iloc[i]
                        history  = pd.concat([
                            history,
                            pd.Series([true_val], index=[test_df.index[i]])
                        ])

                    actual      = test_df[target_col].astype(float).values
                    predictions = np.array(preds)

                    if not (actual != 0).any():
                        continue

                    fold_mapes.append(mape_pct(actual, predictions))
                    fold_smapes.append(smape(actual, predictions))
                    fold_maes.append(mae(actual, predictions))
                    fold_rmses.append(rmse(actual, predictions))

                except Exception:
                    continue

            if not fold_mapes:
                continue

            row = {
                "avg_mape":   np.mean(fold_mapes),
                "mape_std":   np.std(fold_mapes),
                "fold1_mape": fold_mapes[0] if len(fold_mapes) > 0 else np.nan,
                "fold2_mape": fold_mapes[1] if len(fold_mapes) > 1 else np.nan,
                "avg_smape":  np.mean(fold_smapes),
                "avg_mae":    np.mean(fold_maes),
                "avg_rmse":   np.mean(fold_rmses),
                "order":      order,
                "trend":      trend,
            }
            results.append(row)

            if row["avg_mape"] < best["avg_mape"]:
                best = {**row}

    pbar.close()

    out = pd.DataFrame(results).sort_values("avg_mape", na_position="last").reset_index(drop=True)

    print(f"\n  Top {top_k} ARIMA orders:")
    print(f"  {'Order':<14} {'Trend':<7} {'Avg MAPE':>10} {'MAPE Std':>10} {'F1 MAPE':>10} {'F2 MAPE':>10}")
    print(f"  {'─'*14} {'─'*7} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")
    for _, r in out.head(top_k).iterrows():
        print(
            f"  {str(r['order']):<14} {str(r['trend']):<7} "
            f"{r['avg_mape']:>9.2f}% {r['mape_std']:>9.2f}% "
            f"{r['fold1_mape']:>9.2f}% {r['fold2_mape']:>9.2f}%"
        )

    _print_results(
        f"Best Pure ARIMA{best['order']} trend={best['trend']}",
        [best["fold1_mape"], best["fold2_mape"]],
        [best["avg_smape"]] * 2,
        [best["avg_mae"]]   * 2,
        [best["avg_rmse"]]  * 2,
        extra_info={"Best order": best["order"], "Trend": best["trend"]},
    )

    return out.head(top_k)


# ═══════════════════════════════════════════════════════════════
# MODEL 3  —  ETS  (grid search over error / trend / damped)
# ═══════════════════════════════════════════════════════════════

def run_ets_auto(base_df, target_col, test_size=6):
    print(f"\n{'═'*62}")
    print(f"  MODEL 3 → ETS Grid Search  (error × trend × damped)")
    print(f"{'═'*62}")

    work       = base_df[[target_col]].copy()
    work.index = pd.DatetimeIndex(work.index).date
    folds      = generate_2fold_6pt(list(work.index), test_size)

    ets_specs = [
        {"error": "add", "trend": None,  "damped_trend": False},  # Simple ES
        {"error": "add", "trend": "add", "damped_trend": False},  # Holt's linear
        {"error": "add", "trend": "add", "damped_trend": True},   # Damped Holt's
        {"error": "mul", "trend": None,  "damped_trend": False},  # Multiplicative ES
        {"error": "mul", "trend": "add", "damped_trend": False},  # Mult + trend
        {"error": "mul", "trend": "add", "damped_trend": True},   # Mult + damped
    ]

    print(f"  Specs tried : {len(ets_specs)}  (each × {len(folds)} folds)\n")

    results = []
    best    = {"avg_mape": np.inf}
    pbar    = tqdm(total=len(ets_specs), desc="ETS grid search")

    for spec in ets_specs:
        pbar.update(1)

        spec_label = (
            f"error={spec['error']}, "
            f"trend={spec['trend'] or 'None'}"
            f"{', damped' if spec['damped_trend'] else ''}"
        )

        fold_mapes, fold_smapes, fold_maes, fold_rmses = [], [], [], []

        for fold in folds:
            try:
                train_y = work[work.index <= fold["train_end"]][target_col].astype(float)
                test_df = work[
                    (work.index >= fold["test_start"]) &
                    (work.index <= fold["test_end"])
                ]

                if len(train_y) < 6 or len(test_df) == 0:
                    continue

                history = train_y.copy()
                preds   = []

                for i in range(len(test_df)):
                    m = ETSModel(
                        history,
                        error=spec["error"],
                        trend=spec["trend"],
                        damped_trend=spec["damped_trend"],
                        seasonal=None,
                    ).fit(disp=False)

                    pred = m.forecast(1).iloc[0]
                    preds.append(pred)

                    true_val = test_df[target_col].astype(float).iloc[i]
                    history  = pd.concat([
                        history,
                        pd.Series([true_val], index=[test_df.index[i]])
                    ])

                actual      = test_df[target_col].astype(float).values
                predictions = np.array(preds)

                if not (actual != 0).any():
                    continue

                fold_mapes.append(mape_pct(actual, predictions))
                fold_smapes.append(smape(actual, predictions))
                fold_maes.append(mae(actual, predictions))
                fold_rmses.append(rmse(actual, predictions))

            except Exception:
                continue

        if not fold_mapes:
            continue

        row = {
            "avg_mape":   np.mean(fold_mapes),
            "mape_std":   np.std(fold_mapes),
            "fold1_mape": fold_mapes[0] if len(fold_mapes) > 0 else np.nan,
            "fold2_mape": fold_mapes[1] if len(fold_mapes) > 1 else np.nan,
            "avg_smape":  np.mean(fold_smapes),
            "avg_mae":    np.mean(fold_maes),
            "avg_rmse":   np.mean(fold_rmses),
            "spec_label": spec_label,
        }
        results.append(row)

        if row["avg_mape"] < best["avg_mape"]:
            best = {**row}

    pbar.close()

    out = pd.DataFrame(results).sort_values("avg_mape", na_position="last").reset_index(drop=True)

    print(f"\n  All ETS specs ranked:")
    print(f"  {'Spec':<40} {'Avg MAPE':>10} {'MAPE Std':>10} {'F1 MAPE':>10} {'F2 MAPE':>10}")
    print(f"  {'─'*40} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")
    for _, r in out.iterrows():
        print(
            f"  {r['spec_label']:<40} "
            f"{r['avg_mape']:>9.2f}% {r['mape_std']:>9.2f}% "
            f"{r['fold1_mape']:>9.2f}% {r['fold2_mape']:>9.2f}%"
        )

    _print_results(
        f"Best ETS  {best['spec_label']}",
        [best["fold1_mape"], best["fold2_mape"]],
        [best["avg_smape"]] * 2,
        [best["avg_mae"]]   * 2,
        [best["avg_rmse"]]  * 2,
        extra_info={"Best spec": best["spec_label"]},
    )

    return out


# ─────────────────────────────────────────────
# EXAMPLE CALL
# ─────────────────────────────────────────────
"""
run_naive(df_merged, TARGET_METRIC)

run_pure_arima(
    base_df       = df_merged,
    target_col    = TARGET_METRIC,
    p_range       = (0, 1, 2, 3),
    d_range       = (0, 1),
    q_range       = (0, 1, 2),
    trend_options = ("n", "c"),
)

run_ets_auto(
    base_df    = df_merged,
    target_col = TARGET_METRIC,
)
"""
