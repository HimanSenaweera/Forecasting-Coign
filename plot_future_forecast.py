import matplotlib.pyplot as plt
import pandas as pd

# --- Actual: filter to only the months that overlap with forecast ---
future["Date"] = pd.to_datetime(future["Date"])
forecast_start = future["Date"].min()

actual_df = history_df[[TARGET_METRIC]].reset_index()
actual_df.columns = ["Date", "METRIC_VALUE"]
actual_df["Date"] = pd.to_datetime(actual_df["Date"])

# Keep only actuals from forecast start onwards (the overlap window)
actual_df = actual_df[actual_df["Date"] >= forecast_start].copy()
actual_df["FORECAST_TYPE"] = "Actual"
actual_df["month_year"] = actual_df["Date"].dt.strftime("%b-%y")

# --- Forecast df ---
forecast_df = future[["Date", "Future Forcast"]].copy()
forecast_df.columns = ["Date", "METRIC_VALUE"]
forecast_df["FORECAST_TYPE"] = "Statistical Model prediction"
forecast_df["month_year"] = forecast_df["Date"].dt.strftime("%b-%y")

# --- Combine ---
pred_df = pd.concat([actual_df, forecast_df], ignore_index=True)
pred_df = pred_df.sort_values("Date")

# --- Build a shared date-indexed x-axis from forecast dates ---
all_dates = forecast_df.sort_values("Date")["Date"].tolist()
date_to_x = {d: i for i, d in enumerate(all_dates)}
x_labels = [d.strftime("%b-%y") for d in all_dates]

palette = {
    "Actual": "#A2CEED",
    "Statistical Model prediction": "red",
}
order = ["Actual", "Statistical Model prediction"]
existing = [k for k in order if k in pred_df["FORECAST_TYPE"].unique()]

fig, ax = plt.subplots(figsize=(14, 6))
ax.set_title("GROSS CREDIT LOSSES($)", fontsize=14, fontweight="bold")

for label in existing:
    subset = pred_df[pred_df["FORECAST_TYPE"] == label].sort_values("Date")
    # Map each date to the shared x position; drop dates outside forecast range
    subset = subset[subset["Date"].isin(date_to_x.keys())]
    x_pos = subset["Date"].map(date_to_x).values
    ax.plot(
        x_pos,
        subset["METRIC_VALUE"].values,
        label=label,
        color=palette.get(label, None),
        linewidth=2,
    )

ax.set_xticks(range(len(x_labels)))
ax.set_xticklabels(x_labels, rotation=0)
ax.set_xlabel("DATE")

ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
ax.grid(False)
plt.tight_layout()
plt.show()
