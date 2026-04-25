import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

# --- Build a unified pred_df like your existing plot_results function ---

# Actual historical values (from history_df before forecasting)
actual_df = history_df[[TARGET_METRIC]].reset_index()
actual_df.columns = ["Date", "METRIC_VALUE"]
actual_df["FORECAST_TYPE"] = "Actual"

# Future forecast values
forecast_df = future[["Date", "Future Forcast"]].copy()
forecast_df.columns = ["Date", "METRIC_VALUE"]
forecast_df["FORECAST_TYPE"] = "Statistical Model prediction"

# Combine into one pred_df
pred_df = pd.concat([actual_df, forecast_df], ignore_index=True)
pred_df["Date"] = pd.to_datetime(pred_df["Date"])
pred_df["month_year"] = pred_df["Date"].dt.strftime("%b-%y")

# --- Plot using your existing plot_results style ---

palette = {
    "Actual": "#A2CEED",          # blue (matches your existing palette)
    "Statistical Model prediction": "red",
}

order = ["Actual", "Statistical Model prediction"]
existing = [k for k in order if k in pred_df["FORECAST_TYPE"].unique()]

fig, ax = plt.subplots(figsize=(20, 6))
ax.set_title("GROSS CREDIT LOSSES($)", fontsize=14, fontweight="bold")

for label in existing:
    subset = pred_df[pred_df["FORECAST_TYPE"] == label].sort_values("Date")
    ax.plot(
        range(len(subset)),
        subset["METRIC_VALUE"].values,
        label=label,
        color=palette.get(label, None),
        linewidth=2,
    )

# x-axis tick labels from actual (first series)
x_labels = (
    pred_df[pred_df["FORECAST_TYPE"] == existing[0]]
    .sort_values("Date")["month_year"]
    .astype(str)
    .tolist()
)
ax.set_xticks(range(len(x_labels)))
ax.set_xticklabels(x_labels, rotation=0)
ax.set_xlabel("DATE")

# y-axis comma formatting (override % formatter from your original func)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))

ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3)
ax.grid(False)
plt.tight_layout()
plt.show()
