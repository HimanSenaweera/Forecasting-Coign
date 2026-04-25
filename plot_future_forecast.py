import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

# ensure Date is datetime
future["Date"] = pd.to_datetime(future["Date"])

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    future["Date"],
    future["Future Forcast"],
    color="red",
    linewidth=2,
    label="Statistical Model prediction",
)

# x-axis formatting — month-year like your chart
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
ax.xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=0)

# formatting
ax.set_title("GROSS CREDIT LOSSES($)", fontsize=13, fontweight="bold")
ax.set_xlabel("DATE", fontsize=10)
ax.set_ylabel("", fontsize=10)

# y-axis comma formatting
ax.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"{x:,.0f}")
)

ax.legend(loc="lower right", frameon=True)
ax.grid(False)

plt.tight_layout()
plt.show()
