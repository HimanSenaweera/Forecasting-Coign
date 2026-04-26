import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

fig, ax = plt.subplots(figsize=(12, 5))

# ── 1. Actual values from test_df (if available) ──────────────────────────────
has_actuals = (
    test_df is not None
    and len(test_df) > 0
    and TARGET_METRIC in test_df.columns
)

if has_actuals:
    actual_values = test_df[TARGET_METRIC].abs()
    ax.plot(
        actual_values.index,
        actual_values.values,
        color='#ADD8E6',       # light blue — matches your chart
        linewidth=1.5,
        label='Actual',
        alpha=0.9,
        zorder=2
    )

# ── 2. Future forecast from `future` dataframe ────────────────────────────────
future_plot = future.copy()
future_plot['Date'] = pd.to_datetime(future_plot['Date'])
future_plot = future_plot.set_index('Date')

ax.plot(
    future_plot.index,
    future_plot['Future Forcast'].values,
    color='red',               # red — matches your chart
    linewidth=1.5,
    label='Statistical Model prediction',
    zorder=3
)

# ── 3. Optional: vertical line separating actuals vs future ───────────────────
if has_actuals:
    last_actual_date = actual_values.index[-1]
    first_future_date = future_plot.index[0]
    
    # Only draw separator if there's a gap (future starts after actuals end)
    if first_future_date > last_actual_date:
        ax.axvline(
            x=last_actual_date,
            color='gray',
            linestyle='--',
            linewidth=0.8,
            alpha=0.5,
            label='Forecast start'
        )

# ── 4. Formatting ─────────────────────────────────────────────────────────────
ax.set_title('GROSS CREDIT LOSSES($)', fontsize=10, fontweight='bold', pad=12)
ax.set_xlabel('DATE', fontsize=9)
ax.set_ylabel('')

# Y-axis: comma-formatted like your chart
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.2f}'))

# X-axis: Mon-YY format like your chart
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))
ax.xaxis.set_major_locator(mdates.MonthLocator())
fig.autofmt_xdate(rotation=0, ha='center')

# Clean spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', labelsize=8)

# Legend at bottom center — matches your chart layout
ax.legend(
    loc='lower center',
    bbox_to_anchor=(0.5, -0.22),
    ncol=3,
    frameon=False,
    fontsize=8
)

plt.tight_layout()
plt.show()
