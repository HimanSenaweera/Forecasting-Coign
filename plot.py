import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

features_to_plot = [
    'Gross Credit Losses',
    '90+ DQ Rate',
    'Finance Charges Rate',
    'Payment_Rate_lag1',
    '6 to Charge Off Roll Rate',
    'Expected Loss',
    'Expected Loss Rate',
    'DQ Bucket 1 Rate',
    'DQ Bucket 2 Rate',
]

# Filter to the window around the spike
plot_df = data_df.loc['2025-06-30':'2025-11-30', features_to_plot]

fig, axes = plt.subplots(
    nrows=3, ncols=3, 
    figsize=(16, 10),
    sharex=True
)
axes = axes.flatten()

for idx, col in enumerate(features_to_plot):
    ax = axes[idx]
    ax.plot(plot_df.index, plot_df[col], 
            marker='o', linewidth=2, 
            color='steelblue', markersize=5)
    
    # Highlight Sep-25 (month before spike) in orange
    if '2025-09-30' in plot_df.index:
        ax.axvline('2025-09-30', color='orange', 
                   linestyle='--', linewidth=1.5, 
                   label='Sep-25 (pre-spike)')
    
    # Highlight Oct-25 spike in red
    if '2025-10-31' in plot_df.index:
        ax.axvline('2025-10-31', color='red', 
                   linestyle='--', linewidth=1.5, 
                   label='Oct-25 (spike)')
    
    ax.set_title(col, fontsize=10, fontweight='bold')
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.grid(True, alpha=0.3)

# Add legend to first subplot only
axes[0].legend(fontsize=8)

fig.suptitle('Feature Values Around Oct-25 Spike\n(Looking for leading signals in Sep-25)', 
             fontsize=13, fontweight='bold', y=1.02)

plt.tight_layout()
plt.show()
