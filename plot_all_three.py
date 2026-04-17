import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(test_df.index, actuals,
        color='blue', linewidth=2.5,
        label='Actual', marker='o')

ax.plot(test_df.index, predictions_A,
        color='green', linewidth=1.5,
        linestyle='--', label='Model A (2,1,1) 26.66%',
        marker='s')

ax.plot(test_df.index, predictions_B,
        color='red', linewidth=1.5,
        linestyle='--', label='Model B (2,1,2) 20.44%',
        marker='^')

ax.plot(test_df.index, predictions_ensemble,
        color='purple', linewidth=2,
        linestyle='-', label='Ensemble',
        marker='D')

ax.grid(True, alpha=0.3)
ax.legend()
ax.set_title('Model A vs Model B vs Ensemble')
plt.tight_layout()
plt.show()
