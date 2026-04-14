from itertools import combinations
import numpy as np

# ─────────────────────────────────────────────
# STEP 1: Keep a clean base dataframe with NO lag features
# (just the raw columns before any shifting)
# ─────────────────────────────────────────────
base_df = data_df.copy()  # your original merged df before lag creation

# Define which features need lags and how much
LAG_FEATURE_CONFIG = {
    'Gross_Credit_Losses_lag1': ('Gross Credit Losses', 1),  # (source_col, lag)
    'Payment_Rate_lag1':        ('Payment Rate', 1),
    'Some_Feature_lag2':        ('Some Feature', 2),
    'Some_Feature_lag3':        ('Some Feature', 3),
}

# Non-lag features that are always ready to use
NON_LAG_FEATURES = [
    '90+ DQ Rate',
    'Finance Charges Rate',
    'Expected Loss Rate',
    'Expected Loss Roll Ave 3M',
    '60+ DQ Rate',
]

# ─────────────────────────────────────────────
# STEP 2: Function to build train/test 
#         for a SPECIFIC feature combo
# ─────────────────────────────────────────────
def build_dataset_for_combo(base_df, feature_combo):
    """
    Creates lag features ONLY for the features in this combo,
    then dropna() — so row loss is proportional to 
    the lags actually used.
    """
    df = base_df.copy()
    
    # Only create lag columns that are needed for THIS combo
    for feat in feature_combo:
        if feat in LAG_FEATURE_CONFIG:
            source_col, lag_n = LAG_FEATURE_CONFIG[feat]
            df[feat] = df[source_col].shift(lag_n)
    
    # dropna only removes rows based on lags IN THIS COMBO
    df = df.dropna(subset=feature_combo)
    
    # Rebuild train/test split on this combo's cleaned df
    train = df[(df.index >= first_input_date) & 
               (df.index <= last_input_date)]
    test  = df[(df.index >= first_test_date)  & 
               (df.index <= last_test_date)]
    
    return train, test


# ─────────────────────────────────────────────
# STEP 3: Search loop — fair comparison now
# ─────────────────────────────────────────────
anchor   = ['Gross_Credit_Losses_lag1']
optional = [
    'Payment_Rate_lag1',
    '90+ DQ Rate',
    'Finance Charges Rate',
    'Expected Loss Rate',
    'Expected Loss Roll Ave 3M',
    '60+ DQ Rate',
    # 'Some_Feature_lag3',  # if added, only THIS combo loses 3 rows
]

results = []

for r in range(1, 4):
    for combo in combinations(optional, r):
        
        exog_test = anchor + list(combo)
        
        # ← KEY FIX: each combo gets its OWN clean train/test
        train_combo, test_combo = build_dataset_for_combo(
            base_df, exog_test
        )
        
        print(f"\nCombo: {exog_test}")
        print(f"  Train rows: {len(train_combo)}, "
              f"Test rows: {len(test_combo)}")  # ← verify row counts differ correctly
        
        preds = []
        history_y_temp  = train_combo[TARGET_METRIC].astype(float).copy()
        history_df_temp = train_combo.copy()
        
        try:
            for i in range(len(test_combo)):
                model = SARIMAX(
                    history_y_temp,
                    exog=history_df_temp[exog_test],
                    order=(2, 1, 1),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                ).fit(disp=False)
                
                new_row = test_combo.iloc[[i]].copy()
                
                # Update lag features dynamically 
                for feat in exog_test:
                    if feat in LAG_FEATURE_CONFIG:
                        source_col, lag_n = LAG_FEATURE_CONFIG[feat]
                        # use last value from history as the lag
                        if lag_n == 1:
                            new_row[feat] = history_y_temp.iloc[-1] \
                                if source_col == TARGET_METRIC \
                                else history_df_temp[source_col].iloc[-1]
                        else:
                            new_row[feat] = history_df_temp[source_col].iloc[-lag_n]
                
                exog_step = new_row[exog_test]
                pred = model.get_forecast(
                    steps=1, exog=exog_step
                ).predicted_mean.iloc[0]
                
                preds.append(pred)
                
                # walk forward update
                history_y_temp = pd.concat([
                    history_y_temp,
                    pd.Series([pred], index=[test_combo.index[i]])
                ])
                new_row[TARGET_METRIC] = pred
                history_df_temp = pd.concat([history_df_temp, new_row])
            
            actuals  = test_combo[TARGET_METRIC].values
            preds_arr = np.array(preds)
            mape = np.mean(
                np.abs((actuals - preds_arr) / actuals)
            ) * 100
            
            results.append({
                'features':    exog_test,
                'mape':        mape,
                'train_rows':  len(train_combo),  # ← track this!
                'test_rows':   len(test_combo)
            })
            print(f"  MAPE: {mape:.2f}%")
            
        except Exception as e:
            print(f"  Failed: {e}")
            continue

# ─────────────────────────────────────────────
# STEP 4: Results — now apples to apples
# ─────────────────────────────────────────────
results_df = pd.DataFrame(results).sort_values('mape')

print("\nTop 5 Feature Combinations:")
print(results_df[['features', 'mape', 'train_rows']].head())
