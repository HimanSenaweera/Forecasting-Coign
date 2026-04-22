import pandas                 as pd
import numpy                  as np
import itertools
import warnings
warnings.filterwarnings('ignore')

# Stationarity (Step 1)
from statsmodels.tsa.stattools     import adfuller
from statsmodels.tsa.stattools     import kpss

# Causality (Step 5)
from statsmodels.tsa.stattools     import grangercausalitytests

# Non-linearity (Step 4)
from statsmodels.tsa.stattools     import bds          # ✅ correct location
from statsmodels.stats.diagnostic  import acorr_ljungbox
from statsmodels.stats.diagnostic  import linear_reset
from statsmodels.tsa.ar_model      import AutoReg
from statsmodels.regression.linear_model import OLS
from statsmodels.tools             import add_constant

# VIF (Step 3)
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Correlation + MI (Step 2)
from sklearn.feature_selection     import mutual_info_regression
from sklearn.preprocessing         import StandardScaler
from scipy                         import stats as scipy_stats

# Transfer Entropy (Step 5 — non-linear)
try:
    from pyinform import transfer_entropy as pyinform_te
    TE_AVAILABLE = True
    print("✅ Transfer Entropy (pyinform) available")
except ImportError:
    TE_AVAILABLE = False
    print("⚠️  pyinform not installed → pip install pyinform")

print("✅ All imports done")
