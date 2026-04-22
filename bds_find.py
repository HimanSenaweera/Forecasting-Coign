import statsmodels
print(statsmodels.__version__)

# Search for bds across all statsmodels modules
import pkgutil
import importlib
import statsmodels

for importer, modname, ispkg in pkgutil.walk_packages(
    path        = statsmodels.__path__,
    prefix      = statsmodels.__name__ + '.',
    onerror     = lambda x: None
):
    try:
        mod = importlib.import_module(modname)
        if hasattr(mod, 'bds'):
            print(f"✅ Found bds in: {modname}")
    except:
        continue
