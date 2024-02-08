"""Interface to `gristmill`.
"""

try:
    from pyspark import SparkContext
    HAS_SPARK = True
except ImportError:
    HAS_SPARK = False

try:
    import drudge
    HAS_DRUDGE = True
except ImportError:
    HAS_DRUDGE = False

try:
    import gristmill
    HAS_GRISTMILL = True
except ImportError:
    HAS_GRISTMILL = False


