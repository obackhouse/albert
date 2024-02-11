"""
*albert*: Toolbox for manipulating, optimising, and generating code for
Einstein summations

The `albert` package is a toolbox for the manipulation, optimisation,
and code generation for collections of tensor contractions, in
Einstein summation format.
"""

__version__ = "0.0.0"

# Optional dependencies:

try:
    import pyspark
    from pyspark import SparkContext, SparkConf

    SPARK_CONF = SparkConf().setAppName("albert").setMaster("local")
    SPARK_CONTEXT = SparkContext(conf=SPARK_CONF)
    SPARK_CONTEXT.setLogLevel("ERROR")
except:
    pyspark = None
    SPARK_CONF = None
    SPARK_CONTEXT = None

try:
    import sympy
except:
    sympy = None

try:
    import drudge
except:
    drudge = None

try:
    import gristmill
except:
    gristmill = None


def check_dependency(*deps):
    """Decorator to check for a dependency before running a function."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            for dep in deps:
                if dep is None:
                    raise ImportError(f"Function `{func.__name__}` requires `{dep}`")
            return func(*args, **kwargs)

        return wrapper

    return decorator
