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
    try:
        # FIXME this sucks

        import collections
        import collections.abc
        collections.Iterable = collections.abc.Iterable

        import dummy_spark as pyspark
        from dummy_spark import SparkContext, SparkConf
    except ImportError:
        import pyspark
        from pyspark import SparkContext, SparkConf

    SPARK_CONF = SparkConf().setAppName("albert").setMaster("local")
    SPARK_CONTEXT = SparkContext(conf=SPARK_CONF)
    SPARK_CONTEXT.setLogLevel("ERROR")
except ImportError:
    pyspark = None
    SPARK_CONF = None
    SPARK_CONTEXT = None

try:
    import sympy
except ImportError:
    sympy = None

try:
    import drudge
except ImportError:
    drudge = None

try:
    import gristmill
except ImportError:
    gristmill = None


_dependencies = {
    "pyspark": pyspark,
    "sympy": sympy,
    "drudge": drudge,
    "gristmill": gristmill,
}


def check_dependency(*deps):
    """
    Decorate a function to check for a dependency before running a
    function.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            for dep in deps:
                if _dependencies[dep] is None:
                    raise ImportError(f"Function `{func.__name__}` requires `{dep}`")
            return func(*args, **kwargs)

        return wrapper

    return decorator
