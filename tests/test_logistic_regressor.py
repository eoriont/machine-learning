
import sys
sys.path.append("src")
try:
    from otest import do_assert, color_print
    from dataframe import DataFrame
except ImportError as e:
    print(e)
