import sys
sys.path.append("tests")
import railroad_travel_time
from otest import do_assert


railroad_segments = [('B', 'C'), ('B', 'A'), ('A', 'D'),
                     ('E', 'D'), ('C', 'F'), ('G', 'C')]

do_assert("from scratch 1", railroad_travel_time.order_towns_by_travel_time_from_scratch(
    'D', railroad_segments), ['D', 'A', 'E', 'B', 'C', 'F', 'G'])

do_assert("with tree class 1", railroad_travel_time.order_towns_by_travel_time_using_tree_class(
    'D', railroad_segments), ['D', 'E', 'A', 'B', 'C', 'G', 'F'])


do_assert("from scratch 2", railroad_travel_time.order_towns_by_travel_time_from_scratch(
    'A', railroad_segments), ['A', 'B', 'D', 'C', 'F', 'G', 'E'])

do_assert("with tree class 2", railroad_travel_time.order_towns_by_travel_time_using_tree_class(
    'A', railroad_segments), ['A', 'D', 'B', 'E', 'C', 'F', 'G'])
