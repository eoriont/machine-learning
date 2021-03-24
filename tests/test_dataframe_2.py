import sys
import os
sys.path.append("src/models")
from otest import do_assert
from dataframe import DataFrame

df = DataFrame.from_array(
    [
        ['Kevin Fray', 52, 100],
        ['Charles Trapp', 52, 75],
        ['Anna Smith', 52, 50],
        ['Sylvia Mendez', 52, 100],
        ['Kevin Fray', 53, 80],
        ['Charles Trapp', 53, 95],
        ['Anna Smith', 53, 70],
        ['Sylvia Mendez', 53, 90],
        ['Anna Smith', 54, 90],
        ['Sylvia Mendez', 54, 80],
    ],
    ['name', 'assignmentId', 'score']
)

do_assert("group by", df.group_by('name').to_array(),
[
    ['Kevin Fray', [52, 53], [100, 80]],
    ['Charles Trapp', [52, 53], [75, 95]],
    ['Anna Smith', [52, 53, 54], [50, 70, 90]],
    ['Sylvia Mendez', [52, 53, 54], [100, 90, 80]],
])

do_assert("aggregate count", df.group_by('name').aggregate('score', 'count').select(["name", "assignmentId", "countscore"]).to_array(),
[
    ['Kevin Fray', [52, 53], 2],
    ['Charles Trapp', [52, 53], 2],
    ['Anna Smith', [52, 53, 54], 3],
    ['Sylvia Mendez', [52, 53, 54], 3],
])

do_assert("aggregate max", df.group_by('name').aggregate('score', 'max').select(["name", "assignmentId", "maxscore"]).to_array(),
[
    ['Kevin Fray', [52, 53], 100],
    ['Charles Trapp', [52, 53], 95],
    ['Anna Smith', [52, 53, 54], 90],
    ['Sylvia Mendez', [52, 53, 54], 100],
])

do_assert("aggregate min", df.group_by('name').aggregate('score', 'min').select(["name", "assignmentId", "minscore"]).to_array(),
[
    ['Kevin Fray', [52, 53], 80],
    ['Charles Trapp', [52, 53], 75],
    ['Anna Smith', [52, 53, 54], 50],
    ['Sylvia Mendez', [52, 53, 54], 80],
])

do_assert("aggregate sum", df.group_by('name').aggregate('score', 'sum').select(["name", "assignmentId", "sumscore"]).to_array(),
[
    ['Kevin Fray', [52, 53], 180],
    ['Charles Trapp', [52, 53], 170],
    ['Anna Smith', [52, 53, 54], 210],
    ['Sylvia Mendez', [52, 53, 54], 270],
])

do_assert("aggregate avg", df.group_by('name').aggregate('score', 'avg').select(["name", "assignmentId", "avgscore"]).to_array(),
[
    ['Kevin Fray', [52, 53], 90.0],
    ['Charles Trapp', [52, 53], 85.0],
    ['Anna Smith', [52, 53, 54], 70.0],
    ['Sylvia Mendez', [52, 53, 54], 90.0],
])

df = DataFrame.from_array(
    [['Kevin', 'Fray', 5],
    ['Charles', 'Trapp', 17],
    ['Anna', 'Smith', 13],
    ['Sylvia', 'Mendez', 9]],
    columns = ['firstname', 'lastname', 'age']
)

do_assert("query 1", df.query('SELECT firstname, age').to_array(),
[['Kevin', 5],
['Charles', 17],
['Anna', 13],
['Sylvia', 9]])
#>================================================
df = DataFrame.from_array(
    [['Kevin', 'Fray', 5],
    ['Charles', 'Trapp', 17],
    ['Anna', 'Smith', 13],
    ['Sylvia', 'Mendez', 9]],
    columns = ['firstname', 'lastname', 'age']
)

do_assert("query 2", df.query("SELECT lastname, firstname, age ORDER BY age DESC").to_array(),
[['Trapp', 'Charles', 17],
['Smith', 'Anna', 13],
['Mendez', 'Sylvia', 9],
['Fray', 'Kevin', 5]])

do_assert("query 3", df.query("SELECT firstname ORDER BY lastname ASC").to_array(),
[['Kevin'],
['Sylvia'],
['Anna'],
['Charles']])

df = DataFrame.from_array(
    [['Kevin', 'Fray', 5],
    ['Melvin', 'Fray', 5],
    ['Charles', 'Trapp', 17],
    ['Carl', 'Trapp', 17],
    ['Anna', 'Smith', 13],
    ['Hannah', 'Smith', 13],
    ['Sylvia', 'Mendez', 9],
    ['Cynthia', 'Mendez', 9]],
    columns = ['firstname', 'lastname', 'age']
)

do_assert("query 4", df.query("SELECT lastname, firstname, age ORDER BY age ASC, firstname DESC").to_array(),
[['Fray', 'Melvin', 5],
['Fray', 'Kevin', 5],
['Mendez', 'Sylvia', 9],
['Mendez', 'Cynthia', 9],
['Smith', 'Hannah', 13],
['Smith', 'Anna', 13],
['Trapp', 'Charles', 17],
['Trapp', 'Carl', 17]])
