import sys
import os
sys.path.append("src/models")
from otest import do_assert
from k_means import KMeans

data = [[0.14, 0.14, 0.28, 0.44],
        [0.22, 0.1, 0.45, 0.33],
        [0.1, 0.19, 0.25, 0.4],
        [0.02, 0.08, 0.43, 0.45],
        [0.16, 0.08, 0.35, 0.3],
        [0.14, 0.17, 0.31, 0.38],
        [0.05, 0.14, 0.35, 0.5],
        [0.1, 0.21, 0.28, 0.44],
        [0.04, 0.08, 0.35, 0.47],
        [0.11, 0.13, 0.28, 0.45],
        [0.0, 0.07, 0.34, 0.65],
        [0.2, 0.05, 0.4, 0.37],
        [0.12, 0.15, 0.33, 0.45],
        [0.25, 0.1, 0.3, 0.35],
        [0.0, 0.1, 0.4, 0.5],
        [0.15, 0.2, 0.3, 0.37],
        [0.0, 0.13, 0.4, 0.49],
        [0.22, 0.07, 0.4, 0.38],
        [0.2, 0.18, 0.3, 0.4]]

classes = ['Shortbread',
            'Fortune',
            'Shortbread',
            'Sugar',
            'Fortune',
            'Shortbread',
            'Sugar',
            'Shortbread',
            'Sugar',
            'Shortbread',
            'Sugar',
            'Fortune',
            'Shortbread',
            'Fortune',
            'Sugar',
            'Shortbread',
            'Sugar',
            'Fortune',
            'Shortbread']
initial_clusters = {
    1: [0,3,6,9,12,15,18],
    2: [1,4,7,10,13,16],
    3: [2,5,8,11,14,17]
    }
kmeans = KMeans(initial_clusters, data)

### ITERATION 1

do_assert("first update clusters", kmeans.clusters,
{
    1: [0, 3, 6, 9, 12, 15, 18],
    2: [1, 4, 7, 10, 13, 16],
    3: [2, 5, 8, 11, 14, 17]
})
do_assert("first update centers", kmeans.centers,
{
    1: [0.11285714285714286, 0.1457142857142857, 0.3242857142857143, 0.43714285714285717],
    2: [0.12166666666666666, 0.115, 0.35333333333333333, 0.42666666666666675],
    3: [0.11666666666666668, 0.10999999999999999, 0.3516666666666666, 0.4166666666666667]
})
do_assert("first update classes", {cluster_number: [classes[i] for i in cluster_indices] \
    for cluster_number, cluster_indices in kmeans.clusters.items()},
{
    1: ['Shortbread', 'Sugar', 'Sugar', 'Shortbread', 'Shortbread', 'Shortbread', 'Shortbread'],
    2: ['Fortune', 'Fortune', 'Shortbread', 'Sugar', 'Fortune', 'Sugar'],
    3: ['Shortbread', 'Shortbread', 'Sugar', 'Fortune', 'Sugar', 'Fortune']
})

### ITERATION 2
kmeans.update_clusters_once()

do_assert("second update clusters", kmeans.clusters,
{
    1: [0, 2, 5, 6, 7, 9, 10, 12, 15, 18],
    2: [14, 16],
    3: [1, 3, 4, 8, 11, 13, 17]
})

do_assert("second update centers",  kmeans.centers,
{
    1: [0.11100000000000002, 0.15799999999999997, 0.30199999999999994, 0.44800000000000006],
    2: [0.0, 0.115, 0.4, 0.495],
    3: [0.15857142857142859, 0.08, 0.38285714285714284, 0.37857142857142856]
})

do_assert("second update classes",  {cluster_number: [classes[i] for i in cluster_indices] \
    for cluster_number, cluster_indices in kmeans.clusters.items()},
{
    1: ['Shortbread', 'Shortbread', 'Shortbread', 'Sugar', 'Shortbread', 'Shortbread', 'Sugar', 'Shortbread', 'Shortbread', 'Shortbread'],
    2: ['Sugar', 'Sugar'],
    3: ['Fortune', 'Sugar', 'Fortune', 'Sugar', 'Fortune', 'Fortune', 'Fortune']
})

### ITERATION 3
kmeans.update_clusters_once()

do_assert("third update clusters",  kmeans.clusters,
{
    1: [0, 2, 5, 7, 9, 12, 15, 18],
    2: [3, 6, 8, 10, 14, 16],
    3: [1, 4, 11, 13, 17]
})

do_assert("third update centers",  kmeans.centers,
{
    1: [0.1325, 0.17124999999999999, 0.29125, 0.41625000000000006],
    2: [0.018333333333333337, 0.10000000000000002, 0.37833333333333335, 0.5099999999999999],
    3: [0.21000000000000002, 0.07999999999999999, 0.38000000000000006, 0.346]
})

do_assert("third update centers",  {cluster_number: [classes[i] for i in cluster_indices] \
    for cluster_number, cluster_indices in kmeans.clusters.items()},
{
    1: ['Shortbread', 'Shortbread', 'Shortbread', 'Shortbread', 'Shortbread', 'Shortbread', 'Shortbread', 'Shortbread'],
    2: ['Sugar', 'Sugar', 'Sugar', 'Sugar', 'Sugar', 'Sugar'],
    3: ['Fortune', 'Fortune', 'Fortune', 'Fortune', 'Fortune']
})
