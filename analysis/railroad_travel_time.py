import sys
sys.path.append("src/graphs")
from tree import Tree


def order_towns_by_travel_time_using_tree_class(starting_town, segments):
    railroad_segments = segments[:]
    result = []
    starting_index = 0
    for i, segment in enumerate(railroad_segments):
        if starting_town in segment:
            starting_index = i
            if segment.index(starting_town) == 0:
                result.append(segment)
            else:
                result.append(segment[::-1])
            railroad_segments.remove(segment)
            break
    railroad_segments_len = len(railroad_segments)
    railroad_segments = rotate(railroad_segments, starting_index)
    while len(result) < railroad_segments_len + 1:
        for x, y in railroad_segments:
            parents, children = list(zip(*result))
            if x in parents or x in children:
                result.append((x, y))
                railroad_segments.remove((x, y))
            elif y in parents or y in children:
                result.append((y, x))
                railroad_segments.remove((x, y))
    tree = Tree.build_from_edges(result)
    return tree.breadth_first_search()


def rotate(l, n):
    return l[n:] + l[:n]


def order_towns_by_travel_time_from_scratch(starting_town, railroad_segments):
    queue = [starting_town]
    while len(queue) < len(railroad_segments) + 1:
        for item in queue:
            for segment in railroad_segments:
                if item in segment:
                    new_item = list(segment)
                    new_item.remove(item)
                    new_item = new_item[0]
                    if new_item not in queue:
                        queue.insert(0, new_item)
    return queue[::-1]

if __name__ == "__main__":
    railroad_segments = [('B', 'C'), ('B', 'A'), ('A', 'D'),
                        ('E', 'D'), ('C', 'F'), ('G', 'C')]
    print(order_towns_by_travel_time_from_scratch('D', railroad_segments))
