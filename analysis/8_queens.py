import random


def show_board(locs):
    print(''.join('  '.join((str(locs.index((x, y))) if (x, y) in locations else ".")
                            for y in range(8)) + "\n" for x in range(8)))


def calc_cost(locs):
    cost = []
    for i, (x, y) in enumerate(locs):
        for j, (x1, y1) in enumerate(locs):
            if i == j:
                continue
            if x == x1 or y == y1 or abs((y-y1)/(x-x1)) == 1:
                if (j, i) not in cost:
                    cost.append((i, j))
    return len(cost)


def random_optimizer(n):
    locs = [random.sample([(x, y) for x in range(8)
                           for y in range(8)], 8) for _ in range(n)]
    optimized = min(locs, key=calc_cost)
    return {
        'locations': optimized,
        'cost': calc_cost(optimized)
    }


locations = [(0, 0), (6, 1), (2, 2), (5, 3), (4, 4), (7, 5), (1, 6), (2, 6)]
show_board(locations)
print(calc_cost(locations))

print(random_optimizer(10))
print(random_optimizer(50))
print(random_optimizer(100))
print(random_optimizer(500))
print(random_optimizer(1000))
