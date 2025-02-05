# SPACE TIME A* ALGORITHM

import heapq
import copy


class Node:
    def __init__(self, x, y, t, cost, heuristic, parent=None):
        self.x = x
        self.y = y
        self.t = t
        self.cost = cost
        self.heuristic = heuristic
        self.parent = parent

    def __lt__(self, other):
        return (self.cost + self.heuristic) < (other.cost + other.heuristic)


def heuristic(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)


def is_valid(old_x, old_y, x, y, t, grid, dynamic_constraints, edge_constraints):
    if x < 0 or y < 0 or x >= grid.shape[0] or y >= grid.shape[1]:
        return False
    if grid[x, y] == 1:
        return False
    if (x, y, t) in dynamic_constraints:
        return False
    # edge constraints
    if ((x, y, t-1), (old_x, old_y, t)) in edge_constraints:
        return False

    return True


def space_time_astar(grid, start, goals, dynamic_constraints, edge_constraints):
    goals = copy.deepcopy(goals)
    goal = goals.pop(0)
    start_node = Node(start[0], start[1], 0, 0, heuristic(start[0], start[1], goal[0], goal[1]))
    goal_node = Node(goal[0], goal[1], 0, 0, 0)

    open_list = []
    closed_list = set()

    heapq.heappush(open_list, start_node)

    while open_list:
        current_node = heapq.heappop(open_list)

        # IF GOAL IS FOUND
        if (current_node.x, current_node.y) == (goal_node.x, goal_node.y):
            # handle next goal
            if goals:
                goal = goals.pop(0)
                goal_node = Node(goal[0], goal[1], 0, 0, 0)
                new_node = Node(current_node.x, current_node.y, current_node.t, current_node.cost,
                                heuristic(current_node.x, current_node.y, goal_node.x, goal_node.y), current_node.parent)

                open_list = []
                heapq.heappush(open_list, new_node)
                current_node = heapq.heappop(open_list)
                closed_list = set()

            else:
                path = []
                while current_node:
                    path.append(
                        (current_node.x, current_node.y, current_node.t))
                    current_node = current_node.parent
                return path[::-1]

        # DOES ADDING THIS FIX? i think so
        # avoid looping or redundant searches in space-time A*
        if ((current_node.x, current_node.y, current_node.t)) in closed_list:
            continue

        closed_list.add((current_node.x, current_node.y, current_node.t))

        actions = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
        for dx, dy in actions:
            new_x = current_node.x + dx
            new_y = current_node.y + dy
            new_t = current_node.t + 1

            if is_valid(current_node.x, current_node.y, new_x, new_y, new_t, grid, dynamic_constraints, edge_constraints):
                new_node = Node(new_x, new_y, new_t, current_node.cost + 1,
                                heuristic(new_x, new_y, goal_node.x, goal_node.y), current_node)

                if (new_x, new_y, new_t) not in closed_list:
                    heapq.heappush(open_list, new_node)

    # no path found
    print("st_astar: No path found", start, goal)
    return None
