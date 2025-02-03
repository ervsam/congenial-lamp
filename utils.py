# %%
from collections import defaultdict, deque
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import copy


# %%
# TOPOLOGICAL SORT

def find_all_cycles(edges, num_nodes):
    # Create graph representation
    graph = defaultdict(list)
    for (u, v), direction in edges.items():
        if direction == 0:
            graph[u].append(v)
        else:
            graph[v].append(u)

    all_cycles = []

    def dfs(start_node, current_node, visited, path, edge_path):
        # Mark the current node as visited and add it to the path
        visited.add(current_node)
        path.append(current_node)

        for neighbor in graph[current_node]:
            if neighbor == start_node and len(path) > 1:
                # Found a cycle; add it to the list
                all_cycles.append(
                    (path.copy(), edge_path.copy() + [(current_node, neighbor)]))
            elif neighbor not in visited:
                # Continue exploring the graph
                dfs(start_node, neighbor, visited, path,
                    edge_path + [(current_node, neighbor)])

        # Backtrack
        path.pop()
        visited.remove(current_node)

    # Run DFS from each node to find all cycles
    for node in range(num_nodes):
        dfs(node, node, set(), [], [])

    # Removing duplicate cycles (considering permutations)
    unique_cycles = []
    seen = set()

    for cycle_nodes, cycle_edges in all_cycles:
        cycle_signature = tuple(sorted(cycle_nodes))
        if cycle_signature not in seen:
            seen.add(cycle_signature)
            unique_cycles.append((cycle_nodes, cycle_edges))

    return unique_cycles


def topological_sort(edges, num_nodes):
    # Step 1: Create graph representation
    graph = {node: [] for node in range(num_nodes)}
    indegree = {node: 0 for node in range(num_nodes)}

    # To track the reverse graph for cycle detection
    reverse_graph = {node: [] for node in range(num_nodes)}

    for (u, v), direction in edges.items():
        if direction == 0:
            # Edge from u to v
            graph[u].append(v)
            reverse_graph[v].append(u)
            indegree[v] += 1
        else:
            # Edge from v to u
            graph[v].append(u)
            reverse_graph[u].append(v)
            indegree[u] += 1

    # Step 2: Find all nodes with no incoming edges
    zero_indegree = deque([node for node in graph if indegree[node] == 0])

    # Step 3: Perform topological sort
    top_order = []

    while zero_indegree:
        node = zero_indegree.popleft()
        top_order.append(node)

        for neighbor in graph[node]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                zero_indegree.append(neighbor)

    # Step 4: Check for cycle
    if len(top_order) == len(indegree):
        return {"topological_order": top_order, "cycle": None}
    else:
        cycles = find_all_cycles(edges, num_nodes)

        return {"topological_order": None, "cycle": cycles}


# %% LOGGER

class Logger:
    def __init__(self, filename="log.txt"):
        self.filename = filename
        with open(filename, 'w') as file:
            pass

    def save(self, filename):
        with open(filename, 'w') as file:
            for log in self.logs:
                file.write(" ".join(map(str, log)) + "\n")

    def print(self, *args, end="\n"):
        args = [round(a, 3) if isinstance(a, (int, float)) else a for a in args]

        print(*args)

        # Additional content can be appended using 'a' mode (append mode)
        with open(self.filename, 'a') as file:
            file.write(" ".join(map(str, args)) + end)

    def set_filename(self, filename):
        self.filename = filename
        with open(filename, 'w') as file:
            pass

    def reset(self):
        with open(self.filename, 'w') as file:
            pass

# %% PLOT


def plot(x, ylabel, xlabel, filename):
    # save plot of losses
    plt.plot(x)
    # plot running average
    plt.plot(np.convolve(x, np.ones(100)/100, mode='valid'))
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.savefig(filename)
    # plt.show()
    plt.close()


from collections import defaultdict, deque
import time

class DirectedGraph:
    def __init__(self, n):
        self.graph = defaultdict(list)  # Adjacency list representation
        self.n = n  # Number of nodes
    
    def add_edge(self, u, v):
        """Adds a directed edge if it doesn't form a cycle."""
        self.graph[u].append(v)
        if self._has_cycle():
            self.graph[u].pop()  # Remove the edge if it forms a cycle
            return False  # Edge not added (forms a cycle)
        return True  # Edge added successfully
    
    def _has_cycle(self):
        """Detects if the current graph has a cycle using Kahn's Algorithm."""
        indegree = {i: 0 for i in range(self.n)}
        
        # Compute in-degrees
        for u in self.graph:
            for v in self.graph[u]:
                indegree[v] += 1
        
        # Collect all nodes with indegree 0
        queue = deque([node for node in range(self.n) if indegree[node] == 0])
        visited_count = 0
        
        while queue:
            current = queue.popleft()
            visited_count += 1
            for neighbor in self.graph[current]:
                indegree[neighbor] -= 1
                if indegree[neighbor] == 0:
                    queue.append(neighbor)
        
        # If visited nodes are less than total nodes, there is a cycle
        return visited_count != len(indegree)

def sample_priorities(env, logger, close_pairs, preds, policy='random'):
    MAX_TRIES_RESOLVE_CYCLE = 3*len(env.get_close_pairs())

    # Sample an praction based on the probabilities
    probs = F.softmax(preds, dim=1)
    if policy == 'stochastic':
        actions = torch.multinomial(probs, 1)
    elif policy == 'random':
        actions = torch.randint(0, 2, (len(close_pairs), 1))
    elif policy == 'greedy':
        actions = torch.argmax(preds, dim=1).view(-1, 1)

    pairs_action = {}
    pair_qval = {}
    edges = []
    for pair, confidence, action in zip(close_pairs, probs, actions):
        t = round((max(confidence) - min(confidence)).item(), 3)
        if action == 1:
            pair_qval[pair[::-1]] = t
            edges.append(pair[::-1])
        else:
            pair_qval[pair] = t
            edges.append(pair)
        pairs_action[pair] = action
    pair_qval = dict(sorted(pair_qval.items(), key=lambda item: item[1], reverse=True))

    graph = DirectedGraph(env.num_agents)
    for pair, (u, v) in zip(close_pairs, edges):
        if graph.add_edge(u, v): # if edge doesn't form a cycle
            continue
        else:
            pairs_action[pair] = 1 - pairs_action[pair]

    actions = list(pairs_action.values())

    partial_prio = dict()
    pred_value = dict()
    predictions = dict()
    probabilities = dict()

    for i, (agent, neighbor) in enumerate(close_pairs):
        predictions[(agent, neighbor)] = preds[i]
        probabilities[(agent, neighbor)] = probs[i]
        pred_value[(agent, neighbor)] = preds[i][actions[i]]
        partial_prio[(agent, neighbor)] = actions[i]

    # while there is a cycle
    start_time = time.time()
    priorities, cycle = topological_sort(partial_prio, num_nodes=env.num_agents).values()
    if cycle:
        logger.print("Cycle exists")

    cycle_count = 0
    min_confidence_count = 0
    while cycle:
        # if there is a cycle, break the cycle by flipping the priority of the edge with the lowest confidence to 0
        min_confidence = float('inf')
        min_pair = None
        cycle = sorted(cycle, key=lambda x: len(x[0]), reverse=True)[0]
        # cycle_edges = [tuple(sorted(t)) for t in cycle[1]]
        cycle_set = [set(c) for c in cycle[1]]
        cycle_edges = []
        for pairs in close_pairs:
            if set(pairs) in cycle_set:
                cycle_edges.append(pairs)
        # logger.print cycle_edges and their predictions
        logger.print("edges, prio, probs")
        for c in cycle_edges:
            partial_prio[c].item()
            probabilities[c][partial_prio[c]].item()
            logger.print(c, partial_prio[c].item(),
                         probabilities[c][partial_prio[c]].item())
        for pair, confidence in probabilities.items():
            if pair in cycle_edges and confidence[partial_prio[pair]] < min_confidence:
                min_confidence = confidence[partial_prio[pair]]
                min_pair = pair
        logger.print("flip", min_pair, "from",
                     partial_prio[min_pair].item(), "to", 1-partial_prio[min_pair].item())


        # if 3 cycles in a row that consists of all edges confidence 1, resample priorities
        if min_confidence_count == 1:
            min_confidence_count += 1
            if cycle_count == 3:
                actions = torch.multinomial(probs, 1)
                for i, (agent, neighbor) in enumerate(close_pairs):
                    predictions[(agent, neighbor)] = preds[i]
                    probabilities[(agent, neighbor)] = probs[i]
                    pred_value[(agent, neighbor)] = preds[i][actions[i]]
                    partial_prio[(agent, neighbor)] = actions[i]
                min_confidence_count = 0
        else:
            min_confidence_count = 0

            partial_prio[min_pair] = 1-partial_prio[min_pair]
            pred_value[min_pair] = predictions[min_pair][partial_prio[min_pair]]
            # set probability of min_pair to 1 to avoid further flipping
            probabilities[min_pair][partial_prio[min_pair]] = 1

        # if cycle persists after MAX_TRIES_RESOLVE_CYCLE iterations, return None
        cycle_count += 1
        if cycle_count == MAX_TRIES_RESOLVE_CYCLE:
            logger.print(f"Cycle exists after {MAX_TRIES_RESOLVE_CYCLE} tries")
            return None, None, None

        priorities, cycle = topological_sort(partial_prio, num_nodes=env.num_agents).values()

        if cycle is None:
            logger.print("Time to resolve cycles:", time.time()-start_time)

    return priorities, partial_prio, pred_value


def step(env, logger, throughput, q_vals, policy="random"):
    '''
    1. sample priority ordering using q_vals
    2. 

    Returns
    =======
        
    '''

    if q_vals is None:
        priorities = list(range(env.num_agents))
        new_start, new_goals = env.step(priorities)
        return None, None, None, new_start, new_goals
    
    # while priorities are not feasible, sample new pairwise priorities
    tries = 0
    while tries < 10:
        tries += 1
        priorities, partial_prio, pred_value = sample_priorities(env, logger, env.get_close_pairs(), q_vals[0], policy=policy)
        
        # cycle unresolvable
        if priorities is None:
            logger.print(
                "Cycle unresolved, skipping instance\n")
            env.reset()
            new_start, new_goals = env.starts, env.goals
            return None, None, None, None, None

        # print time it takes to take step in env (A* planning)
        start_time = time.time()
        new_start, new_goals = env.step(priorities)
        logger.print("Time to env.step:", time.time()-start_time)
        if new_start is not None:
            throughput.append(env.goal_reached)
            break
        logger.print("Priorities not feasible, resampling\n")
    # problem instance unresolvable
    if tries == 10:
        logger.print(
            "Priorities not feasible after 10 tries, skipping instance\n")
        env.reset()
        new_start, new_goals = env.starts, env.goals
        return None, None, None, None, None

    return priorities, partial_prio, pred_value, new_start, new_goals, throughput


# %% PRIORITY BASED SEARCH (PBS)

import heapq
from collections import defaultdict, deque

class Node:
    def __init__(self):
        self.plan = {}  # Plan for each agent
        self.constraints = set()  # Constraints on agents
        self.priority_order = set()  # Partial order of agent priorities
        self.cost = float('inf')  # Cost of the plan

def low_level_search(start, goal, grid, constraints, agent_id, max_time=100):
    """A* search for a single agent that respects constraints."""
    def heuristic(pos):
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    def get_neighbors(pos):
        x, y = pos
        neighbors = [(x, y)]
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and grid[nx][ny] == 0:
                neighbors.append((nx, ny))
        return neighbors

    open_set = []
    heapq.heappush(open_set, (heuristic(start), 0, start, []))
    visited = set()

    while open_set:
        _, g, current, path = heapq.heappop(open_set)
        if (current, len(path)) in visited:
            continue
        visited.add((current, len(path)))

        # Goal check
        if current == goal:
            return path + [current]

        # Expand neighbors
        for neighbor in get_neighbors(current):
            conflict = False

            for c in constraints:
                if c['agent'] != agent_id:
                    if c['type'] == 'vertex':
                        # Check for vertex conflict
                        if c['pos'] == neighbor and c['time'] == g + 1:
                            conflict = True
                            break
                    elif c['type'] == 'edge':
                        # Check for edge conflict
                        
                        # if current != neighbor:
                        if c['edge'] == (neighbor, current) and c['time'] == g + 1:
                            conflict = True
                            break
                        # Also check reverse edge for wait actions being misinterpreted as conflicts
                        if c['edge'] == (neighbor, current) and c['time'] == g + 1:
                            conflict = True
                            break

            if not conflict:
                heapq.heappush(open_set, (g + heuristic(neighbor), g + 1, neighbor, path + [current]))


    return None  # No path found

def detect_collision(plan):
    """Detect the first vertex or edge collision in the plan."""
    time_dict = defaultdict(list)
    edge_dict = defaultdict(list)

    for agent, path in plan.items():
        for t, pos in enumerate(path):
            time_dict[(pos, t)].append(agent)

            agents = time_dict[(pos, t)]
            if len(agents) > 1:
                return {'type': 'vertex', 'time': t, 'pos': pos, 'agents': agents}
            
            if t < len(path) - 1:
                edge = (path[t], path[t + 1])
                if edge[0] != edge[1]:
                    edge_dict[(edge, t + 1)].append(agent)

                    agents = edge_dict[(edge, t + 1)]
                    reverse_edge = ((edge[1], edge[0]), t+1)
                    if reverse_edge in edge_dict.keys():
                        agents += edge_dict[reverse_edge]
                        return {'type': 'edge', 'time': t, 'edge': edge, 'agents': agents}

    # for (pos, t), agents in time_dict.items():
    #     if len(agents) > 1:
    #         return {'type': 'vertex', 'time': t, 'pos': pos, 'agents': agents}
        
    # for (edge, t), agents in edge_dict.items():
    #     reverse_edge = ((edge[1], edge[0]), t)
    #     if reverse_edge in edge_dict.keys():
    #         agents += edge_dict[reverse_edge]
    #         return {'type': 'edge', 'time': t, 'edge': edge, 'agents': agents}

    return None  # No collisions detected

def update_plan(node, agent_id, grid, starts, goals):
    """Update the plan for an agent considering the current priority order."""
    # Topological sorting of priorities
    graph = defaultdict(list)
    # for agent in range(len(starts)):
    #     graph[agent] = []

    for a, b in node.priority_order:
        graph[a].append(b)
        if b not in graph:
            graph[b] = []

    try:
        sorted_agents = topological_sort_pbs(graph)
    except ValueError:
        return False  # Cyclic dependency in priorities
    
    constraints = []  # Constraints are derived from priorities
    higher_priority_agents = sorted_agents[:sorted_agents.index(agent_id)]
    for higher_agent in higher_priority_agents:
        higher_agent_path = node.plan[higher_agent]
        for t, pos in enumerate(higher_agent_path):
            constraints.append({'type': 'vertex', 'agent': higher_agent, 'pos': pos, 'time': t})
            if t < len(higher_agent_path) - 1:
                edge = (higher_agent_path[t], higher_agent_path[t + 1])
                constraints.append({'type': 'edge', 'agent': higher_agent, 'edge': edge, 'time': t + 1})
    
    lower_priority_agents = sorted_agents[sorted_agents.index(agent_id):]
    for agent in lower_priority_agents:
        # Perform low-level search for the current agent
        path = low_level_search(starts[agent], goals[agent], grid, constraints, agent)
        if path is None:
            return False  # No solution found for the current agent
        node.plan[agent] = path
    
        for time, pos in enumerate(node.plan[agent]):
            # Add a vertex constraint
            constraints.append({'type': 'vertex', 'agent': agent, 'pos': pos, 'time': time})
            # Add an edge constraint (to prevent moving into `a`'s next position)
            if time < len(node.plan[agent]) - 1:
                edge = (node.plan[agent][time], node.plan[agent][time + 1])
                constraints.append({'type': 'edge', 'agent': agent, 'edge': edge, 'time': time + 1})

    return True

def topological_sort_pbs(graph):
    """Topological sort on a directed graph."""
    in_degree = {node: 0 for node in graph}
    for node in graph:
        for neighbor in graph[node]:
            in_degree[neighbor] += 1

    zero_in_degree = deque([node for node in graph if in_degree[node] == 0])
    sorted_order = []

    while zero_in_degree:
        current = zero_in_degree.popleft()
        sorted_order.append(current)
        for neighbor in graph[current]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                zero_in_degree.append(neighbor)

    if len(sorted_order) == len(graph):
        return sorted_order
    else:
        raise ValueError("Graph has a cycle and cannot be topologically sorted.")


def priority_based_search(grid, starts, goals):
    """Main PBS algorithm."""
    root = Node()
    for i, start in enumerate(starts):
        path = low_level_search(start, goals[i], grid, [], i)
        if path is None:
            return "No Solution"
        root.plan[i] = path

    root.cost = sum(len(path) for path in root.plan.values())
    stack = [root]

    while stack:
        node = stack.pop()
        collision = detect_collision(node.plan)

        if not collision:
            return node.plan, node.priority_order

        new_nodes = []
        ai, aj = collision['agents'][:2]  # Handle the first collision
        for agent in [ai, aj]:
            new_node = Node()
            new_node.plan = dict(node.plan)
            new_node.priority_order = set(node.priority_order)
            new_node.priority_order.add((aj, ai) if agent == ai else (ai, aj))

            success = update_plan(new_node, agent, grid, starts, goals)
            if success:
                new_node.cost = sum(len(path) for path in new_node.plan.values())
                new_nodes.append(new_node)

        # sort non-increasing order of cost
        new_nodes.sort(key=lambda x: x.cost, reverse=True)
        stack.extend(new_nodes)

    return "No Solution"



def prioritized_planning(grid, starts, goals, priorities):
    constraints = []
    paths = [[] for _ in range(len(starts))]

    for agent in priorities:
        path = low_level_search(starts[agent], goals[agent], grid, constraints, agent)
        if path is None:
            return agent
        paths[agent] = path
    
        for time, pos in enumerate(paths[agent]):
            # Add a vertex constraint
            constraints.append({'type': 'vertex', 'agent': agent, 'pos': pos, 'time': time})
            # Add an edge constraint (to prevent moving into `a`'s next position)
            if time < len(paths[agent]) - 1:
                edge = (paths[agent][time], paths[agent][time + 1])
                constraints.append({'type': 'edge', 'agent': agent, 'edge': edge, 'time': time + 1})

    return paths


def smoothen_paths(paths, smoothness=10):
    new_paths = []
    for path in paths:
        new_path = []
        for i in range(len(path)-1):
            new_path.append(path[i])
            for j in range(1, smoothness):
                dir_y = path[i+1][0] - path[i][0]
                dir_x = path[i+1][1] - path[i][1]
                new_c = (path[i][0]+(dir_y/smoothness*j), path[i][1]+(dir_x/smoothness*j))
                new_path.append(new_c)
        new_paths.append(new_path)

    return new_paths