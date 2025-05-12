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
import random

from st_astar import space_time_astar


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
    plt.yscale('log')
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

class SafeGraph:
    def __init__(self, n):
        self.n = n
        self.adj = {i: set() for i in range(n)}
        self.reach = {i: set() for i in range(n)}  # transitive closure approximation

    def add_edge(self, u, v):
        if u == v or v in self.reach[u]:  # would create a cycle
            return False

        # Add edge safely
        self.adj[u].add(v)

        # Update reachable sets
        # Everyone who can reach u now can also reach v and v's reachable
        for x in range(self.n):
            if u in self.reach[x]:
                self.reach[x].update(self.reach[v])
                self.reach[x].add(v)
        self.reach[u].update(self.reach[v])
        self.reach[u].add(v)
        return True

def sample_priorities(env, logger, close_pairs, preds, policy='random', epsilon=0.1):
    MAX_TRIES_RESOLVE_CYCLE = 3*len(env.get_close_pairs())

    # Sample an praction based on the probabilities
    probs = F.softmax(preds, dim=1)
    if policy == 'stochastic':
        actions = torch.multinomial(probs, 1)
    elif policy == 'random':
        actions = torch.randint(0, 2, (len(close_pairs), 1))
    elif policy == 'greedy':
        actions = torch.argmax(preds, dim=1).view(-1, 1)
    elif policy == 'epsilon_greedy':
        actions = torch.multinomial(probs, 1)
        if random.random() < epsilon:
            actions = torch.randint(0, 2, (len(close_pairs), 1))
        else:
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
    for u, v in list(pair_qval.keys()):
        ori_pair = (min(u,v), max(u,v))
        if graph.add_edge(u, v): # if edge doesn't form a cycle
            continue
        else:
            assert graph.add_edge(v, u)
            pairs_action[ori_pair] = 1 - pairs_action[ori_pair]

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


def step(env, logger, throughput, q_vals, policy="random", epsilon=0.1, pbs_epsilon=0.1, obs_fovs=None, buffer=None, old_start=None, old_goals=None):    
    USE_PENALTY = False

    if q_vals is None:
        priorities = list(range(env.num_agents))
        start_time = time.time()
        new_start, new_goals = env.step(priorities)
        logger.print("Time to env.step:", time.time()-start_time)
        if new_start is not None:
            throughput.append(env.goal_reached)
        return None, None, None, new_start, new_goals, throughput

    close_pairs = env.get_close_pairs()
    
    # while priorities are not feasible, sample new pairwise priorities
    if random.random() < pbs_epsilon:
        # use PBS to sample priorities
        result = priority_based_search(env.grid_map, env.starts, env.goals, env.window_size)
        if result == "No Solution":
            logger.print("No solution found using PBS, skipping instance\n")
            env.reset()
            new_start, new_goals = env.starts, env.goals
            return None, None, None, None, None, throughput
        else:
            plan, priority_order = result

            paths = plan.values()
            for i, path in enumerate(paths):
                print(f"Agent {i}: {path}")

            graph = defaultdict(list)
            for agent in range(env.num_agents):
                graph[agent] = []
            for a, b in priority_order:
                graph[a].append(b)
            priorities = topological_sort_pbs(graph)

            print(priorities)
            start_time = time.time()
            new_start, new_goals = env.step(priorities)
            logger.print("Time to env.step:", time.time()-start_time)

            # assert new_start is not None
            if new_start is None:
                logger.print("PBS and PP does not match, skipping instance\n")
                env.reset()
                new_start, new_goals = env.starts, env.goals
                return None, None, None, None, None, throughput

            partial_prio = dict()
            pred_value = dict()
            for i, (agent, neighbor) in enumerate(close_pairs):
                action = int(priorities.index(agent) > priorities.index(neighbor))
                partial_prio[(agent, neighbor)] = torch.tensor([action])
                pred_value[(agent, neighbor)] = q_vals[0][i][action]
                
            logger.print(f"Solution found using PBS with PBS_epsilon: {pbs_epsilon:.2}\n")

    else:
        # run the sampling process with PBS as a fallback
        tries = 0
        MAX_TRIES = 100
        while tries < MAX_TRIES:
            tries += 1
            priorities, partial_prio, pred_value = sample_priorities(env, logger, close_pairs, q_vals[0], policy=policy, epsilon=epsilon)
            
            # cycle unresolvable
            if priorities is None:
                logger.print("Cycle unresolved, skipping instance\n")
                env.reset()
                new_start, new_goals = env.starts, env.goals
                return None, None, None, None, None, throughput

            # print time it takes to take step in env (A* planning)
            start_time = time.time()
            new_start, new_goals = env.step(priorities)
            logger.print("Time to env.step:", time.time()-start_time)
            if new_start is not None:
                throughput.append(env.goal_reached)
                break

            if USE_PENALTY:
                # if priorities are not feasible, save samples and give negative reward
                # 1. step per group, and get delay of group
                # 2. insert to buffer
                groups = env.connected_edge_groups()
                local_rewards = []
                group_agents = []

                for group in groups:
                    set_s = set()
                    for pair in group:
                        set_s.update(pair)
                    group_agents.append(list(set_s))
                    
                for group in group_agents:
                    prio = []
                    for a in priorities:
                        if a in group:
                            prio.append(a)

                    delays = env.step_per_group(prio)
                    if delays is None:
                        # local delay is -10
                        logger.print(f"group {group} failed")
                        local_rewards.append(-10)
                    else:
                        # local delay is average
                        local_rewards.append(10 * sum([-delays[agent] for agent in group]) / len(group))

                # save to buffer
                buffer.insert(obs_fovs, partial_prio, -1, local_rewards, groups, old_start, old_goals)

            logger.print(f"buffer length in function step: {len(buffer)}")

            logger.print("Priorities not feasible, resampling\n")
            
        # problem instance unresolvable
        if tries == MAX_TRIES:
            logger.print(f"Priorities not feasible after {MAX_TRIES} tries")
            # try using PBS
            logger.print(f"running PBS...")
            logger.print(env.starts, env.goals)
            result = priority_based_search(env.grid_map, env.starts, env.goals, env.window_size)
            if result == "No Solution":
                logger.print("No solution found using PBS, skipping instance\n")
                env.reset()
                new_start, new_goals = env.starts, env.goals
                return None, None, None, None, None, throughput

            else:
                plan, priority_order = result

                paths = plan.values()
                for i, path in enumerate(paths):
                    print(f"Agent {i}: {path}")

                graph = defaultdict(list)
                for agent in range(env.num_agents):
                    graph[agent] = []
                for a, b in priority_order:
                    graph[a].append(b)
                priorities = topological_sort_pbs(graph)

                print(priorities)
                start_time = time.time()
                new_start, new_goals = env.step(priorities)
                logger.print("Time to env.step:", time.time()-start_time)

                # assert new_start is not None
                if new_start is None:
                    logger.print("PBS and PP does not match, skipping instance\n")
                    env.reset()
                    new_start, new_goals = env.starts, env.goals
                    return None, None, None, None, None, throughput

                partial_prio = dict()
                pred_value = dict()
                for i, (agent, neighbor) in enumerate(close_pairs):
                    action = int(priorities.index(agent) > priorities.index(neighbor))
                    partial_prio[(agent, neighbor)] = torch.tensor([action])
                    pred_value[(agent, neighbor)] = q_vals[0][i][action]
                    
                logger.print("Solution found using PBS\n")

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

def low_level_search(start, goals, grid, constraints, agent_id, window_size, max_time=100):
    """A* search for a single agent that respects constraints."""
    def heuristic(pos, goal):
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    def get_neighbors(pos):
        x, y = pos
        neighbors = [(x, y)] # wait action
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and grid[nx][ny] == 0:
                neighbors.append((nx, ny))
        return neighbors

    # just look for path to the first 5 goals
    goals = goals[:window_size].copy()
    goal = goals.pop(0)

    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), 0, start, []))
    visited = set()

    while open_set:
        _, g, current, path = heapq.heappop(open_set)
        if (current, len(path)) in visited:
            continue
        visited.add((current, len(path)))

        # Goal check
        if current == goal:
            # handle next goal
            if goals:
                goal = goals.pop(0)
                open_set = []
                heapq.heappush(open_set, (heuristic(current, goal), g, current, path))
                visited = set()
                _, g, current, path = heapq.heappop(open_set)

            else:
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
                heapq.heappush(open_set, (g + heuristic(neighbor, goal), g + 1, neighbor, path + [current]))


    return None  # No path found

def detect_collision(plan, window_size):
    """Detect the first vertex or edge collision in the plan."""
    time_dict = defaultdict(list)
    edge_dict = defaultdict(list)

    for agent, path in plan.items():
        for t, pos in enumerate(path):
            if t > window_size:
                break
            
            time_dict[(pos, t)].append(agent)

            agents = time_dict[(pos, t)]
            if len(agents) > 1:
                return {'type': 'vertex', 'time': t, 'pos': pos, 'agents': agents}
            
            if t < window_size:
                edge = (path[t], path[t + 1])
                if edge[0] != edge[1]:
                    edge_dict[(edge, t + 1)].append(agent)

                    agents = edge_dict[(edge, t + 1)]
                    reverse_edge = ((edge[1], edge[0]), t+1)
                    if reverse_edge in edge_dict.keys():
                        agents += edge_dict[reverse_edge]
                        return {'type': 'edge', 'time': t, 'edge': edge, 'agents': agents}

    return None  # No collisions detected

def get_sub_order(agent, priority_order, lower=True):
    """
    Given an agent and a set of pairwise constraints (priority_order),
    return the set S = {agent} ∪ {j | agent ≺_N j} (transitively).
    
    Args:
        agent: The starting agent (e.g., an integer or identifier).
        priority_order: A set of tuples (a, b) meaning agent a has higher priority than agent b.
        
    Returns:
        A set of agents that includes the starting agent and all agents that come
        after it in the priority ordering.
    """
    S = set()
    stack = [agent]
    while stack:
        current = stack.pop()
        # Find all agents j such that there is a constraint (current, j)
        for (a, b) in priority_order:
            if lower:
                if a == current and b not in S:
                    S.add(b)
                    stack.append(b)
            else:
                if b == current and a not in S:
                    S.add(a)
                    stack.append(a)
    return S

def update_plan(node, agent_id, grid, starts, goals, window_size):
    """Update the plan for an agent considering the current priority order."""
    # Topological sorting of priorities
    graph = defaultdict(list)

    for a, b in node.priority_order:
        graph[a].append(b)
        if b not in graph:
            graph[b] = []

    # LIST ← topological sorting on partially ordered set ({i} ∪ {j|i ≺N j}, ≺≺≺N )
    higher_than_agent = get_sub_order(agent_id, node.priority_order, lower=False)
    lower_than_agent = get_sub_order(agent_id, node.priority_order, lower=True)

    try:
        sorted_agents = topological_sort_pbs(graph)
    except ValueError:
        return False  # Cyclic dependency in priorities
    
    dynamic_constraints = dict()
    edge_constraints = dict()

    for higher_agent in higher_than_agent:
        higher_agent_path = node.plan[higher_agent]

        for t, pos in enumerate(higher_agent_path):
            if t > window_size:
                break

            y, x = pos
            dynamic_constraints[(y, x, t)] = higher_agent
            # add edge constraints (only add the first window elements)
            if t < window_size:
                y_2, x_2 = higher_agent_path[t+1]
                edge = ((y, x, t), (y_2, x_2, t+1))
                edge_constraints[edge] = higher_agent
    
    lower_priority_agents = sorted_agents[sorted_agents.index(agent_id):]

    for agent in lower_priority_agents:
        if agent == agent_id or agent in lower_than_agent:
            # Perform low-level search for the current agent
            path = space_time_astar(np.array(grid), starts[agent], goals[agent][:window_size], dynamic_constraints, edge_constraints)
            
            if path is None:
                return False  # No solution found for the current agent

            mod_path = [(x, y) for (x, y, t) in path]

            node.plan[agent] = mod_path

            for y, x, t in path:
                if t > window_size:
                    break
                dynamic_constraints[(y, x, t)] = agent

                if t < window_size:
                    y_2, x_2, _ = path[t+1]
                    edge = ((y, x, t), (y_2, x_2, t+1))
                    edge_constraints[edge] = agent

    return True

def topological_sort_pbs(graph):
    """Topological sort on a directed graph."""
    in_degree = {node: 0 for node in graph}

    random_order = list(graph.keys())
    random.shuffle(random_order)

    for node in graph:
        for neighbor in graph[node]:
            in_degree[neighbor] += 1

    zero_in_degree = deque([node for node in random_order if in_degree[node] == 0])
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


def priority_based_search(grid, starts, goals, window_size, max_time=10000):
    """Main PBS algorithm."""
    start_time = time.time()
    root = Node()
    for agent, start in enumerate(starts):
        # path to next goal
        # path = low_level_search(start, goals[agent], grid, [], agent, window_size)
        path = space_time_astar(np.array(grid), start, goals[agent][:window_size], dict(), dict())

        if path is None:
            return "No Solution"
        
        path = [(x, y) for (x, y, t) in path]

        root.plan[agent] = path

    root.cost = sum(len(path) for path in root.plan.values())
    stack = [root]

    while stack:
        if time.time() - start_time >= max_time:
            print("MAX PBS TIME REACHED")
            return "No Solution"
        
        node = stack.pop()
        collision = detect_collision(node.plan, window_size)

        if not collision:
            return (node.plan, node.priority_order)

        new_nodes = []
        ai, aj = collision['agents'][:2]  # Handle the first collision
        for agent in [ai, aj]:
            new_node = Node()
            new_node.plan = dict(node.plan)
            new_node.priority_order = set(node.priority_order)
            new_node.priority_order.add((aj, ai) if agent == ai else (ai, aj))

            success = update_plan(new_node, agent, grid, starts, goals, window_size)
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



# %% WAREHOUSE MAP
from matplotlib.colors import ListedColormap

cmap = ListedColormap(["white", "black", "yellow", "red"])

def generate_warehouse(rows, cols, num_agents, num_goals, seed=None):
    np.random.seed(seed)

    people = np.array(
        [[2,2,0,2,2],
        [2,2,0,2,2],
        [2,2,0,2,2],
        ]
    )

    shelves = np.array(
        [[3]*10,
        [1]*10,
        [3]*10,
        ]
    )

    hall_ver = np.array(
        [[0],
        [0],
        [0],
        ]
    )

    row = people

    for j in range(cols):
        row = np.concatenate((row, hall_ver, shelves), axis=1)

    row = np.concatenate((row, hall_ver, people), axis=1)

    hall_hor = np.array(
        [[0]*len(row[0])]
    )

    warehouse = row

    for i in range(rows-1):
        warehouse = np.concatenate((warehouse, hall_hor, row), axis=0)

    # pad the warehouse with a row of zeros
    warehouse = np.pad(warehouse, 1, 'constant')

    plt.imshow(warehouse, cmap=cmap, vmin=0, vmax=3)

    # generate random starts and goals
    station_locs = np.where(warehouse == 2)
    station_locs = np.array(list(zip(station_locs[0], station_locs[1])))

    # get possible goal locations
    shelves_locs = np.where(warehouse == 3)
    shelves_locs = np.array(list(zip(shelves_locs[0], shelves_locs[1])))

    # generate randomly selected start and goal locations
    np.random.shuffle(station_locs)
    np.random.shuffle(shelves_locs)

    start_locs = station_locs[:num_agents]

    rng = np.random.default_rng()
    rng.shuffle(shelves_locs)
    goal_locs = np.array([[i] for i in shelves_locs[:num_agents].copy()])
    for i in range(num_goals):
        rng.shuffle(shelves_locs)
        new_shelves = shelves_locs[:num_agents].copy()
        goal_locs = np.concatenate((goal_locs, new_shelves[:, None, :]), axis=1)

    return warehouse, start_locs, goal_locs


