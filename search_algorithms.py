import heapq
import time
import random
import math
import matplotlib.pyplot as plt
from collections import deque


# ============================================================
# GRAPH DATA
# ============================================================

UNWEIGHTED_GRAPH = {
    'S': ['A', 'B', 'C'],
    'A': ['S', 'D', 'E'],
    'B': ['S', 'E'],
    'C': ['S', 'F'],
    'D': ['A', 'G'],
    'E': ['A', 'B', 'G'],
    'F': ['C', 'G'],
    'G': ['D', 'E', 'F']
}

WEIGHTED_GRAPH = {
    'S': [('A', 2), ('B', 4), ('C', 5)],
    'A': [('S', 2), ('D', 4), ('E', 3)],
    'B': [('S', 4), ('E', 2)],
    'C': [('S', 5), ('F', 2)],
    'D': [('A', 4), ('G', 5)],
    'E': [('A', 3), ('B', 2), ('G', 4)],
    'F': [('C', 2), ('G', 3)],
    'G': [('D', 5), ('E', 4), ('F', 3)]
}

HEURISTIC = {
    'S': 7,
    'A': 6,
    'B': 5,
    'C': 4,
    'D': 4,
    'E': 3,
    'F': 2,
    'G': 0
}

START = 'S'
GOAL = 'G'


# ============================================================
# BFS
# ============================================================

def bfs(graph, start, goal):
    start_time = time.perf_counter()

    queue = deque([[start]])
    visited = {start}
    nodes = 0

    while queue:
        path = queue.popleft()
        current = path[-1]
        nodes += 1

        if current == goal:
            return path, nodes, time.perf_counter() - start_time

        for neighbour in graph[current]:
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append(path + [neighbour])

    return None, nodes, time.perf_counter() - start_time


# ============================================================
# DFS
# ============================================================

def dfs(graph, start, goal):
    start_time = time.perf_counter()

    stack = [[start]]
    visited = set()
    nodes = 0

    while stack:
        path = stack.pop()
        current = path[-1]

        if current in visited:
            continue

        visited.add(current)
        nodes += 1

        if current == goal:
            return path, nodes, time.perf_counter() - start_time

        for neighbour in reversed(graph[current]):
            if neighbour not in visited:
                stack.append(path + [neighbour])

    return None, nodes, time.perf_counter() - start_time


# ============================================================
# UNIFORM COST SEARCH
# ============================================================

def ucs(graph, start, goal):
    start_time = time.perf_counter()

    queue = [(0, start, [start])]
    visited = set()
    nodes = 0

    while queue:
        cost, current, path = heapq.heappop(queue)

        if current in visited:
            continue

        visited.add(current)
        nodes += 1

        if current == goal:
            return path, cost, nodes, time.perf_counter() - start_time

        for neighbour, weight in graph[current]:
            if neighbour not in visited:
                heapq.heappush(
                    queue,
                    (cost + weight, neighbour, path + [neighbour])
                )

    return None, 0, nodes, time.perf_counter() - start_time


# ============================================================
# GREEDY BEST FIRST SEARCH
# ============================================================

def greedy_best_first(graph, start, goal, heuristic):
    start_time = time.perf_counter()

    queue = [(heuristic[start], start, [start], 0)]
    visited = set()
    nodes = 0

    while queue:
        h, current, path, cost = heapq.heappop(queue)

        if current in visited:
            continue

        visited.add(current)
        nodes += 1

        if current == goal:
            return path, cost, nodes, time.perf_counter() - start_time

        for neighbour, weight in graph[current]:
            if neighbour not in visited:
                heapq.heappush(
                    queue,
                    (
                        heuristic[neighbour],
                        neighbour,
                        path + [neighbour],
                        cost + weight
                    )
                )

    return None, 0, nodes, time.perf_counter() - start_time


# ============================================================
# A STAR
# ============================================================

def a_star(graph, start, goal, heuristic):
    start_time = time.perf_counter()

    queue = [(heuristic[start], 0, start, [start])]
    visited = set()
    nodes = 0

    while queue:
        f, g, current, path = heapq.heappop(queue)

        if current in visited:
            continue

        visited.add(current)
        nodes += 1

        if current == goal:
            return path, g, nodes, time.perf_counter() - start_time

        for neighbour, weight in graph[current]:
            if neighbour not in visited:
                new_g = g + weight
                new_f = new_g + heuristic[neighbour]

                heapq.heappush(
                    queue,
                    (new_f, new_g, neighbour, path + [neighbour])
                )

    return None, 0, nodes, time.perf_counter() - start_time


# ============================================================
# N QUEENS COMMON
# ============================================================

def count_conflicts(board):
    conflicts = 0
    n = len(board)

    for i in range(n):
        for j in range(i + 1, n):

            if (
                board[i] == board[j]
                or abs(board[i] - board[j]) == abs(i - j)
            ):
                conflicts += 1

    return conflicts


def random_board(n):
    return [random.randint(0, n - 1) for _ in range(n)]


# ============================================================
# HILL CLIMBING
# ============================================================

def hill_climbing(n=8, max_restarts=100):
    start_time = time.perf_counter()

    states = 0
    best_board = None
    best_conflicts = float('inf')

    for _ in range(max_restarts):

        board = random_board(n)
        current_conflicts = count_conflicts(board)

        while True:
            states += 1

            best_move_board = board[:]
            best_move_conflicts = current_conflicts
            improved = False

            for col in range(n):

                original_row = board[col]

                for row in range(n):

                    if row == original_row:
                        continue

                    new_board = board[:]
                    new_board[col] = row

                    conflicts = count_conflicts(new_board)

                    if conflicts < best_move_conflicts:
                        best_move_conflicts = conflicts
                        best_move_board = new_board
                        improved = True

            board = best_move_board
            current_conflicts = best_move_conflicts

            if current_conflicts < best_conflicts:
                best_conflicts = current_conflicts
                best_board = board[:]

            if current_conflicts == 0:
                return (
                    board,
                    current_conflicts,
                    states,
                    time.perf_counter() - start_time
                )

            if not improved:
                break

    return (
        best_board,
        best_conflicts,
        states,
        time.perf_counter() - start_time
    )


# ============================================================
# SIMULATED ANNEALING
# ============================================================

def simulated_annealing(
    n=8,
    temperature=100.0,
    cooling_rate=0.995,
    minimum_temperature=0.01
):
    start_time = time.perf_counter()

    board = random_board(n)
    current_conflicts = count_conflicts(board)

    best_board = board[:]
    best_conflicts = current_conflicts

    states = 0

    while temperature > minimum_temperature:

        states += 1

        col = random.randint(0, n - 1)
        new_row = random.randint(0, n - 1)

        old_row = board[col]
        board[col] = new_row

        new_conflicts = count_conflicts(board)
        difference = new_conflicts - current_conflicts

        if difference < 0:
            current_conflicts = new_conflicts

        else:
            probability = math.exp(-difference / temperature)

            if random.random() < probability:
                current_conflicts = new_conflicts
            else:
                board[col] = old_row

        if current_conflicts < best_conflicts:
            best_conflicts = current_conflicts
            best_board = board[:]

        if best_conflicts == 0:
            break

        temperature *= cooling_rate

    return (
        best_board,
        best_conflicts,
        states,
        time.perf_counter() - start_time
    )


# ============================================================
# BACKTRACKING
# ============================================================

def is_safe(board, row, col):
    for previous_col in range(col):

        if (
            board[previous_col] == row
            or abs(board[previous_col] - row)
            == abs(previous_col - col)
        ):
            return False

    return True


def solve_backtracking(board, col, n, counter):

    if col == n:
        return True

    for row in range(n):

        counter[0] += 1

        if is_safe(board, row, col):

            board[col] = row

            if solve_backtracking(board, col + 1, n, counter):
                return True

            board[col] = -1

    return False


def backtracking_nqueens(n=8):
    start_time = time.perf_counter()

    board = [-1] * n
    counter = [0]

    solved = solve_backtracking(board, 0, n, counter)

    if solved:
        return (
            board,
            0,
            counter[0],
            time.perf_counter() - start_time
        )

    return (
        None,
        -1,
        counter[0],
        time.perf_counter() - start_time
    )


# ============================================================
# FORWARD CHECKING
# ============================================================

def forward_checking(board, col, n, domains, counter):

    if col == n:
        return True

    for row in list(domains[col]):

        counter[0] += 1
        board[col] = row

        saved_domains = [d.copy() for d in domains]

        valid = True

        for next_col in range(col + 1, n):

            new_domain = set()

            for possible_row in domains[next_col]:

                if (
                    possible_row != row
                    and abs(possible_row - row)
                    != abs(next_col - col)
                ):
                    new_domain.add(possible_row)

            domains[next_col] = new_domain

            if len(domains[next_col]) == 0:
                valid = False
                break

        if valid:

            if forward_checking(
                board,
                col + 1,
                n,
                domains,
                counter
            ):
                return True

        for i in range(n):
            domains[i] = saved_domains[i].copy()

        board[col] = -1

    return False


def forward_checking_nqueens(n=8):
    start_time = time.perf_counter()

    board = [-1] * n

    domains = [
        set(range(n))
        for _ in range(n)
    ]

    counter = [0]

    solved = forward_checking(
        board,
        0,
        n,
        domains,
        counter
    )

    if solved:
        return (
            board,
            0,
            counter[0],
            time.perf_counter() - start_time
        )

    return (
        None,
        -1,
        counter[0],
        time.perf_counter() - start_time
    )


# ============================================================
# SAVE GRAPH FUNCTION
# ============================================================

def save_bar_graph(names, values, title, ylabel, filename):

    plt.figure(figsize=(8, 5))

    plt.bar(names, values)

    plt.title(title)
    plt.xlabel("Algorithms")
    plt.ylabel(ylabel)

    plt.xticks(rotation=15)

    plt.tight_layout()

    plt.savefig(filename)

    plt.close()


# ============================================================
# MAIN
# ============================================================

def run_all():

    print("\n" + "=" * 70)
    print("AI SEARCH ALGORITHMS - PERFORMANCE EVALUATION")
    print("=" * 70)

    # GRAPH SEARCH

    bfs_path, bfs_nodes, bfs_time = bfs(
        UNWEIGHTED_GRAPH,
        START,
        GOAL
    )

    dfs_path, dfs_nodes, dfs_time = dfs(
        UNWEIGHTED_GRAPH,
        START,
        GOAL
    )

    ucs_path, ucs_cost, ucs_nodes, ucs_time = ucs(
        WEIGHTED_GRAPH,
        START,
        GOAL
    )

    greedy_path, greedy_cost, greedy_nodes, greedy_time = (
        greedy_best_first(
            WEIGHTED_GRAPH,
            START,
            GOAL,
            HEURISTIC
        )
    )

    astar_path, astar_cost, astar_nodes, astar_time = a_star(
        WEIGHTED_GRAPH,
        START,
        GOAL,
        HEURISTIC
    )

    graph_results = [
        ("BFS", bfs_path, 0, bfs_nodes, bfs_time),
        ("DFS", dfs_path, 0, dfs_nodes, dfs_time),
        ("UCS", ucs_path, ucs_cost, ucs_nodes, ucs_time),
        ("Greedy", greedy_path, greedy_cost, greedy_nodes, greedy_time),
        ("A*", astar_path, astar_cost, astar_nodes, astar_time)
    ]

    print("\nGRAPH SEARCH RESULTS")
    print("-" * 70)

    print(
        f"{'Algorithm':<12}"
        f"{'Path':<25}"
        f"{'Cost':<8}"
        f"{'Nodes':<10}"
        f"{'Time(s)':<10}"
    )

    print("-" * 70)

    for algorithm, path, cost, nodes, execution_time in graph_results:

        path_text = " -> ".join(path)

        cost_text = "-" if cost == 0 else str(cost)

        print(
            f"{algorithm:<12}"
            f"{path_text:<25}"
            f"{cost_text:<8}"
            f"{nodes:<10}"
            f"{execution_time:.6f}"
        )

    # N QUEENS

    hc_board, hc_conflicts, hc_states, hc_time = hill_climbing()

    sa_board, sa_conflicts, sa_states, sa_time = simulated_annealing()

    bt_board, bt_conflicts, bt_states, bt_time = backtracking_nqueens()

    fc_board, fc_conflicts, fc_states, fc_time = (
        forward_checking_nqueens()
    )

    queen_results = [
        ("Hill Climbing", hc_board, hc_conflicts, hc_states, hc_time),
        ("Sim Annealing", sa_board, sa_conflicts, sa_states, sa_time),
        ("Backtracking", bt_board, bt_conflicts, bt_states, bt_time),
        ("Forward Check", fc_board, fc_conflicts, fc_states, fc_time)
    ]

    print("\n8-QUEENS RESULTS")
    print("-" * 70)

    print(
        f"{'Algorithm':<18}"
        f"{'Conflicts':<12}"
        f"{'States':<12}"
        f"{'Time(s)':<10}"
    )

    print("-" * 70)

    for algorithm, board, conflicts, states, execution_time in queen_results:

        print(
            f"{algorithm:<18}"
            f"{conflicts:<12}"
            f"{states:<12}"
            f"{execution_time:.6f}"
        )

        print("Board:", board)

    # GRAPH SEARCH PNGS

    graph_names = [r[0] for r in graph_results]

    graph_nodes = [r[3] for r in graph_results]

    graph_times = [r[4] * 1000 for r in graph_results]

    graph_costs = [r[2] for r in graph_results]

    save_bar_graph(
        graph_names,
        graph_nodes,
        "Graph Search - Nodes Explored",
        "Nodes Explored",
        "graph_nodes.png"
    )

    save_bar_graph(
        graph_names,
        graph_times,
        "Graph Search - Execution Time",
        "Time in milliseconds",
        "graph_time.png"
    )

    save_bar_graph(
        graph_names,
        graph_costs,
        "Graph Search - Path Cost",
        "Path Cost",
        "graph_cost.png"
    )

    # N QUEENS PNGS

    queen_names = [r[0] for r in queen_results]

    queen_states = [r[3] for r in queen_results]

    queen_times = [r[4] * 1000 for r in queen_results]

    queen_conflicts = [r[2] for r in queen_results]

    save_bar_graph(
        queen_names,
        queen_states,
        "8-Queens - States Explored",
        "States Explored",
        "nqueens_states.png"
    )

    save_bar_graph(
        queen_names,
        queen_times,
        "8-Queens - Execution Time",
        "Time in milliseconds",
        "nqueens_time.png"
    )

    save_bar_graph(
        queen_names,
        queen_conflicts,
        "8-Queens - Final Conflicts",
        "Conflicts",
        "nqueens_conflicts.png"
    )

    print("\nAll result graphs saved successfully:")
    print("graph_nodes.png")
    print("graph_time.png")
    print("graph_cost.png")
    print("nqueens_states.png")
    print("nqueens_time.png")
    print("nqueens_conflicts.png")


# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":

    random.seed(42)

    run_all()