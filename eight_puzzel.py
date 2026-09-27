"""
8-puzzle solver using A*, comparing four heuristics:

    manhattan   sum of each tile's Manhattan distance to its goal square
    misplaced   number of tiles not on their goal square
    nmaxswap    cost of the relaxed puzzle where the blank may swap with any tile
    rowcol      number of tiles in the wrong row + number in the wrong column

All four are admissible, so A* returns an optimal solution with each of them.
They differ only in how many states the search has to expand.

    python eight_puzzel.py
    python eight_puzzel.py --trials 20
    python eight_puzzel.py --puzzle "7 2 4 5 0 6 8 3 1"
"""

import argparse
import heapq
import random
import statistics
import time

# A state is a 9-tuple read row by row; 0 is the blank.
SIZE = 3
BLANK = 0
GOAL = (1, 2, 3, 4, 5, 6, 7, 8, 0)

# GOAL_INDEX[tile] = square where that tile belongs
GOAL_INDEX = tuple(GOAL.index(t) for t in range(9))


def _neighbours(i):
    r, c = divmod(i, SIZE)
    return tuple(
        (r + dr) * SIZE + (c + dc)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
        if 0 <= r + dr < SIZE and 0 <= c + dc < SIZE
    )


NEIGHBOURS = tuple(_neighbours(i) for i in range(9))


def successors(state):
    """Every state one slide away."""
    b = state.index(BLANK)
    for n in NEIGHBOURS[b]:
        nxt = list(state)
        nxt[b], nxt[n] = nxt[n], nxt[b]
        yield tuple(nxt)


def is_solvable(state):
    """Solvable iff the tile permutation has an even number of inversions."""
    tiles = [t for t in state if t != BLANK]
    return sum(
        1
        for i in range(len(tiles))
        for j in range(i + 1, len(tiles))
        if tiles[i] > tiles[j]
    ) % 2 == 0


# --------------------------------------------------------------------------- #
# Heuristics
# --------------------------------------------------------------------------- #

def h_manhattan(state):
    """Sum of row + column distance from each tile to its goal square."""
    total = 0
    for i, tile in enumerate(state):
        if tile != BLANK:
            r, c = divmod(i, SIZE)
            gr, gc = divmod(GOAL_INDEX[tile], SIZE)
            total += abs(r - gr) + abs(c - gc)
    return total


def h_misplaced(state):
    """Count of tiles not on their goal square."""
    return sum(1 for i, t in enumerate(state) if t != BLANK and t != GOAL[i])


def h_nmaxswap(state):
    """Exact cost of the relaxed puzzle where the blank swaps with any tile.

    Greedy and optimal: if the blank is not home, swap in the tile that belongs
    on its square, which places that tile for good. If the blank is home but
    tiles remain misplaced, spend one swap to enter the next cycle.
    """
    perm = list(state)
    blank = perm.index(BLANK)
    swaps = 0
    while True:
        if blank != GOAL_INDEX[BLANK]:
            src = perm.index(GOAL[blank])
        else:
            src = next(
                (i for i, t in enumerate(perm) if t != BLANK and t != GOAL[i]),
                None,
            )
            if src is None:
                return swaps
        perm[blank], perm[src] = perm[src], perm[blank]
        blank = src
        swaps += 1


def h_rowcol(state):
    """Tiles in the wrong row, plus tiles in the wrong column."""
    total = 0
    for i, tile in enumerate(state):
        if tile != BLANK:
            r, c = divmod(i, SIZE)
            gr, gc = divmod(GOAL_INDEX[tile], SIZE)
            total += (r != gr) + (c != gc)
    return total


HEURISTICS = {
    "manhattan": h_manhattan,
    "misplaced": h_misplaced,
    "nmaxswap": h_nmaxswap,
    "rowcol": h_rowcol,
}


# --------------------------------------------------------------------------- #
# A* search
# --------------------------------------------------------------------------- #

def astar(start, heuristic):
    """A* with unit step costs. Returns (cost, nodes expanded, seconds, path)."""
    t0 = time.perf_counter()
    tie = 0
    frontier = [(heuristic(start), 0, tie, start)]
    best_g = {start: 0}
    parent = {start: None}
    expanded = 0

    while frontier:
        _, g, _, state = heapq.heappop(frontier)
        if g > best_g[state]:           # stale heap entry
            continue
        if state == GOAL:
            return g, expanded, time.perf_counter() - t0, _path(parent, state)

        expanded += 1
        ng = g + 1
        for nxt in successors(state):
            if ng < best_g.get(nxt, 1 << 30):
                best_g[nxt] = ng
                parent[nxt] = state
                tie += 1
                heapq.heappush(frontier, (ng + heuristic(nxt), ng, tie, nxt))

    raise ValueError("unsolvable puzzle")


def _path(parent, state):
    """Walk parent links back from `state` to the start board."""
    out = [state]
    while parent[out[-1]] is not None:
        out.append(parent[out[-1]])
    out.reverse()
    return out


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #

def random_board(rng):
    """A uniformly random solvable board."""
    tiles = list(range(9))
    while True:
        rng.shuffle(tiles)
        if is_solvable(tiles):
            return tuple(tiles)


def parse_puzzle(text):
    state = tuple(int(p) for p in text.replace(",", " ").split())
    if sorted(state) != list(range(9)):
        raise ValueError("need a permutation of 0..8, with 0 as the blank")
    if not is_solvable(state):
        raise ValueError("that board is unsolvable")
    return state


def print_solution(path, per_row=6):
    """Draw the solution boards left to right, wrapping every `per_row`."""
    print(f"\none optimal solution ({len(path) - 1} moves):\n")
    for start in range(0, len(path), per_row):
        chunk = path[start:start + per_row]
        more = start + per_row < len(path)
        for r in range(SIZE):
            cells = [
                " ".join("_" if v == BLANK else str(v)
                         for v in b[r * SIZE:(r + 1) * SIZE])
                for b in chunk
            ]
            sep = "  >  " if r == 1 else "     "
            tail = "  >" if r == 1 and more else ""
            print("  " + sep.join(cells) + tail)
        print()


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--puzzle", help='one board, e.g. "7 2 4 5 0 6 8 3 1"')
    p.add_argument("--trials", type=int, default=10, help="boards to test")
    p.add_argument("--seed", type=int, default=2024)
    args = p.parse_args()

    if args.puzzle:
        try:
            boards = [parse_puzzle(args.puzzle)]
        except ValueError as exc:
            p.error(str(exc))
    else:
        rng = random.Random(args.seed)
        boards = [random_board(rng) for _ in range(args.trials)]

    names = list(HEURISTICS)
    expanded = {n: [] for n in names}
    times = {n: [] for n in names}
    depths = []

    print("\n8-PUZZLE: A* HEURISTIC COMPARISON\n")
    print(f"{'board':<22}{'depth':>6}  " + "".join(f"{n:>12}" for n in names))
    print("-" * (30 + 12 * len(names)))

    for board in boards:
        results = {n: astar(board, h) for n, h in HEURISTICS.items()}
        costs = {r[0] for r in results.values()}
        assert len(costs) == 1, f"heuristics disagree on optimal cost: {costs}"
        cost = costs.pop()
        depths.append(cost)

        for n in names:
            expanded[n].append(results[n][1])
            times[n].append(results[n][2])

        print(f"{' '.join(map(str, board)):<22}{cost:>6}  "
              + "".join(f"{results[n][1]:>12,}" for n in names))

    means = {n: statistics.fmean(expanded[n]) for n in names}
    ranked = sorted(names, key=lambda x: means[x])
    best, worst = ranked[0], ranked[-1]

    print(f"\nnodes expanded, mean over {len(boards)} board(s) "
          f"(mean depth {statistics.fmean(depths):.1f}):\n")
    for n in ranked:
        # Every board was already solved, so there is nothing to compare.
        ratio = f"{means[n] / means[best]:>8.1f}x" if means[best] else ""
        print(f"  {n:<12}{means[n]:>10,.0f} nodes"
              f"{statistics.fmean(times[n]) * 1000:>9.1f} ms" + ratio)

    if means[best]:
        print(f"\nbest: {best} ({means[worst] / means[best]:.1f}x fewer nodes "
              f"than {worst})")

    if len(boards) == 1:
        print_solution(results["manhattan"][3])
    else:
        print()


if __name__ == "__main__":
    main()
