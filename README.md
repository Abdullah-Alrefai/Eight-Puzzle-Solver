# Eight Puzzle Solver

Solves the 8-puzzle with A* search and compares four heuristics:

| Heuristic | What it counts |
|---|---|
| `manhattan` | How far each tile is from its goal square |
| `misplaced` | Tiles not on their goal square |
| `nmaxswap` | Moves needed if the blank could swap with any tile |
| `rowcol` | Tiles in the wrong row + tiles in the wrong column |

All four find the shortest solution. The program shows how many states each one has to check.

## Run it

No packages needed.

```bash
python eight_puzzel.py
python eight_puzzel.py --trials 20
python eight_puzzel.py --puzzle "7 2 4 5 0 6 8 3 1"
```
