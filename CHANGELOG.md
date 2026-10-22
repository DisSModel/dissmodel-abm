# Changelog

## [Unreleased]

- `Group.filter()` and `Trajectory.filter()` now follow TerraME: they apply
  `select` again to the current members only. The new `rebuild()` builds
  the subset again from its target (what `filter()` did before), and
  `sort()` sorts it. `filter()` without `select`, `sort()` without a key, and
  `sort` with `random` raise `ValueError`.
- `Schelling` changes with it: as in TerraME, the group of unhappy agents
  empties while about 5 % of the agents are still unhappy (before, almost
  none were).
- Tests against TerraME: `tests/test_goldens.py` compares every ported
  `logo` model with 50 TerraME runs from `terrame-goldens`
  (`pytest -m golden`), and a test checks the mean population of
  `LifeCycle` against its exact expectation. The README replaces the
  comparison with the single reference charts of `logo`: the differences
  it listed in `Ants` and `Disease` were chance.

## [0.2.0] — unreleased

Complete rewrite in the style of TerraME. The 0.1 API (agents as rows of a
`GeoDataFrame`, `AgentModel` methods such as `walk`, `die_if`,
`reproduce_if`) is removed.

- Core: `Agent` (one Python class per agent type, with `init()` and
  `execute()`), `Society`, `Group`, `Trajectory`, a raster `CellSpace` whose
  cell attributes are the bands of a dissmodel `RasterBackend`, and
  `AgentModel` with TerraME's placement strategies (`random`, `uniform`,
  `spread`, `void`).
- Reproducibility: `AgentModel(seed=...)` owns one NumPy generator used by
  every society, cell space and agent; the same seed gives the same run. In
  0.1 the seed covered only the initial state.
- Social networks and messages: `SocialNetwork`,
  `Society.create_social_network` (nine strategies, `inmemory`,
  `symmetric`), `Agent.message` with `on_message` and `on_<subject>`, and
  delayed delivery through `Society.synchronize`.
- Models ported from TerraME `logo` 0.5.6: `SingleAgent`, `Schelling`,
  `PredatorPrey`, `LifeCycle`, `Ants`, `Disease`, checked against the
  reference charts of `logo` (see the README for the two open differences,
  in `Ants` and `Disease`).
- Examples: `examples/cli/run_logo.py` and
  `examples/streamlit/logo_explorer.py`.
- `CellSpace` cost follows the number of agents, not the size of the grid:
  placement, `sample()` and the new `sample_empty()`, `count_empty()` and
  `add_agent_band()` work on NumPy indices and create `Cell` objects only
  for the cells reached by agents; band access through `Cell` is 2.5 to 7
  times faster. With 1000 agents on a 1000 × 1000 grid, setup went from
  4.4 s to 0.04 s and a step from 725 ms to 18 ms. The ported models use
  agent bands and give the same results as before for the same seed.
- `examples/cli/fire_brigade.py` and `tests/test_cellspace_scale.py`: agents
  and a `RasterCellularAutomaton` (the fire of `dissmodel-ca`) on one shared
  backend.
- Requires Python 3.11 and dissmodel 0.6.6. CI runs ruff, mypy and pytest.

## [0.1.0]

First version: agents as rows of a `GeoDataFrame` (not released).
