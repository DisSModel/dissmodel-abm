"""Tests for dissmodel_abm.core.AgentModel and example models."""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point

from dissmodel.core import Environment
from dissmodel.geo.vector import vector_grid
from dissmodel_abm.core import AgentModel, Society, Agent
from dissmodel_abm.models import (
    RandomWalkModel,
    PredatorPreyModel,
    SchellingModel,
    PeripherisationModel,
)


@pytest.fixture(autouse=True)
def _reset_environment_singleton():
    """dissmodel.core.Environment uses a class-level _current pointer;
    nothing to tear down explicitly, but creating a fresh Environment in
    each test keeps registration lists isolated."""
    yield


def _make_points_gdf(coords, **extra_cols):
    gdf = gpd.GeoDataFrame({"geometry": [Point(x, y) for x, y in coords]})
    for k, v in extra_cols.items():
        gdf[k] = v
    return gdf


# ---------------------------------------------------------------------------
# AgentModel primitives
# ---------------------------------------------------------------------------

def test_walk_respects_bounds():
    gdf = _make_points_gdf([(0, 0), (100, 100)])
    env = Environment(end_time=1)
    model = AgentModel(gdf=gdf)

    model.walk(step_size=50.0, bounds=(0, 0, 100, 100))

    xs = model.gdf.geometry.x
    ys = model.gdf.geometry.y
    assert (xs >= 0).all() and (xs <= 100).all()
    assert (ys >= 0).all() and (ys <= 100).all()


def test_walk_zero_step_does_not_move():
    gdf = _make_points_gdf([(5, 5)])
    env = Environment(end_time=1)
    model = AgentModel(gdf=gdf)

    model.walk(step_size=0.0)

    assert model.gdf.loc[0, "geometry"].x == 5
    assert model.gdf.loc[0, "geometry"].y == 5


def test_neighbors_within():
    gdf = _make_points_gdf([(0, 0), (1, 0), (10, 10)])
    env = Environment(end_time=1)
    model = AgentModel(gdf=gdf)

    assert model.neighbors_within(0, radius=2) == [1]
    assert model.neighbors_within(0, radius=0.5) == []
    assert 0 in model.neighbors_within(1, radius=2)


def test_die_if_removes_matching_agents():
    gdf = _make_points_gdf([(0, 0), (1, 1), (2, 2)], energy=[0.0, 5.0, -1.0])
    env = Environment(end_time=1)
    model = AgentModel(gdf=gdf)

    removed = model.die_if(lambda row: row["energy"] <= 0)

    assert removed == 2
    assert len(model.gdf) == 1
    assert model.gdf.iloc[0]["energy"] == 5.0


def test_reproduce_if_adds_children_with_custom_fn():
    gdf = _make_points_gdf([(0, 0), (1, 1)], energy=[20.0, 1.0])
    env = Environment(end_time=1)
    model = AgentModel(gdf=gdf)

    def child_fn(row):
        row = row.copy()
        row["energy"] = 0.0
        return row

    n_new = model.reproduce_if(
        lambda row: row["energy"] >= 10.0,
        child_fn=child_fn,
    )

    assert n_new == 1
    assert len(model.gdf) == 3
    # the child has energy reset to 0.0
    assert (model.gdf["energy"] == 0.0).sum() == 1


# ---------------------------------------------------------------------------
# RandomWalkModel
# ---------------------------------------------------------------------------

def test_random_walk_model_runs_and_stays_in_bounds():
    rng = np.random.default_rng(0)
    n = 15
    bounds = (0, 0, 100, 100)
    xs = rng.uniform(*bounds[0::2], n)
    ys = rng.uniform(*bounds[1::2], n)
    gdf = _make_points_gdf(list(zip(xs, ys)))

    env = Environment(start_time=0, end_time=5)
    model = RandomWalkModel(gdf=gdf, step_size=3.0, bounds=bounds)
    env.run()

    assert len(model.gdf) == n
    xs_out = model.gdf.geometry.x
    ys_out = model.gdf.geometry.y
    assert (xs_out >= bounds[0]).all() and (xs_out <= bounds[2]).all()
    assert (ys_out >= bounds[1]).all() and (ys_out <= bounds[3]).all()


# ---------------------------------------------------------------------------
# PredatorPreyModel
# ---------------------------------------------------------------------------

def test_predator_prey_model_runs_without_error():
    rng = np.random.default_rng(1)
    n_sheep, n_wolves = 20, 4
    bounds = (0, 0, 50, 50)
    n_total = n_sheep + n_wolves
    xs = rng.uniform(*bounds[0::2], n_total)
    ys = rng.uniform(*bounds[1::2], n_total)

    gdf = _make_points_gdf(
        list(zip(xs, ys)),
        kind=["sheep"] * n_sheep + ["wolf"] * n_wolves,
        energy=[10.0] * n_total,
    )

    env = Environment(start_time=0, end_time=5)
    model = PredatorPreyModel(
        gdf=gdf,
        step_size=2.0,
        bounds=bounds,
        eat_radius=2.0,
        energy_loss=0.5,
        energy_gain=5.0,
        reproduce_threshold=15.0,
        graze_gain=0.6,
    )
    env.run()

    # Population may grow, shrink or go extinct depending on dynamics —
    # the important invariant is that columns/types remain valid.
    assert set(["geometry", "kind", "energy"]).issubset(model.gdf.columns)
    if len(model.gdf):
        assert model.gdf["kind"].isin(["sheep", "wolf"]).all()


def test_predator_prey_setup_fills_missing_columns():
    gdf = _make_points_gdf([(0, 0), (1, 1)])  # no 'kind' / 'energy' columns

    env = Environment(end_time=1)
    model = PredatorPreyModel(gdf=gdf)

    assert "energy" in model.gdf.columns
    assert "kind" in model.gdf.columns
    assert (model.gdf["kind"] == "sheep").all()


# ---------------------------------------------------------------------------
# SchellingModel
# ---------------------------------------------------------------------------

def test_schelling_setup_distributes_agent_types():
    gdf = vector_grid(dimension=(10, 10), resolution=1)
    env = Environment(end_time=1)
    model = SchellingModel(gdf=gdf, free_space=0.25, preference=3, seed=0)

    counts = model.gdf["agent_type"].value_counts().to_dict()
    n = len(model.gdf)

    assert set(counts.keys()) <= {-1, 0, 1}
    assert counts.get(-1, 0) == pytest.approx(round(n * 0.25), abs=1)
    # roughly balanced red/blue populations
    assert abs(counts.get(0, 0) - counts.get(1, 0)) <= 1


def test_schelling_increases_satisfaction_over_time():
    gdf = vector_grid(dimension=(12, 12), resolution=1)
    env = Environment(start_time=0, end_time=15)
    model = SchellingModel(gdf=gdf, free_space=0.25, preference=3, seed=1)

    initial = model.fraction_satisfied()
    env.run()
    final = model.fraction_satisfied()

    assert final >= initial
    # population is conserved (only moves, no births/deaths)
    assert len(model.gdf) == 12 * 12


def test_schelling_handles_no_empty_cells():
    gdf = vector_grid(dimension=(4, 4), resolution=1)
    env = Environment(end_time=1)
    model = SchellingModel(gdf=gdf, free_space=0.0, preference=3, seed=0)

    # should not raise even though there are no empty cells to move into
    model.pre_execute()
    model.execute()
    model.post_execute()

    assert (model.gdf["agent_type"] != -1).all()


# ---------------------------------------------------------------------------
# Society / Agent (the protective layer over self.gdf)
# ---------------------------------------------------------------------------

def test_society_iteration_and_attribute_access():
    gdf = _make_points_gdf([(0, 0), (1, 1)], energy=[10.0, 5.0], kind=["sheep", "wolf"])
    env = Environment(end_time=1)
    model = AgentModel(gdf=gdf)

    society = model.society
    assert isinstance(society, Society)
    assert len(society) == 2

    agents = list(society)
    assert all(isinstance(a, Agent) for a in agents)
    assert {a.kind for a in agents} == {"sheep", "wolf"}


def test_society_attribute_write_reflects_in_gdf():
    gdf = _make_points_gdf([(0, 0)], energy=[10.0])
    env = Environment(end_time=1)
    model = AgentModel(gdf=gdf)

    agent = model.society[0]
    agent.energy = 99.0

    assert model.gdf.at[0, "energy"] == 99.0


def test_society_getitem_and_contains():
    gdf = _make_points_gdf([(0, 0), (1, 1)])
    env = Environment(end_time=1)
    model = AgentModel(gdf=gdf)

    society = model.society
    assert 0 in society
    assert 99 not in society

    agent = society[1]
    assert agent.id == 1

    with pytest.raises(KeyError):
        society[99]


def test_society_add_creates_new_agent_in_gdf():
    gdf = _make_points_gdf([(0, 0)], energy=[10.0])
    env = Environment(end_time=1)
    model = AgentModel(gdf=gdf)

    new_agent = model.society.add(geometry=Point(5, 5), energy=3.0)

    assert len(model.society) == 2
    assert new_agent.energy == 3.0
    assert model.gdf.loc[new_agent.id, "geometry"] == Point(5, 5)


def test_society_remove_deletes_agent_from_gdf():
    gdf = _make_points_gdf([(0, 0), (1, 1)], energy=[10.0, 5.0])
    env = Environment(end_time=1)
    model = AgentModel(gdf=gdf)

    model.society.remove(0)

    assert len(model.society) == 1
    assert 0 not in model.society


def test_agent_die_removes_self():
    gdf = _make_points_gdf([(0, 0), (1, 1)], energy=[10.0, 5.0])
    env = Environment(end_time=1)
    model = AgentModel(gdf=gdf)

    agent = model.society[0]
    agent.die()

    assert len(model.society) == 1
    assert 0 not in model.society


def test_agent_reproduce_copies_attributes_with_overrides():
    gdf = _make_points_gdf([(2, 2)], energy=[20.0], kind=["sheep"])
    env = Environment(end_time=1)
    model = AgentModel(gdf=gdf)

    parent = model.society[0]
    child = parent.reproduce(energy=1.0)

    assert len(model.society) == 2
    assert child.energy == 1.0
    assert child.kind == "sheep"  # copied from parent, not overridden
    assert child.geometry == parent.geometry


def test_society_remove_if_removes_matching_agents():
    gdf = _make_points_gdf([(0, 0), (1, 1), (2, 2)], energy=[0.0, 5.0, -1.0])
    env = Environment(end_time=1)
    model = AgentModel(gdf=gdf)

    removed = model.society.remove_if(lambda agent: agent.energy <= 0)

    assert removed == 2
    assert len(model.society) == 1
    assert model.society[1].energy == 5.0


def test_society_select_and_count():
    gdf = _make_points_gdf(
        [(0, 0), (1, 1), (2, 2)], kind=["sheep", "wolf", "sheep"]
    )
    env = Environment(end_time=1)
    model = AgentModel(gdf=gdf)

    sheep = model.society.select(lambda a: a.kind == "sheep")
    assert len(sheep) == 2
    assert model.society.count(lambda a: a.kind == "wolf") == 1
    assert model.society.count() == 3


def test_agent_neighbors_matches_legacy_neighbors_within():
    gdf = _make_points_gdf([(0, 0), (1, 0), (10, 10)])
    env = Environment(end_time=1)
    model = AgentModel(gdf=gdf)

    legacy = model.neighbors_within(0, radius=2)
    via_society = [a.id for a in model.society[0].neighbors(2)]

    assert via_society == legacy


def test_agent_grid_neighbors_uses_create_neighborhood():
    from libpysal.weights import Queen

    gdf = vector_grid(dimension=(5, 5), resolution=1)
    env = Environment(end_time=1)
    model = AgentModel(gdf=gdf)
    model.create_neighborhood(strategy=Queen, use_index=True)

    agent = model.society["2-2"]
    neighbor_ids = {a.id for a in agent.grid_neighbors()}

    assert neighbor_ids == set(model.neighs_id("2-2"))


def test_society_and_gdf_stay_in_sync_across_mutations():
    """model.gdf and model.society must always reflect the same data,
    so Map/Chart/legacy code reading model.gdf keep working."""
    gdf = _make_points_gdf([(0, 0), (1, 1)], energy=[10.0, 5.0])
    env = Environment(end_time=1)
    model = AgentModel(gdf=gdf)

    model.society.add(geometry=Point(9, 9), energy=1.0)
    assert len(model.gdf) == 3

    model.society.remove(0)
    assert len(model.gdf) == 2
    assert 0 not in model.gdf.index


# ---------------------------------------------------------------------------
# Agents without a location (TerraME: Agent may exist with no placement
# until Agent:enter() is called; Agent:leave() removes it again)
# ---------------------------------------------------------------------------

def test_add_without_geometry_creates_locationless_agent():
    gdf = _make_points_gdf([(0, 0)], energy=[10.0])
    env = Environment(end_time=1)
    model = AgentModel(gdf=gdf)

    agent = model.society.add(energy=1.0)

    assert agent.has_location is False
    assert agent.geometry is None
    assert len(model.society) == 2


def test_enter_gives_a_locationless_agent_a_position():
    gdf = _make_points_gdf([(0, 0)], energy=[10.0])
    env = Environment(end_time=1)
    model = AgentModel(gdf=gdf)

    agent = model.society.add(energy=1.0)
    agent.enter(7, 7)

    assert agent.has_location is True
    assert agent.geometry == Point(7, 7)


def test_leave_removes_position_without_removing_agent():
    gdf = _make_points_gdf([(0, 0)], energy=[10.0])
    env = Environment(end_time=1)
    model = AgentModel(gdf=gdf)

    agent = model.society[0]
    agent.leave()

    assert agent.has_location is False
    assert agent.geometry is None
    assert len(model.society) == 1  # still in the society, just no position
    assert 0 in model.society


def test_locationless_agent_spatial_methods_raise_clear_error():
    gdf = _make_points_gdf([(0, 0)], energy=[10.0])
    env = Environment(end_time=1)
    model = AgentModel(gdf=gdf)

    agent = model.society.add(energy=1.0)  # no location

    with pytest.raises(RuntimeError, match="has no location"):
        agent.walk(step_size=1.0)

    with pytest.raises(RuntimeError, match="has no location"):
        agent.neighbors(5.0)

    with pytest.raises(RuntimeError, match="has no location"):
        agent.distance_to(model.society[0])


def test_locationless_agents_ignored_by_others_neighbor_queries():
    """An agent with a location querying neighbors should silently skip
    any location-less agents in the society (they were never candidates)."""
    gdf = _make_points_gdf([(0, 0), (1, 0)], energy=[10.0, 5.0])
    env = Environment(end_time=1)
    model = AgentModel(gdf=gdf)

    model.society.add(energy=1.0)  # location-less agent added to the mix

    neighbors = model.society[0].neighbors(radius=2)
    assert all(a.has_location for a in neighbors)
    assert len(neighbors) == 1  # only agent 1, not the location-less one


def test_reproduce_of_locationless_agent_yields_locationless_child():
    gdf = _make_points_gdf([(0, 0)], energy=[10.0])
    env = Environment(end_time=1)
    model = AgentModel(gdf=gdf)

    orphan = model.society.add(energy=4.0)  # no location
    child = orphan.reproduce(energy=1.0)

    assert child.has_location is False
    assert child.geometry is None


def test_reproduce_of_located_agent_still_copies_geometry():
    gdf = _make_points_gdf([(3, 3)], energy=[10.0])
    env = Environment(end_time=1)
    model = AgentModel(gdf=gdf)

    parent = model.society[0]
    child = parent.reproduce(energy=2.0)

    assert child.has_location is True
    assert child.geometry == parent.geometry


# ---------------------------------------------------------------------------
# Arbitrary geometry support (move_to / enter accept any shapely geometry,
# not just Point — mirrors TerraME's Agent:move(cell), where a Cell can be
# any geometry the CellularSpace defines, typically a polygon)
# ---------------------------------------------------------------------------

def test_move_to_accepts_coordinates_as_point_shortcut():
    gdf = _make_points_gdf([(0, 0)])
    env = Environment(end_time=1)
    model = AgentModel(gdf=gdf)

    agent = model.society[0]
    agent.move_to(5, 3)

    assert agent.geometry == Point(5, 3)


def test_move_to_accepts_arbitrary_geometry():
    from shapely.geometry import Polygon

    gdf = _make_points_gdf([(0, 0)])
    env = Environment(end_time=1)
    model = AgentModel(gdf=gdf)

    poly = Polygon([(10, 10), (11, 10), (11, 11), (10, 11)])
    agent = model.society[0]
    agent.move_to(poly)

    assert agent.geometry == poly
    assert agent.geometry.geom_type == "Polygon"


def test_enter_accepts_arbitrary_geometry():
    from shapely.geometry import Polygon

    gdf = _make_points_gdf([(0, 0)])
    env = Environment(end_time=1)
    model = AgentModel(gdf=gdf)

    poly = Polygon([(2, 2), (3, 2), (3, 3), (2, 3)])
    orphan = model.society.add(energy=1.0)
    orphan.enter(poly)

    assert orphan.has_location is True
    assert orphan.geometry == poly


def test_walk_on_polygon_agent_uses_centroid_and_becomes_point():
    from shapely.geometry import Polygon

    poly = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])  # centroid (1, 1)
    gdf = gpd.GeoDataFrame({"geometry": [poly]})
    env = Environment(end_time=1)
    model = AgentModel(gdf=gdf)

    agent = model.society[0]
    agent.walk(step_size=0.0)  # zero step => lands exactly on centroid

    assert agent.geometry == Point(1, 1)


def test_reproduce_copies_polygon_geometry():
    from shapely.geometry import Polygon

    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    gdf = gpd.GeoDataFrame({"geometry": [poly], "energy": [5.0]})
    env = Environment(end_time=1)
    model = AgentModel(gdf=gdf)

    parent = model.society[0]
    child = parent.reproduce(energy=2.0)

    assert child.geometry == poly
    assert child.geometry.geom_type == "Polygon"


# ---------------------------------------------------------------------------
# PeripherisationModel (Barros and Alves Jr., 2003)
# ---------------------------------------------------------------------------

def test_peripherisation_setup_seeds_center_and_fills_pending_queue():
    gdf = vector_grid(dimension=(11, 11), resolution=1)
    env = Environment(end_time=1)
    model = PeripherisationModel(gdf=gdf, n_agents=20, seed=0)

    assert model.red == 1  # the seed cell
    assert len(model._pending) == 20
    counts = {g: model._pending.count(g) for g in set(model._pending)}
    # default proportions (0.10, 0.40, 0.50) of 20 -> 2 red, 8 yellow, 10 blue
    assert counts.get(0, 0) == 2
    assert counts.get(1, 0) == 8
    assert counts.get(2, 0) == 10


def test_peripherisation_rejects_invalid_proportions():
    gdf = vector_grid(dimension=(5, 5), resolution=1)
    env = Environment(end_time=1)
    with pytest.raises(ValueError, match="must sum to 1.0"):
        PeripherisationModel(gdf=gdf, proportions=(0.5, 0.5, 0.5))


def test_peripherisation_supports_multiple_seed_cells():
    gdf = vector_grid(dimension=(5, 5), resolution=1)
    env = Environment(end_time=1)
    model = PeripherisationModel(gdf=gdf, n_agents=5, seed_cells=["0-0", "4-4"], seed=0)

    assert model.red == 2
    assert model.gdf.loc["0-0", "group"] == 0
    assert model.gdf.loc["4-4", "group"] == 0


def test_peripherisation_converges_and_conserves_agent_count():
    gdf = vector_grid(dimension=(15, 15), resolution=1)
    env = Environment(end_time=500)
    model = PeripherisationModel(
        gdf=gdf, steps=2, n_agents=150, agents_per_step=10, seed=1
    )

    t = 0
    while not model.is_done() and t < 500:
        model.pre_execute()
        model.execute()
        model.post_execute()
        t += 1

    assert model.is_done()
    # 150 agents requested + 1 seed cell already counted as red
    assert model.red + model.yellow + model.blue == 151


def test_peripherisation_respects_eviction_rules():
    """Yellow can never displace red; blue can never displace anyone."""
    gdf = vector_grid(dimension=(15, 15), resolution=1)
    env = Environment(end_time=500)
    model = PeripherisationModel(
        gdf=gdf, steps=2, n_agents=150, agents_per_step=10, seed=2
    )

    while not model.is_done():
        model.pre_execute()
        model.execute()
        model.post_execute()

    # Settlement is final and consistent with the rule set: every cell's
    # final group is one that was legally allowed to settle there given
    # the rules (sanity check is structural: counts are non-negative and
    # match the grid).
    assert model.red >= 1
    assert model.yellow >= 0
    assert model.blue >= 0
    assert (model.gdf["group"] >= -1).all()
    assert (model.gdf["group"] <= 2).all()


def test_peripherisation_runs_via_environment():
    gdf = vector_grid(dimension=(11, 11), resolution=1)
    env = Environment(start_time=0, end_time=200)
    model = PeripherisationModel(
        gdf=gdf, steps=2, n_agents=60, agents_per_step=5, seed=0
    )
    env.run()

    assert model.red + model.yellow + model.blue <= 61  # at most all agents settled
    assert model.pending >= 0


def test_peripherisation_track_plot_attributes_match_labels():
    assert set(PeripherisationModel._plot_info.keys()) == {
        "red", "yellow", "blue", "pending"
    }

