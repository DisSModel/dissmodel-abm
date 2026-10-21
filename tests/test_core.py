"""Tests of the core: CellSpace, Agent, Society, Group, Trajectory, AgentModel."""
from __future__ import annotations

import numpy as np
import pytest

from dissmodel.core import Environment
from dissmodel.geo.raster.backend import RasterBackend
from dissmodel_abm import (
    OCCUPANCY_BAND,
    Agent,
    AgentModel,
    CellSpace,
    Group,
    Society,
    Trajectory,
)


@pytest.fixture
def env():
    return Environment(start_time=1, end_time=5)


def rng(seed=0):
    return np.random.default_rng(seed)


# -- CellSpace ---------------------------------------------------------------------


def test_grid_shape_and_bands():
    cs = CellSpace.grid(4, 3, rng=rng(), cover=0, chemical=0.5)
    assert (cs.xdim, cs.ydim) == (4, 3)
    assert cs.backend.shape == (3, 4)
    assert len(cs) == 12
    assert set(cs.backend.arrays) == {"cover", "chemical", OCCUPANCY_BAND}


def test_get_and_band_access():
    cs = CellSpace.grid(3, rng=rng(), cover=0)
    cell = cs.get(2, 1)
    assert (cell.x, cell.y, cell.id) == (2, 1, 5)
    cell.cover = 7
    assert cs.backend.arrays["cover"][1, 2] == 7
    assert cell.cover == 7 and isinstance(cell.cover, int)
    assert cs.get(2, 1) is cell
    assert cs.get(3, 0) is None and cs.get(-1, 0) is None


def test_unknown_band_raises():
    cell = CellSpace.grid(2, rng=rng()).get(0, 0)
    with pytest.raises(AttributeError, match="add_band"):
        cell.covr = 1
    with pytest.raises(AttributeError):
        _ = cell.covr


def test_moore_neighborhood_order_and_border():
    cs = CellSpace.grid(3, rng=rng())
    cs.create_neighborhood()
    center = [(c.x, c.y) for c in cs.get(1, 1).neighborhood]
    assert center == [(0, 0), (1, 0), (2, 0), (0, 1), (2, 1), (0, 2), (1, 2), (2, 2)]
    assert len(cs.get(0, 0).neighborhood) == 3


def test_vonneumann_wrap_and_self():
    cs = CellSpace.grid(4, rng=rng())
    cs.create_neighborhood("vonneumann", wrap=True, include_self=True)
    got = {(c.x, c.y) for c in cs.get(0, 0).neighborhood}
    assert got == {(0, 3), (3, 0), (0, 0), (1, 0), (0, 1)}


def test_wrap_on_small_space_has_no_duplicates():
    cs = CellSpace.grid(2, rng=rng())
    cs.create_neighborhood(wrap=True)
    neigh = cs.get(0, 0).neighborhood
    assert len(neigh) == 3 and len({c.id for c in neigh}) == 3


def test_neighborhood_required():
    cs = CellSpace.grid(2, rng=rng())
    with pytest.raises(RuntimeError, match="create_neighborhood"):
        _ = cs.get(0, 0).neighborhood


def test_nodata_cells_do_not_exist():
    backend = RasterBackend(shape=(2, 3))
    backend.set("mask", np.array([[1, 1, 0], [1, 1, 1]]))
    cs = CellSpace(backend, rng=rng())
    cs.create_neighborhood()
    assert cs.get(2, 0) is None
    assert len(cs) == 5
    assert all((c.x, c.y) != (2, 0) for c in cs.get(1, 1).neighborhood)


def test_derived_band_and_refresh():
    cs = CellSpace.grid(2, rng=rng(), value=1)
    cs.add_derived_band("double", lambda c: c.value * 2)
    assert (cs.backend.arrays["double"] == 2).all()
    cs.get(0, 0).value = 5
    cs.refresh()
    assert cs.backend.arrays["double"][0, 0] == 10


def test_trajectory_select_and_sample():
    cs = CellSpace.grid(3, rng=rng(), v=0)
    cs.get(1, 1).v = 1
    t = Trajectory(cs, select=lambda c: c.v == 1)
    assert len(t) == 1 and t.sample() is cs.get(1, 1)
    cs.get(0, 0).v = 1
    t.filter()
    assert len(t) == 2


# -- Agent and Society ------------------------------------------------------------------


class Counter(Agent):
    energy = 3

    def init(self):
        self.steps = 0

    def execute(self):
        self.steps += 1


def test_society_creates_agents_with_defaults_and_init():
    soc = Society(Counter, quantity=3, rng=rng(), energy=9)
    assert [a.id for a in soc] == [1, 2, 3]
    assert all(a.energy == 9 and a.steps == 0 and a.society is soc for a in soc)


def test_execute_skips_dead_and_newborn():
    class Breeder(Agent):
        def execute(self):
            self.ran = True
            if self.id == 1:
                self.society[2].die()
                self.reproduce()

    soc = Society(Breeder, quantity=3, rng=rng())
    soc.execute()
    ids = [a.id for a in soc]
    assert ids == [1, 3, 4]
    assert getattr(soc[4], "ran", False) is False
    assert soc[3].ran


def test_random_order_is_a_permutation():
    order = []

    class Rec(Agent):
        def execute(self):
            order.append(self.id)

    soc = Society(Rec, quantity=20, rng=rng(1))
    soc.execute(order="random")
    assert sorted(order) == list(range(1, 21)) and order != list(range(1, 21))


def test_reproduce_uses_defaults_and_parent_cell():
    cs = CellSpace.grid(3, rng=rng())
    soc = Society(Counter, quantity=1, rng=rng())
    parent = soc[1]
    parent.energy = 100
    parent.enter(cs.get(1, 1))
    child = parent.reproduce(steps=4)
    assert type(child) is Counter
    assert child.energy == 3
    assert child.steps == 0  # init() runs after the attributes, as in TerraME
    assert child.cell is parent.cell
    assert cs.occupancy[1, 1] == 2


def test_enter_move_leave_keep_occupancy():
    cs = CellSpace.grid(3, rng=rng())
    a = Society(Agent, quantity=1, rng=rng())[1]
    a.enter(cs.get(0, 0))
    assert cs.get(0, 0).agent is a and not cs.get(0, 0).is_empty()
    a.move(cs.get(2, 2))
    assert cs.get(0, 0).is_empty() and cs.get(2, 2).agents == [a]
    assert cs.occupancy.sum() == 1
    a.leave()
    assert a.cell is None and cs.occupancy.sum() == 0
    with pytest.raises(RuntimeError):
        a.move(cs.get(0, 0))


def test_enter_twice_warns_and_moves():
    cs = CellSpace.grid(3, rng=rng())
    a = Society(Agent, quantity=1, rng=rng())[1]
    a.enter(cs.get(0, 0))
    with pytest.warns(UserWarning, match="move"):
        a.enter(cs.get(1, 1))
    assert cs.occupancy.sum() == 1 and a.cell is cs.get(1, 1)


def test_die_removes_from_cell_and_society():
    cs = CellSpace.grid(3, rng=rng())
    soc = Society(Agent, quantity=2, rng=rng())
    a = soc[1]
    a.enter(cs.get(1, 1))
    a.die()
    assert len(soc) == 1 and a not in soc and cs.get(1, 1).is_empty() and not a.is_alive
    a.die()  # no error the second time


def test_walk_to_empty_and_empty_neighbor():
    cs = CellSpace.grid(3, rng=rng())
    cs.create_neighborhood()
    soc = Society(Agent, quantity=9, rng=rng())
    for agent, cell in zip(soc, cs, strict=True):
        agent.enter(cell)
    center = cs.get(1, 1).agent
    assert center.empty_neighbor() is None and center.walk_to_empty() is False
    cs.get(0, 0).agent.die()
    assert center.walk_to_empty() is True and center.cell is cs.get(0, 0)


def test_neighbor_agents():
    cs = CellSpace.grid(3, rng=rng())
    cs.create_neighborhood()
    soc = Society(Agent, quantity=3, rng=rng())
    soc[1].enter(cs.get(1, 1))
    soc[2].enter(cs.get(0, 0))
    soc[3].enter(cs.get(2, 2))
    assert set(a.id for a in soc[1].neighbor_agents()) == {2, 3}


def test_group_filter_sort_sample():
    soc = Society(Counter, quantity=5, rng=rng())
    for a in soc:
        a.energy = a.id
    g = Group(soc, select=lambda a: a.energy > 2, sort=lambda a: -a.energy)
    assert [a.id for a in g] == [5, 4, 3]
    soc[5].die()
    assert len(g) == 3  # a Group does not follow the society until filter()
    g.filter()
    assert [a.id for a in g] == [4, 3]
    assert g.sample() in (soc[3], soc[4])


def test_society_call_and_count():
    soc = Society(Counter, quantity=4, rng=rng())
    soc.call("execute")
    assert all(a.steps == 1 for a in soc)
    assert soc.count(lambda a: a.id % 2 == 0) == 2


def test_to_geodataframe_uses_cell_centers():
    cs = CellSpace.grid(3, rng=rng())
    soc = Society(Counter, quantity=2, rng=rng())
    soc[1].enter(cs.get(2, 1))
    gdf = soc.to_geodataframe(["energy"])
    assert list(gdf["id"]) == [1, 2]
    assert (gdf.geometry[0].x, gdf.geometry[0].y) == (2.5, 1.5)
    assert gdf.geometry[1] is None
    assert list(gdf["energy"]) == [3, 3]


# -- AgentModel ---------------------------------------------------------------------------


class Walkers(AgentModel):
    def setup(self, dim=6, n=10, strategy="random", max=1):
        self.cs = CellSpace.grid(dim)
        self.cs.create_neighborhood()
        self.soc = Society(Counter, quantity=n)
        self.place(self.soc, cellspace=self.cs, strategy=strategy, max=max)

    def execute(self):
        for a in self.soc:
            a.walk_to_empty()


def test_model_shares_its_generator(env):
    m = Walkers(seed=3)
    assert m.soc.rng is m.rng and m.cs.rng is m.rng and m.soc[1].rng is m.rng
    assert m.soc.model is m and m.soc[1].model is m


def test_placement_random_one_per_cell(env):
    m = Walkers(seed=3, n=36)
    assert (m.cs.occupancy == 1).all()


def test_placement_errors_when_too_many(env):
    with pytest.raises(ValueError, match="amount of agents"):
        Walkers(seed=3, n=37)


def test_placement_strategies(env):
    m = Walkers(seed=3, n=8, strategy="uniform")
    assert [(a.cell.x, a.cell.y) for a in m.soc][:3] == [(0, 0), (1, 0), (2, 0)]
    m = Walkers(seed=3, n=72, strategy="spread", max=2)
    assert (m.cs.occupancy == 2).all()
    m = Walkers(seed=3, n=5, strategy="void")
    assert all(a.cell is None for a in m.soc)
    m = Walkers(seed=3, n=30, strategy="random", max=3)
    assert m.cs.occupancy.max() <= 3


def test_same_seed_same_result_and_seed_used():
    def run(seed):
        Environment(start_time=1, end_time=20)
        m = Walkers(seed=seed)
        m.env.run()
        return m.cs.occupancy.tobytes(), m.seed_used

    a, sa = run(11)
    b, sb = run(11)
    c, _ = run(12)
    assert a == b and sa == sb == 11 and a != c
    _, s_none = run(None)
    assert run(s_none)[0] == run(s_none)[0]
