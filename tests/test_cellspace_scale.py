"""
Cost of CellSpace on large grids, and agents sharing a backend with a
cellular automaton.

The scale tests count the Cell objects that an operation creates instead of
timing it, so that they do not depend on the machine: work that follows the
number of agents creates a handful of cells, work that follows the size of
the grid creates them all.
"""
from __future__ import annotations

import numpy as np
import pytest

from dissmodel.core import Environment
from dissmodel.geo import RasterCellularAutomaton
from dissmodel.geo.raster.backend import DIRS_VON_NEUMANN, RasterBackend
from dissmodel_abm import Agent, AgentModel, CellSpace, Society, Trajectory

BIG = 2000  # 4 million cells


def rng(seed=0):
    return np.random.default_rng(seed)


class Walker(Agent):
    def execute(self):
        self.walk_to_empty()


class Crowd(AgentModel):
    def setup(self, dim=BIG, n=500, strategy="random", max=1):
        self.cs = CellSpace.grid(dim, cover=np.int8(0))
        self.cs.create_neighborhood()
        self.society = Society(Walker, quantity=n)
        self.place(self.society, cellspace=self.cs, strategy=strategy, max=max)
        self.cs.add_agent_band("who", lambda cell: len(cell.agents), empty=0)
        self.cs.add_agent_band("view", lambda cell: 9, background="cover", dtype=np.int8)

    def execute(self):
        self.society.execute()


# -- scale ------------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", ["random", "uniform", "spread"])
def test_placement_creates_only_the_cells_it_uses(strategy):
    Environment(start_time=1, end_time=1)
    m = Crowd(seed=0, strategy=strategy)
    assert m.cs.n_cells_created <= 500
    assert m.cs.occupancy.sum() == 500


def test_steps_on_a_large_grid_touch_only_cells_near_agents():
    env = Environment(start_time=1, end_time=5)
    m = Crowd(seed=0)
    env.run()
    # each agent creates its cell and at most 8 neighbors per step
    assert m.cs.n_cells_created <= 500 * (1 + 8 * 5)
    assert (m.cs.backend.arrays["who"] > 0).sum() == 500


def test_sampling_does_not_create_cells():
    cs = CellSpace.grid(BIG, rng=rng())
    for _ in range(100):
        cs.sample()
        cs.sample_empty()
    assert cs.n_cells_created <= 200
    assert cs.count_empty() == BIG * BIG


# -- agent bands ----------------------------------------------------------------------------


def test_agent_band_follows_moves_and_deaths():
    cs = CellSpace.grid(4, rng=rng(), cover=np.int8(5))
    cs.add_agent_band("who", lambda cell: cell.agent.id, empty=-1)
    cs.add_agent_band("view", lambda cell: 9, background="cover", dtype=np.int8)
    soc = Society(Agent, quantity=2, rng=rng())
    a, b = soc.agents
    a.enter(cs.get(0, 0))
    b.enter(cs.get(3, 3))
    cs.refresh()
    who = cs.backend.arrays["who"]
    assert who[0, 0] == a.id and who[3, 3] == b.id and (who == -1).sum() == 14
    a.move(cs.get(1, 0))
    b.die()
    cs.get(2, 2).cover = 7
    cs.refresh()
    who, view = cs.backend.arrays["who"], cs.backend.arrays["view"]
    assert who[0, 1] == a.id and (who == -1).sum() == 15
    assert view[0, 1] == 9 and view[2, 2] == 7 and view[0, 0] == 5


def test_agent_band_background_must_exist():
    cs = CellSpace.grid(2, rng=rng())
    with pytest.raises(ValueError, match="background"):
        cs.add_agent_band("view", lambda c: 1, background="cover")


def test_sample_empty_draws_like_a_trajectory():
    def make(seed):
        cs = CellSpace.grid(6, rng=rng(seed))
        soc = Society(Agent, quantity=10, rng=rng(seed))
        for agent, i in zip(soc, (0, 3, 5, 7, 11, 13, 20, 21, 30, 35), strict=True):
            agent.enter(cs.get(i % 6, i // 6))
        return cs

    for seed in range(5):
        via_trajectory = Trajectory(make(seed), select=lambda c: c.is_empty()).sample()
        via_sample = make(seed).sample_empty()
        assert (via_trajectory.x, via_trajectory.y) == (via_sample.x, via_sample.y)


def test_sample_empty_on_a_full_space():
    cs = CellSpace.grid(1, rng=rng())
    Society(Agent, quantity=1, rng=rng())[1].enter(cs.get(0, 0))
    assert cs.sample_empty() is None and cs.count_empty() == 0


# -- sharing a backend with a cellular automaton -----------------------------------------------

FOREST, BURNING, BURNED = 0, 1, 2


class Fire(RasterCellularAutomaton):
    """The forest fire of dissmodel-ca, written against dissmodel's core only."""

    def rule(self, arrays):
        state = arrays["state"]
        burning_near = self.backend.focal_sum_mask(state == BURNING, DIRS_VON_NEUMANN) > 0
        new = np.where(state == BURNING, BURNED, state)
        new = np.where((state == FOREST) & burning_near, BURNING, new)
        return {"state": new.astype(np.int8)}


class Firefighter(Agent):
    def execute(self):
        for neigh in self.cell.neighborhood:
            if neigh.state == BURNING:
                neigh.state = BURNED
                self.model.extinguished += 1
        self.walk_to_empty()


class Brigade(AgentModel):
    def setup(self, backend, n=0):
        self.cs = CellSpace(backend)
        self.cs.create_neighborhood()
        self.society = Society(Firefighter, quantity=n)
        self.place(self.society, cellspace=self.cs)
        self.extinguished = 0

    def execute(self):
        self.society.execute()


def burn(n_firefighters, seed=1):
    backend = RasterBackend(shape=(60, 60))
    state = np.full((60, 60), FOREST, np.int8)
    state[30, 30] = BURNING
    backend.set("state", state)
    env = Environment(start_time=1, end_time=40)
    Fire(backend=backend)
    brigade = Brigade(backend=backend, n=n_firefighters, seed=seed)
    env.run()
    return backend, brigade


def test_agents_and_a_cellular_automaton_share_one_backend():
    backend, _ = burn(0)
    left_alone = int((backend.arrays["state"] == FOREST).sum())
    backend, brigade = burn(600)
    with_brigade = int((backend.arrays["state"] == FOREST).sum())
    assert brigade.extinguished > 0
    assert with_brigade > left_alone
    # the automaton replaces the array at every step; cells still see it
    cell = brigade.cs.get(0, 0)
    backend.arrays["state"][0, 0] = BURNED
    assert cell.state == BURNED
    assert set(backend.arrays) == {"state", "n_agents"}


def test_same_seed_same_fire():
    a, _ = burn(300, seed=4)
    b, _ = burn(300, seed=4)
    assert np.array_equal(a.arrays["state"], b.arrays["state"])
