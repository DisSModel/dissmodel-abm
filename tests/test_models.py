"""
Behavior tests of the models ported from TerraME's ``logo`` package.

The random generators of TerraME and NumPy differ, so runs cannot be
compared value by value. These tests check the rules (invariants) and,
where ``logo`` keeps a reference chart (``log/*-chart-1.bmp``), that the
port lands in the same range over several seeds.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from dissmodel.core import Environment
from dissmodel_abm.models import Ants, Disease, LifeCycle, PredatorPrey, Schelling, SingleAgent
from dissmodel_abm.models import ants as ants_mod
from dissmodel_abm.models import predator_prey as pp_mod
from dissmodel_abm.models import schelling as sch_mod


def run(cls, steps, seed=0, **params):
    env = Environment(start_time=1, end_time=steps)
    model = cls(seed=seed, **params)
    env.run()
    return model


def digest(model) -> str:
    h = hashlib.sha256()
    if not hasattr(model, "cs"):  # models without space: hash the agents' state
        for agent in model.society:
            plain = {
                k: v
                for k, v in vars(agent).items()
                if not k.startswith("_") and isinstance(v, (int, float, str, bool))
            }
            h.update(repr((agent.id, sorted(plain.items()))).encode())
        return h.hexdigest()
    arrays = model.cs.backend.arrays
    for name in sorted(arrays):
        h.update(name.encode())
        h.update(np.ascontiguousarray(arrays[name]).tobytes())
    return h.hexdigest()


# -- reproducibility ---------------------------------------------------------------


@pytest.mark.parametrize(
    "cls, steps, params",
    [
        (SingleAgent, 30, {"quantity": 5}),
        (Schelling, 50, {"dim": 15}),
        (PredatorPrey, 15, {"dim": 15}),
        (LifeCycle, 40, {}),
        (Ants, 40, {"dimension": 30}),
        (Disease, 15, {"quantity": 300}),
    ],
)
def test_same_seed_same_state(cls, steps, params):
    a = digest(run(cls, steps, seed=5, **params))
    b = digest(run(cls, steps, seed=5, **params))
    c = digest(run(cls, steps, seed=6, **params))
    assert a == b
    assert a != c


# -- SingleAgent -------------------------------------------------------------------------


def test_single_agent_moves_one_cell_at_a_time():
    env = Environment(start_time=1, end_time=1)
    m = SingleAgent(seed=1)
    agent = m.society.agents[0]
    for _ in range(30):
        before = agent.cell
        m.execute()
        assert max(abs(agent.cell.x - before.x), abs(agent.cell.y - before.y)) == 1
    assert m.cs.occupancy.sum() == 1
    del env


# -- Schelling -------------------------------------------------------------------------------


def test_schelling_setup_follows_the_original():
    m = run(Schelling, 1, seed=0)
    assert len(m.society) == int(25 * 25 * 0.95)  # floor(dim² · (1 − 0.05))
    assert m.cs.occupancy.max() == 1
    assert len(m.cs.get(0, 0).neighborhood) == 8  # wrapped borders
    states = m.cs.backend.arrays["state"]
    assert (states == sch_mod.FREE).sum() == 625 - len(m.society)


def test_schelling_moves_at_most_one_agent_per_step():
    env = Environment(start_time=1, end_time=1)
    m = Schelling(seed=2)
    for _ in range(20):
        before = {a.id: a.cell for a in m.society}
        m.execute()
        moved = sum(1 for a in m.society if a.cell is not before[a.id])
        assert moved <= 1
    del env


def test_schelling_initial_unhappiness_matches_reference_chart():
    # logo/log/Schelling-chart-1.bmp starts at about 18.4 % unhappy
    values = [run(Schelling, 1, seed=s).percent_unhappy() for s in range(10)]
    assert 15 < np.median(values) < 22


def test_schelling_converges():
    env = Environment(start_time=1, end_time=500)
    m = Schelling(seed=0, dim=15, free_space=0.2)
    start = m.percent_unhappy()
    env.run()
    assert start > 20
    # the group of unhappy agents empties; as in TerraME, agents that became
    # unhappy after leaving it are not moved again
    assert m.unhappy == 0 and len(m.unhappy_agents) == 0
    assert m.percent_unhappy() < start / 2


# -- PredatorPrey --------------------------------------------------------------------------------


def test_predator_prey_invariants():
    m = run(PredatorPrey, 10, seed=0)
    assert m.cs.occupancy.max() == 1
    who = m.cs.backend.arrays["who"]
    assert (who == pp_mod.RABBIT).sum() == m.rabbits
    assert (who == pp_mod.WOLF).sum() == m.wolves
    count = m.cs.backend.arrays["count"]
    state = m.cs.backend.arrays["state"]
    assert count[state == pp_mod.PASTURE].max() == 0
    assert count.max() < 4


def test_predator_prey_matches_reference_chart():
    # logo/log/PredatorPrey-chart-1.bmp: at t = 20, about 858 rabbits and 17 wolves
    runs = [run(PredatorPrey, 20, seed=s) for s in range(5)]
    assert 800 < np.median([m.rabbits for m in runs]) < 900
    assert 10 <= np.median([m.wolves for m in runs]) <= 20


# -- LifeCycle ---------------------------------------------------------------------------------------


def test_life_cycle_no_births_before_age_15():
    m = run(LifeCycle, 14, seed=0)
    assert m.agents <= 10
    assert all(a.age == 14 for a in m.society)


def test_life_cycle_matches_reference_chart_before_the_cap():
    # logo/log/LifeCycle-chart-1.bmp: about 91 agents at t = 40 (see the module
    # docstring about the rule change after that image)
    values = [run(LifeCycle, 40, seed=s).agents for s in range(10)]
    assert 70 < np.median(values) < 115


def life_cycle_expected(steps, quantity=10):
    """Exact expected population: ages advance, births at 15-30 (p = 0.3), deaths from 20 (p = 0.05)."""
    pop = np.zeros(steps + 1)
    pop[0] = quantity
    for _ in range(steps):
        new = np.zeros_like(pop)
        ages = np.arange(1, steps + 1)
        survive = np.where(ages >= 20, 0.95, 1.0)
        new[1:] = pop[:-1] * survive
        new[0] = (pop[:-1] * ((ages >= 15) & (ages <= 30)) * 0.3).sum()
        pop = new
    return pop.sum()


def test_life_cycle_mean_matches_the_exact_expectation():
    # The population is a branching process whose mean can be computed exactly.
    # 400 runs: standard error about 0.8 agents at t = 40
    values = np.array([run(LifeCycle, 40, seed=s).agents for s in range(400)])
    se = values.std(ddof=1) / np.sqrt(len(values))
    assert abs(values.mean() - life_cycle_expected(40)) < 4 * se


def test_life_cycle_child_can_share_a_cell():
    m = run(LifeCycle, 80, seed=0)
    assert len(m.society) > len(m.cs)
    assert m.cs.occupancy.sum() == len(m.society)


# -- Ants ----------------------------------------------------------------------------------------------


def test_ants_setup():
    env = Environment(start_time=1, end_time=1)
    m = Ants(seed=0)
    cover = m.cs.backend.arrays["cover"]
    assert (cover == ants_mod.NEST).sum() == 9
    assert (cover == ants_mod.FOOD).sum() == 4 * 25
    starts = {(c.x, c.y) for c in m.start_cells}
    assert {(a.cell.x, a.cell.y) for a in m.society} <= starts
    assert len(m.society) == 10
    del env


def test_ants_collect_food_and_chemical_evaporates():
    m = run(Ants, 60, seed=0)
    assert m.food < 100
    chem = m.cs.backend.arrays["chemical"]
    cover = m.cs.backend.arrays["cover"]
    assert not ((chem <= 0) & (cover == ants_mod.CHEMICAL)).any()


def test_ants_first_collection_time_matches_reference_chart():
    # logo/log/Ants-chart-1.bmp: first unit of food taken at t = 8, which is
    # also the earliest step seen in the port over many seeds.
    firsts = []
    for s in range(10):
        run(Ants, 30, seed=s)
        data = Ants._plot_info["food"]["data"]
        firsts.append(next(i + 1 for i, v in enumerate(data) if v < 100))
    assert min(firsts) == 8


# -- Disease ------------------------------------------------------------------------------------------


def test_disease_counters_match_states():
    m = run(Disease, 12, seed=1, quantity=400)
    states = [a.state for a in m.society]
    assert m.susceptible == states.count("susceptible")
    assert m.infected == states.count("infected")
    assert m.recovered == states.count("recovered")
    assert m.susceptible + m.infected + m.recovered == 400


def test_disease_each_case_sends_duration_plus_two_rounds():
    sent: dict[int, int] = {}

    class Counting(Disease):
        pass

    env = Environment(start_time=1, end_time=30)
    m = Counting(seed=0, quantity=300)
    original = type(m.society.agents[0]).message

    def counting_message(self, receiver, **kw):
        sent[self.id] = sent.get(self.id, 0) + 1
        return original(self, receiver, **kw)

    type(m.society.agents[0]).message = counting_message
    try:
        env.run()
    finally:
        type(m.society.agents[0]).message = original
    recovered = [a.id for a in m.society if a.state == "recovered"]
    assert recovered
    assert all(sent[i] == (2 + 2) * 6 for i in recovered)


def test_disease_peak_matches_reference_chart():
    # logo/log/Disease-chart-1.bmp: peak near 650 infected at step 10 (see the
    # module docstring for the final number of susceptible agents)
    peaks, steps = [], []
    for s in range(8):
        run(Disease, 20, seed=s)
        data = Disease._plot_info["infected"]["data"]
        peaks.append(max(data))
        steps.append(int(np.argmax(data)) + 1)
    assert 600 < np.median(peaks) < 720
    assert 9 <= np.median(steps) <= 11
