"""
The logo models against TerraME itself.

``goldens/logo_reference.csv`` holds what TerraME 2.0.1 computed running the
models of the ``logo`` package for seeds 1 to 50 (an extract of
``terrame-goldens``, see ``goldens/make_reference.py``). TerraME and NumPy
draw different random numbers, so the runs are compared by distribution: at
each step listed in the reference, the values of ``SEEDS`` runs of the port
and the 50 of TerraME must pass a two-sample Kolmogorov-Smirnov test
(p >= ``P_MIN``), and their means must not differ by more than ``Z_MAX``
standard errors.

The runs use fixed seeds, so the tests are deterministic.

The closest case is ``LifeCycle`` at t = 100 (p about 0.004): the 50
TerraME runs happen to lie above the exact expected population (5405
against 5281 agents, 0.9 standard errors), and the 20 runs of the port
below it. ``test_models.py`` checks the port against that expectation with
400 runs.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from dissmodel.core import Environment, Model
from dissmodel_abm.models import Ants, Disease, LifeCycle, PredatorPrey, Schelling
from dissmodel_abm.models import predator_prey as pp

pytestmark = pytest.mark.golden

GOLDENS = Path(__file__).resolve().parent / "goldens"
SEEDS = range(1, 21)
Z_MAX = 4.0
P_MIN = 0.001


def ks_pvalue(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov test, asymptotic p-value (Numerical Recipes ``kstwo``)."""
    grid = np.union1d(a, b)
    d = np.abs(
        np.searchsorted(np.sort(a), grid, side="right") / len(a)
        - np.searchsorted(np.sort(b), grid, side="right") / len(b)
    ).max()
    en = np.sqrt(len(a) * len(b) / (len(a) + len(b)))
    lam = (en + 0.12 + 0.11 / en) * d
    if lam < 1e-3:
        return 1.0
    k = np.arange(1, 101)
    return float(np.clip(2 * np.sum((-1.0) ** (k - 1) * np.exp(-2 * (k * lam) ** 2)), 0, 1))


@cache
def reference() -> dict[tuple[str, str, int], np.ndarray]:
    values: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    with open(GOLDENS / "logo_reference.csv", newline="") as f:
        for r in csv.DictReader(f):
            values[(r["case"], r["variable"], int(r["step"]))].append(float(r["value"]))
    return {k: np.array(v) for k, v in values.items()}


CHECKPOINTS: dict[str, list[int]] = json.loads(
    (GOLDENS / "logo_reference.json").read_text()
)["checkpoints"]


class _Recorder(Model):
    """Reads the model after each step (it is created after the model, so it runs after it)."""

    def setup(self, read: Callable[[], dict[str, float]], steps: set[int]) -> None:  # type: ignore[override]
        self.read, self.steps, self.rows = read, steps, {}

    def execute(self) -> None:
        t = int(self.env.now())
        if t in self.steps:
            self.rows[t] = self.read()


READERS: dict[str, tuple[type, Callable[[Any], dict[str, float]]]] = {
    "schelling": (Schelling, lambda m: {
        "unhappy": m.unhappy, "unhappy_after": m.percent_unhappy()}),
    "predator_prey": (PredatorPrey, lambda m: {
        "rabbits": m.rabbits, "wolves": m.wolves,
        "pasture": int((m.cs.backend.arrays["state"] == pp.PASTURE).sum())}),
    "life_cycle": (LifeCycle, lambda m: {"agents": m.agents}),
    "ants": (Ants, lambda m: {
        "food": m.food, "bringing": sum(a.state == "bringing" for a in m.society)}),
    "disease": (Disease, lambda m: {
        "susceptible": m.susceptible, "infected": m.infected, "recovered": m.recovered}),
}


@cache
def port_runs(case: str) -> dict[tuple[str, int], np.ndarray]:
    cls, read = READERS[case]
    steps = set(CHECKPOINTS[case])
    out: dict[tuple[str, int], list[float]] = defaultdict(list)
    for seed in SEEDS:
        env = Environment(start_time=1, end_time=max(steps))
        model = cls(seed=seed)
        rec = _Recorder(read=lambda m=model: read(m), steps=steps)
        env.run()
        for t, row in rec.rows.items():
            for var, value in row.items():
                out[(var, t)].append(float(value))
    return {k: np.array(v) for k, v in out.items()}


CASES = sorted(
    (case, var, step) for (case, var, step) in reference() if case in READERS
)


@pytest.mark.parametrize("case,variable,step", CASES, ids=[f"{c}-{v}-t{s}" for c, v, s in CASES])
def test_port_matches_terrame(case: str, variable: str, step: int) -> None:
    tm = reference()[(case, variable, step)]
    ours = port_runs(case)[(variable, step)]
    se = np.sqrt(tm.var(ddof=1) / len(tm) + ours.var(ddof=1) / len(ours))
    diff = ours.mean() - tm.mean()
    p = ks_pvalue(tm, ours)
    msg = (f"{case}.{variable} at t={step}: TerraME {tm.mean():.2f} ± {tm.std():.2f}, "
           f"port {ours.mean():.2f} ± {ours.std():.2f}, KS p = {p:.2g}")
    assert p >= P_MIN, msg
    if se > 0:
        assert abs(diff) <= Z_MAX * se, f"{msg}, z = {diff / se:.1f}"
    else:
        assert diff == 0, msg


def test_reference_covers_every_case() -> None:
    assert {c for c, _, _ in CASES} == set(READERS)
    assert all(len(v) == 50 for v in reference().values())
