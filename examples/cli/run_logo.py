"""
Run one of the models ported from TerraME's ``logo`` package.

Usage
-----
    python examples/cli/run_logo.py schelling
    python examples/cli/run_logo.py predator_prey --steps 100 --seed 7
    python examples/cli/run_logo.py ants --frames      # PNG per step in raster_map_frames/

Steps default to the ``finalTime`` of the original model. The run is
reproducible: the same ``--seed`` gives the same result, and the last line
prints a SHA-256 of the final state (all bands, or the agents' states for
``disease``, which has no space) so that two runs can be compared.
"""
from __future__ import annotations

import argparse
import hashlib

import numpy as np

from dissmodel.core import Environment
from dissmodel_abm.models import Ants, Disease, LifeCycle, PredatorPrey, Schelling, SingleAgent
from dissmodel_abm.models import ants, life_cycle, predator_prey, schelling

MODELS = {
    # name: (class, finalTime of the original, band to draw, colors, labels, summary)
    "single_agent": (SingleAgent, 100, "n_agents", None, None,
                     lambda m: f"agent at {m.society.agents[0].cell}"),
    "schelling": (Schelling, 500, "state", schelling.COLORS, schelling.LABELS,
                  lambda m: f"unhappy = {m.percent_unhappy():.1f} %"),
    "predator_prey": (PredatorPrey, 500, "who", predator_prey.WHO_COLORS, predator_prey.WHO_LABELS,
                      lambda m: f"rabbits = {m.rabbits}, wolves = {m.wolves}"),
    "life_cycle": (LifeCycle, 100, "state", life_cycle.COLORS, life_cycle.LABELS,
                   lambda m: f"agents = {m.agents}"),
    "ants": (Ants, 450, "view", ants.COLORS, ants.LABELS,
             lambda m: f"food = {m.food}"),
    "disease": (Disease, 30, None, None, None,
                lambda m: f"S = {m.susceptible}, I = {m.infected}, R = {m.recovered}"),
}


def state_sha256(model) -> str:
    """SHA-256 of all bands or, for a model without space, of the agents' states."""
    h = hashlib.sha256()
    if not hasattr(model, "cs"):
        for agent in model.society:
            h.update(f"{agent.id}:{agent.state}:{agent.counter}".encode())
        return h.hexdigest()
    for name, arr in sorted(model.cs.backend.arrays.items()):
        h.update(name.encode())
        h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("model", choices=sorted(MODELS))
    parser.add_argument("--steps", type=int, help="number of steps (default: the original finalTime)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frames", action="store_true", help="save one PNG per step with RasterMap")
    args = parser.parse_args()

    cls, final_time, band, colors, labels, summary = MODELS[args.model]
    steps = args.steps or final_time

    # TerraME runs events at t = 1 .. finalTime: same number of steps here.
    env = Environment(start_time=1, end_time=steps)
    model = cls(seed=args.seed)

    if args.frames and band is None:
        parser.error(f"{args.model} has no space to draw")
    if args.frames:
        from dissmodel.visualization import RasterMap

        RasterMap(
            backend=model.cs.backend,
            band=band,
            title=args.model,
            color_map=colors,
            labels=labels,
            save_frames=True,
        )

    env.run()
    print(f"{args.model}: {summary(model)} after {steps} steps (seed {args.seed})")
    print(f"sha256 of the final state: {state_sha256(model)}")


if __name__ == "__main__":
    main()
