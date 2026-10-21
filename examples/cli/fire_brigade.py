"""
Firefighters (agents) against the forest fire of dissmodel-ca (a cellular
automaton), on one shared RasterBackend.

The fire is ``dissmodel_ca.models.fire_model_raster.FireModel``, used as it
is: a ``RasterCellularAutomaton`` that rewrites the ``state`` band at every
step. The firefighters are agents in a ``CellSpace`` over the same backend:
each step they put out the burning cells around them (``BURNED``) and walk
to an empty neighbor. Both models write the same arrays, so no data is
copied between them.

Usage
-----
    pip install git+https://github.com/DisSModel/dissmodel-ca
    python examples/cli/fire_brigade.py
    python examples/cli/fire_brigade.py --firefighters 800 --frames   # PNGs in raster_map_frames/
"""
from __future__ import annotations

import argparse

import numpy as np

from dissmodel.core import Environment
from dissmodel.geo.raster.backend import RasterBackend
from dissmodel_abm import Agent, AgentModel, CellSpace, Society

try:
    from dissmodel_ca.models.fire_model_raster import FireModel, FireState
except ImportError as exc:  # pragma: no cover - example only
    raise SystemExit(
        "This example needs dissmodel-ca: pip install git+https://github.com/DisSModel/dissmodel-ca"
    ) from exc

#: Extra code of the ``view`` band: a cell with a firefighter.
FIREFIGHTER = 4
COLORS = {
    int(FireState.FOREST): "#1a9850",
    int(FireState.BURNING): "#d73027",
    int(FireState.BURNED): "#4d4d4d",
    int(FireState.EMPTY): "#f6e8c3",
    FIREFIGHTER: "#2166ac",
}
LABELS = {
    int(FireState.FOREST): "forest",
    int(FireState.BURNING): "burning",
    int(FireState.BURNED): "burned",
    int(FireState.EMPTY): "empty",
    FIREFIGHTER: "firefighter",
}


class Firefighter(Agent):
    def execute(self) -> None:
        for neigh in self.cell.neighborhood:
            if neigh.state == FireState.BURNING:
                neigh.state = int(FireState.BURNED)
                self.model.extinguished += 1
        self.walk_to_empty()


class Brigade(AgentModel):
    def setup(self, backend: RasterBackend, quantity: int = 200) -> None:
        self.cs = CellSpace(backend)  # same arrays as the fire model
        self.cs.create_neighborhood()
        self.society = Society(Firefighter, quantity=quantity)
        self.place(self.society, cellspace=self.cs)
        self.cs.add_agent_band(
            "view", lambda cell: FIREFIGHTER, background="state", dtype=np.int8
        )
        self.extinguished = 0

    def execute(self) -> None:
        self.society.execute()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dim", type=int, default=100)
    parser.add_argument("--firefighters", type=int, default=300)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frames", action="store_true")
    args = parser.parse_args()

    backend = RasterBackend(shape=(args.dim, args.dim))
    env = Environment(start_time=1, end_time=args.steps)

    fire = FireModel(backend=backend, seed=args.seed)
    fire.initialize()  # dissmodel-ca's own setup: 10 % empty, one cell on fire
    brigade = Brigade(backend=backend, quantity=args.firefighters, seed=args.seed)

    if args.frames:
        from dissmodel.visualization import RasterMap

        RasterMap(backend=backend, band="view", title="fire brigade",
                  color_map=COLORS, labels=LABELS, save_frames=True)

    env.run()
    state = backend.arrays["state"]
    counts = {LABELS[int(s)]: int((state == s).sum()) for s in FireState}
    print(f"after {args.steps} steps: {counts}; fires put out: {brigade.extinguished}")


if __name__ == "__main__":
    main()
