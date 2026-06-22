"""
dissmodel_abm/models/peripherisation.py
=========================================
The Peripherisation Model (Barros and Alves Jr., 2003), ported from the
TerraME/Lua implementation (``agent_barro_maio2.lua``).

The model
---------
Population is divided into three economic groups:

- **red** (``group=0``)    — high income
- **yellow** (``group=1``) — middle income
- **blue** (``group=2``)   — low income

Settlement rules (faithful to the Lua ``random_localize`` function):

- **red** and **yellow** may settle on any empty cell, sorted by
  ``dist_centro`` ascending — i.e. they pick the available empty cell
  that is *closest to the existing settlement core*.
- **blue** may only settle in *low-density* areas: empty cells whose
  fraction of occupied Queen-neighbors is ≤ ``low_density_threshold``
  (matching ``rg[3]``, density ≤ 0.4, in the Lua model). Within those
  cells they also prefer the one closest to the core.
- After every tick, blue agents that drifted into the *centro* region
  (local density ≥ ``centro_density_threshold``) are removed and
  returned to the queue (mirroring ``SpatialAgent:execute()`` in Lua).

``dist_centro`` is defined as the Euclidean distance from each cell's
centroid to the centroid of all currently-occupied cells (proxy for
distance to the "centro" region used in the Lua model).

Parameters
----------
proportions : tuple of float
    ``(red, yellow, blue)`` fractions, default ``(0.10, 0.40, 0.50)``.
n_agents : int or None
    Total agents to place (default: fills the grid minus seed cells).
agents_per_step : int
    Agents processed per tick, default 5.
low_density_threshold : float
    Maximum local density for blue-income cells, default 0.4.
centro_density_threshold : float
    Minimum local density to be considered "centro"; blue is banned
    from these cells, default 0.9.
seed_cells : list or None
    Cell ids pre-seeded as red (default: single central cell).
seed : int or None
    Random seed.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from libpysal.weights import Queen

from dissmodel.visualization import track_plot
from dissmodel_abm.core import AgentModel

EMPTY = -1
RED = 0      # high income
YELLOW = 1   # middle income
BLUE = 2     # low income


@track_plot("Red", "tab:red")
@track_plot("Yellow", "tab:olive")
@track_plot("Blue", "tab:blue")
@track_plot("Pending", "gray")
class PeripherisationModel(AgentModel):
    """
    The Peripherisation Model (Barros and Alves Jr., 2003) — Lua variant.

    See module docstring for full description.
    """

    red: int = 0
    yellow: int = 0
    blue: int = 0
    pending: int = 0

    def setup(
        self,
        proportions: tuple[float, float, float] = (0.10, 0.40, 0.50),
        n_agents: Optional[int] = None,
        agents_per_step: int = 5,
        low_density_threshold: float = 0.4,
        centro_density_threshold: float = 0.9,
        seed_cells: Optional[list] = None,
        seed: Optional[int] = None,
    ) -> None:
        if abs(sum(proportions) - 1.0) > 1e-6:
            raise ValueError(f"proportions must sum to 1.0, got {proportions}")

        self.proportions = proportions
        self.agents_per_step = agents_per_step
        self._low_dens = low_density_threshold
        self._centro_dens = centro_density_threshold
        self._rng = np.random.default_rng(seed)

        self.create_neighborhood(strategy=Queen, use_index=True)

        self.gdf["group"] = EMPTY

        # --- fast lookup structures -------------------------------------------
        idx_list = list(self.gdf.index)
        id_to_idx = {k: i for i, k in enumerate(idx_list)}
        self._idx_to_id = idx_list
        cx = self.gdf.geometry.centroid.x
        cy = self.gdf.geometry.centroid.y
        self._cx = np.array([cx[k] for k in idx_list], dtype=float)
        self._cy = np.array([cy[k] for k in idx_list], dtype=float)

        # 1-step neighbor lists (Queen)
        neigh1 = [
            [id_to_idx[n] for n in self.neighs_id(k)] for k in idx_list
        ]

        # 2-step neighborhood: union of neighbors-of-neighbors, excluding self.
        # This approximates the coarser-grid density used in the Lua model
        # (cs2 with res=2), smoothing the density field so that the blue
        # density threshold is not too restrictive.
        neigh2 = []
        for i, ns1 in enumerate(neigh1):
            extended: set[int] = set(ns1)
            for j in ns1:
                extended.update(neigh1[j])
            extended.discard(i)
            neigh2.append(sorted(extended))

        max_k = max(len(ns) for ns in neigh2) if neigh2 else 0
        self._neigh_mat = np.full((len(idx_list), max_k), -1, dtype=np.int32)
        self._neigh_cnt = np.zeros(len(idx_list), dtype=np.int32)
        for i, ns in enumerate(neigh2):
            self._neigh_mat[i, : len(ns)] = ns
            self._neigh_cnt[i] = len(ns)

        # Writable numpy array mirroring gdf["group"] — avoids pandas read-only
        # views inside execute() and gives O(1) indexed access.
        self._groups = np.full(len(idx_list), EMPTY, dtype=np.int32)

        # Incremental group counters (avoid full-grid pandas scans per tick)
        self._count = {RED: 0, YELLOW: 0, BLUE: 0}

        # --- seed red cells ---------------------------------------------------
        if seed_cells is None:
            seed_cells = [self._central_cell_id()]
        for cell_id in seed_cells:
            self.society[cell_id].group = RED
            self._groups[id_to_idx[cell_id]] = RED
            self._count[RED] += 1

        if n_agents is None:
            n_agents = len(self.gdf) - len(seed_cells)

        n_red = round(n_agents * proportions[0])
        n_yellow = round(n_agents * proportions[1])
        n_blue = n_agents - n_red - n_yellow

        # Separate queues per group: process RED → YELLOW → BLUE each tick
        # (matches the sequential order in the Lua model)
        self._q_red: list[int] = [RED] * n_red
        self._q_yellow: list[int] = [YELLOW] * n_yellow
        self._q_blue: list[int] = [BLUE] * n_blue

        self._update_tracked_counts()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def execute(self) -> None:
        total = len(self._q_red) + len(self._q_yellow) + len(self._q_blue)
        if total == 0:
            self._update_tracked_counts()
            return

        groups = self._groups  # writable numpy array, kept in sync with gdf

        # Local density: fraction of Queen-neighbors that are occupied
        density = self._local_density(groups)

        # dist_centro: distance to centroid of all occupied cells
        occ_mask = groups != EMPTY
        if occ_mask.any():
            occ_cx = self._cx[occ_mask].mean()
            occ_cy = self._cy[occ_mask].mean()
        else:
            occ_cx, occ_cy = self._cx.mean(), self._cy.mean()
        dist_centro = np.hypot(self._cx - occ_cx, self._cy - occ_cy)

        # Place 1 agent of each group per "slot" — matches the Lua model
        # (1 high, 1 middle, 1 low per tick).  The total composition is
        # determined by queue sizes (set by proportions in setup), so the
        # final proportions are respected even with equal per-tick rates.
        # This also produces cleaner ring separation: red always picks
        # before yellow each tick, so red stays closer to the core.
        n_per_group = max(1, round(self.agents_per_step / 3))

        for queue, group in (
            (self._q_red,    RED),
            (self._q_yellow, YELLOW),
            (self._q_blue,   BLUE),
        ):
            for _ in range(n_per_group):
                if not queue:
                    break

                queue.pop(0)
                empty = groups == EMPTY

                if group == BLUE:
                    # Low income restricted to low-density peripheral cells
                    cand = empty & (density <= self._low_dens)
                    if not cand.any():
                        queue.insert(0, group)  # no suitable cell this tick
                        break
                else:
                    cand = empty
                    if not cand.any():
                        queue.insert(0, group)
                        break

                cand_idx = np.where(cand)[0]
                cand_dist = dist_centro[cand_idx]

                # Sort ascending: closest to core first (like Lua's sort by
                # dist_centro and then pick cells[1])
                order = np.argsort(cand_dist)
                sorted_cands = cand_idx[order]

                # RandomTrajectory randomness: pick from top-sqrt(n) candidates
                k = max(1, int(np.sqrt(len(sorted_cands))))
                chosen = sorted_cands[self._rng.integers(min(k, len(sorted_cands)))]

                groups[chosen] = group
                self._count[group] += 1

        # Ban blue from "centro" (high-density core) — mirrors
        # SpatialAgent:execute() in the Lua model.  After expulsion the
        # agent immediately tries to re-settle in a low-density cell
        # (same tick), using the same dist_centro snapshot.
        ban = (groups == BLUE) & (density >= self._centro_dens)
        if ban.any():
            ban_idx = np.where(ban)[0]
            groups[ban_idx] = EMPTY
            self._count[BLUE] -= len(ban_idx)

            for _ in range(len(ban_idx)):
                empty = groups == EMPTY
                cand = empty & (density <= self._low_dens)
                if not cand.any():
                    self._q_blue.append(BLUE)  # no space now, retry next tick
                    continue
                cand_idx = np.where(cand)[0]
                order = np.argsort(dist_centro[cand_idx])
                sorted_cands = cand_idx[order]
                k = max(1, int(np.sqrt(len(sorted_cands))))
                chosen = sorted_cands[self._rng.integers(min(k, len(sorted_cands)))]
                groups[chosen] = BLUE
                self._count[BLUE] += 1

        # Sync _groups → gdf["group"] once per tick (batch update is much
        # faster than calling society[cell_id].group = x for every agent)
        self.gdf["group"] = groups

        self._update_tracked_counts()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _central_cell_id(self) -> str:
        gdf = self.gdf
        overall = gdf.geometry.union_all().centroid
        dists = gdf.geometry.centroid.distance(overall)
        return gdf.index[dists.argmin()]

    def _local_density(self, groups: np.ndarray) -> np.ndarray:
        """Vectorised: fraction of Queen-neighbors occupied for each cell."""
        # Append a padding 0 so index -1 (unused slots in neigh_mat) → 0
        occ = np.empty(len(groups) + 1, dtype=np.float32)
        occ[:-1] = (groups != EMPTY).astype(np.float32)
        occ[-1] = 0.0
        neigh_occ = occ[self._neigh_mat]          # (n, max_k)
        valid = (self._neigh_mat >= 0).astype(np.float32)  # (n, max_k)
        density = (neigh_occ * valid).sum(axis=1) / np.maximum(self._neigh_cnt, 1)
        return density

    def _update_tracked_counts(self) -> None:
        self.red = self._count[RED]
        self.yellow = self._count[YELLOW]
        self.blue = self._count[BLUE]
        self.pending = len(self._q_red) + len(self._q_yellow) + len(self._q_blue)

    # ------------------------------------------------------------------
    # Convenience metrics
    # ------------------------------------------------------------------

    def occupied_fraction(self) -> float:
        n = len(self.gdf)
        return 0.0 if n == 0 else (self.red + self.yellow + self.blue) / n

    def is_done(self) -> bool:
        return self.pending == 0


__all__ = ["PeripherisationModel", "EMPTY", "RED", "YELLOW", "BLUE"]
