"""
TerraME-style cellular space over a dissmodel ``RasterBackend``.

A :class:`CellSpace` is the space agents live in. Cell attributes are the
bands of a :class:`dissmodel.geo.raster.backend.RasterBackend`, so the same
arrays can be read and written per cell (``cell.cover = FOOD``) by agents and
updated in bulk with NumPy by cell rules (evaporation, regrowth), and drawn
with ``dissmodel.visualization.RasterMap``.

Mapping to TerraME
------------------
====================================  =========================================
TerraME                               dissmodel-abm
====================================  =========================================
``CellularSpace{xdim = n}``           ``CellSpace.grid(n)``
``cs:get(x, y)``                      ``cs.get(x, y)``
``cs:sample()``                       ``cs.sample()``
``cs:createNeighborhood{...}``        ``cs.create_neighborhood(...)``
``Cell{state = function(cell) end}``  ``cs.add_agent_band("state", fn)`` or
                                      ``cs.add_derived_band("state", fn)``
``forEachCell(cs, f)``                ``for cell in cs: ...``
``forEachNeighbor(cell, f)``          ``for neigh in cell.neighborhood: ...``
``cell:getNeighborhood():sample()``   ``cell.neighborhood.sample()``
``cell:isEmpty()``                    ``cell.is_empty()``
``cell:getAgent()``                   ``cell.agent``
``cell:getAgents()``                  ``cell.agents``
====================================  =========================================

Coordinates follow TerraME: ``x`` is the column and ``y`` the row, both
starting at 0, with ``y = 0`` the first row of the array.

Cost
----
Agents create the ``Cell`` objects they touch, and nothing else does:
placement, ``sample()``, ``sample_empty()`` and agent bands work on NumPy
indices, so their cost follows the number of agents, not the size of the
grid. Two operations still visit every cell in Python and are meant for
small spaces: iterating the space (``for cell in cs``, ``Trajectory``) and
``add_derived_band``. Rules that apply to every cell (regrowth, diffusion,
evaporation) are best written with NumPy over the whole band, or as a
``dissmodel.geo.RasterCellularAutomaton`` sharing the same backend.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from dissmodel.geo.raster.backend import RasterBackend

from . import _context

if TYPE_CHECKING:  # pragma: no cover
    from .agent import Agent

#: Band maintained by every ``CellSpace``: number of agents in each cell.
OCCUPANCY_BAND = "n_agents"

_MOORE = [(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]


class Cell:
    """
    One cell of a :class:`CellSpace`.

    Reading or writing an attribute that is a band of the backend reads or
    writes that band at this cell (``cell.cover``, ``cell.chemical = 2.0``).
    Bands must exist before they are written (:meth:`CellSpace.add_band`), so
    a misspelled name raises instead of silently creating a new attribute.

    Cells are created once per position and reused, so ``cell_a is cell_b``
    is a valid identity test.
    """

    __slots__ = ("_cs", "_arrays", "x", "y", "id", "agents")

    def __init__(self, cs: CellSpace, x: int, y: int) -> None:
        object.__setattr__(self, "_cs", cs)
        # The backend's dict of bands. Models such as RasterCellularAutomaton
        # replace the arrays in it at every step, so a cell keeps the dict and
        # looks the array up on each access.
        object.__setattr__(self, "_arrays", cs.backend.arrays)
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "id", y * cs.xdim + x)
        object.__setattr__(self, "agents", [])

    # -- band access ---------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        # Only called when normal lookup fails, i.e. for band names.
        try:
            return self._arrays[name].item(self.y, self.x)
        except KeyError:
            if name.startswith("__"):
                raise AttributeError(name) from None
            raise AttributeError(
                f"Cell has no attribute {name!r}: there is no band {name!r} "
                f"in the CellSpace (bands: {sorted(self._arrays)})."
            ) from None
        except ValueError:
            raise TypeError(
                f"Band {name!r} is temporal; read it with cs.backend.get({name!r}, time=...)."
            ) from None

    def __setattr__(self, name: str, value: Any) -> None:
        arr = self._arrays.get(name)
        if arr is None:
            if name in Cell.__slots__:
                raise AttributeError(f"Cell attribute {name!r} is read-only.")
            raise AttributeError(
                f"Cannot set {name!r}: there is no band {name!r}. "
                f"Create it first with cs.add_band({name!r}, ...)."
            )
        if arr.ndim != 2:
            raise TypeError(f"Band {name!r} is temporal and cannot be set per cell.")
        arr[self.y, self.x] = value

    # -- agents ----------------------------------------------------------------

    @property
    def agent(self) -> Agent | None:
        """The first agent in this cell, or ``None`` (TerraME ``getAgent``)."""
        return self.agents[0] if self.agents else None

    def is_empty(self) -> bool:
        """``True`` when no agent is in this cell (TerraME ``isEmpty``)."""
        return not self.agents

    # -- neighborhood -----------------------------------------------------------

    @property
    def neighborhood(self) -> Neighborhood:
        """The neighbors of this cell (TerraME ``getNeighborhood``)."""
        return self._cs.neighborhood_of(self)

    @property
    def cellspace(self) -> CellSpace:
        return self._cs

    def __repr__(self) -> str:
        return f"Cell(x={self.x}, y={self.y})"


class Neighborhood(Sequence[Cell]):
    """
    Ordered, read-only collection of neighbor cells.

    The order is the one TerraME builds: row by row from ``dy = -1`` to
    ``dy = 1`` and, in each row, from ``dx = -1`` to ``dx = 1``. Iterating in
    this order and stopping at the first match reproduces TerraME's
    ``forEachNeighbor`` with ``return false``.
    """

    __slots__ = ("_cells", "_cs")

    def __init__(self, cs: CellSpace, cells: list[Cell]) -> None:
        self._cs = cs
        self._cells = cells

    def __getitem__(self, i):  # type: ignore[override]
        return self._cells[i]

    def __len__(self) -> int:
        return len(self._cells)

    def __iter__(self) -> Iterator[Cell]:
        return iter(self._cells)

    def sample(self) -> Cell:
        """
        A neighbor chosen at random with the model's generator.

        Raises ``ValueError`` when the cell has no neighbors (an isolated cell
        of a masked space), as TerraME fails when sampling an empty set.
        """
        if not self._cells:
            raise ValueError("Cannot sample from an empty neighborhood.")
        return self._cells[int(self._cs.rng.integers(len(self._cells)))]

    def empty(self) -> list[Cell]:
        """The neighbors with no agent."""
        return [c for c in self._cells if c.is_empty()]

    def __repr__(self) -> str:
        return f"Neighborhood({self._cells!r})"


class CellSpace:
    """
    Regular cellular space backed by a ``RasterBackend``.

    Parameters
    ----------
    backend : RasterBackend
        Arrays of shape ``(ydim, xdim)``. Cells outside ``backend.nodata_mask``
        (when the backend defines one) do not exist: ``get`` returns ``None``,
        they are skipped by iteration and never appear as neighbors.
    rng : numpy.random.Generator or int, optional
        Random generator for ``sample()`` and ``Neighborhood.sample()``. By
        default, the generator of the ``AgentModel`` being set up.

    Notes
    -----
    The band :data:`OCCUPANCY_BAND` (``"n_agents"``) is created and kept up
    to date with the number of agents in each cell, so a ``RasterMap`` can
    draw where agents are without extra code.
    """

    def __init__(self, backend: RasterBackend, rng: Any = None) -> None:
        self.backend = backend
        self.ydim, self.xdim = (int(n) for n in backend.shape)
        self.rng = _context.resolve_rng(rng)

        mask = backend.nodata_mask
        self._valid = (
            np.ones(backend.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        )
        # Flat indices (y * xdim + x) of the valid cells, row by row: the
        # order of iteration, placement and sampling.
        self._valid_idx = np.flatnonzero(self._valid)
        self._cells: dict[int, Cell] = {}
        self._occupied: dict[int, Cell] = {}
        self._neighs: dict[int, Neighborhood] = {}
        self._neigh_spec: tuple[str, bool, bool] | None = None
        self._derived: dict[str, Callable[[Cell], Any]] = {}
        self._agent_bands: dict[str, tuple[Callable[[Cell], Any], Any, str | None]] = {}

        backend.set(OCCUPANCY_BAND, np.zeros(backend.shape, dtype=np.int32))

        model = _context.current_model()
        if model is not None:
            model._register_cellspace(self)

    # -- construction ------------------------------------------------------------

    @classmethod
    def grid(
        cls,
        xdim: int,
        ydim: int | None = None,
        rng: Any = None,
        **bands: Any,
    ) -> CellSpace:
        """
        Create a ``xdim`` × ``ydim`` space (square when ``ydim`` is omitted).

        Extra keyword arguments create bands filled with that value, as in
        ``CellSpace.grid(30, state=PASTURE, count=0)``.
        """
        ydim = xdim if ydim is None else ydim
        cs = cls(RasterBackend(shape=(ydim, xdim)), rng=rng)
        for name, value in bands.items():
            cs.add_band(name, value)
        return cs

    def add_band(self, name: str, value: Any = 0, dtype: Any = None) -> np.ndarray:
        """
        Create a band filled with ``value`` and return its array.

        Strings get ``object`` dtype so that longer strings can be written
        later; prefer integer codes when the band will be drawn.
        """
        if dtype is None and isinstance(value, str):
            dtype = object
        arr = np.full(self.backend.shape, value, dtype=dtype)
        self.backend.set(name, arr)
        return self.backend.arrays[name]

    def add_agent_band(
        self,
        name: str,
        fn: Callable[[Cell], Any],
        empty: Any = 0,
        background: str | None = None,
        dtype: Any = np.int32,
    ) -> None:
        """
        Declare a band that shows the agents: ``fn(cell)`` on the cells that
        have agents, ``empty`` (or the band ``background``) elsewhere.

        This is the scalable form of TerraME's
        ``Cell{state = function(cell) ... end}`` when the value depends on the
        agents in the cell. :meth:`refresh` (called by ``AgentModel`` after
        setup and after every step) fills the band with NumPy and then calls
        ``fn`` only for the occupied cells, so its cost follows the number of
        agents, not the size of the grid.

        Parameters
        ----------
        fn : callable
            Value of an occupied cell, such as
            ``lambda cell: WOLF if cell.agent.name == "wolf" else RABBIT``.
        empty : scalar
            Value of the cells without agents.
        background : str, optional
            Name of a band copied to the cells without agents instead of
            ``empty``, to draw agents over a cell attribute (for example the
            land cover under the ants).
        """
        if background is not None and background not in self.backend.arrays:
            raise ValueError(f"There is no band {background!r} to use as background.")
        self._agent_bands[name] = (fn, empty, background)
        self.backend.set(name, np.full(self.backend.shape, empty, dtype=dtype))
        self._refresh_agent_band(name)

    def add_derived_band(
        self,
        name: str,
        fn: Callable[[Cell], Any],
        dtype: Any = np.int32,
        fill: Any = 0,
    ) -> None:
        """
        Declare a band computed from every cell, as TerraME's
        ``Cell{state = function(cell) ... end}``.

        :meth:`refresh` calls ``fn`` for every valid cell in Python, which is
        fine for small spaces (0.5 s per step at 250 × 250) but not for large
        ones (8 s per step at 1000 × 1000). Prefer :meth:`add_agent_band`
        when the value depends on the agents, or compute the band with NumPy
        in the model's ``execute``. Cells outside the mask get ``fill``.
        """
        self._derived[name] = fn
        self.backend.set(name, np.full(self.backend.shape, fill, dtype=dtype))
        self._refresh_band(name)

    def refresh(self) -> None:
        """Recompute the agent bands and the derived bands."""
        for name in self._agent_bands:
            self._refresh_agent_band(name)
        for name in self._derived:
            self._refresh_band(name)

    def _refresh_band(self, name: str) -> None:
        fn = self._derived[name]
        arr = self.backend.arrays[name]
        for cell in self:
            arr[cell.y, cell.x] = fn(cell)

    def _refresh_agent_band(self, name: str) -> None:
        fn, empty, background = self._agent_bands[name]
        arr = self.backend.arrays[name]
        if background is None:
            arr[...] = empty
        else:
            arr[...] = self.backend.arrays[background]
        for cell in self._occupied.values():
            arr[cell.y, cell.x] = fn(cell)

    # -- cells -----------------------------------------------------------------

    def get(self, x: int, y: int) -> Cell | None:
        """The cell at column ``x`` and row ``y``, or ``None`` outside the space."""
        if not (0 <= x < self.xdim and 0 <= y < self.ydim) or not self._valid[y, x]:
            return None
        i = y * self.xdim + x
        cell = self._cells.get(i)
        if cell is None:
            cell = Cell(self, x, y)
            self._cells[i] = cell
        return cell

    def _cell_at(self, i: int) -> Cell:
        """The cell with flat index ``i``, which must be valid."""
        cell = self._cells.get(i)
        if cell is None:
            cell = Cell(self, i % self.xdim, i // self.xdim)
            self._cells[i] = cell
        return cell

    def __iter__(self) -> Iterator[Cell]:
        """
        Valid cells, row by row (TerraME ``forEachCell``).

        Creates a ``Cell`` for every valid cell: fine for small spaces, slow
        for large ones (see the module notes).
        """
        for i in self._valid_idx.tolist():
            yield self._cell_at(i)

    def __len__(self) -> int:
        return len(self._valid_idx)

    def sample(self) -> Cell:
        """A valid cell chosen at random."""
        i = int(self.rng.integers(len(self._valid_idx)))
        return self._cell_at(int(self._valid_idx[i]))

    def _empty_idx(self) -> np.ndarray:
        occ = self.backend.arrays[OCCUPANCY_BAND]
        return np.flatnonzero(self._valid & (occ == 0))

    def sample_empty(self) -> Cell | None:
        """
        A valid cell without agents, chosen at random, or ``None``.

        Draws the same cell as ``Trajectory(cs, select=lambda c:
        c.is_empty()).sample()`` but without visiting every cell in Python.
        """
        idx = self._empty_idx()
        if len(idx) == 0:
            return None
        return self._cell_at(int(idx[int(self.rng.integers(len(idx)))]))

    def count_empty(self) -> int:
        """Number of valid cells without agents."""
        return len(self._empty_idx())

    def empty_cells(self) -> list[Cell]:
        """Valid cells with no agent, row by row (one ``Cell`` per empty cell)."""
        return [self._cell_at(i) for i in self._empty_idx().tolist()]

    @property
    def occupancy(self) -> np.ndarray:
        """Number of agents per cell (the ``n_agents`` band)."""
        return self.backend.arrays[OCCUPANCY_BAND]

    # -- neighborhood -------------------------------------------------------------

    def create_neighborhood(
        self,
        strategy: str = "moore",
        wrap: bool = False,
        include_self: bool = False,
    ) -> None:
        """
        Define the neighborhood of every cell (TerraME ``createNeighborhood``).

        Parameters
        ----------
        strategy : {"moore", "vonneumann"}
            Eight surrounding cells, or the four sharing an edge. Default
            ``"moore"``, as in TerraME.
        wrap : bool
            Connect opposite borders (torus). Default ``False``, as in TerraME.
        include_self : bool
            Include the cell itself (TerraME's ``self = true``). Default ``False``.
        """
        strategy = strategy.lower()
        if strategy not in ("moore", "vonneumann"):
            raise ValueError(f"Unknown strategy {strategy!r}; use 'moore' or 'vonneumann'.")
        self._neigh_spec = (strategy, wrap, include_self)
        self._neighs.clear()

    def neighborhood_of(self, cell: Cell) -> Neighborhood:
        neigh = self._neighs.get(cell.id)
        if neigh is not None:
            return neigh
        if self._neigh_spec is None:
            raise RuntimeError(
                "The CellSpace has no neighborhood. Call cs.create_neighborhood() first."
            )
        strategy, wrap, include_self = self._neigh_spec
        cells: list[Cell] = []
        for dx, dy in _MOORE:
            if dx == 0 and dy == 0:
                if not include_self:
                    continue
            elif strategy == "vonneumann" and dx != 0 and dy != 0:
                continue
            x, y = cell.x + dx, cell.y + dy
            if wrap:
                x %= self.xdim
                y %= self.ydim
            other = self.get(x, y)
            if other is not None and other not in cells:
                cells.append(other)
        neigh = Neighborhood(self, cells)
        self._neighs[cell.id] = neigh
        return neigh

    # -- placement bookkeeping (used by Agent) ------------------------------------

    def _add_agent(self, cell: Cell, agent: Agent) -> None:
        cell.agents.append(agent)
        self._occupied[cell.id] = cell
        self.backend.arrays[OCCUPANCY_BAND][cell.y, cell.x] += 1

    def _remove_agent(self, cell: Cell, agent: Agent) -> None:
        cell.agents.remove(agent)
        if not cell.agents:
            del self._occupied[cell.id]
        self.backend.arrays[OCCUPANCY_BAND][cell.y, cell.x] -= 1

    @property
    def n_cells_created(self) -> int:
        """Number of ``Cell`` objects created so far (to check the cost of a model)."""
        return len(self._cells)

    def __repr__(self) -> str:
        return f"CellSpace(xdim={self.xdim}, ydim={self.ydim}, bands={sorted(self.backend.arrays)})"


__all__ = ["Cell", "CellSpace", "Neighborhood", "OCCUPANCY_BAND"]
