"""
TerraME-style societies and groups.

Mapping to TerraME
------------------
=========================================  ========================================
TerraME                                    dissmodel-abm
=========================================  ========================================
``Society{instance = ag, quantity = n}``   ``Society(MyAgent, quantity=n)``
``#soc``                                   ``len(soc)``
``soc:add{...}``                           ``soc.add(...)``
``soc:remove(agent)``                      ``soc.remove(agent)``
``soc:sample()``                           ``soc.sample()``
``Event{action = soc}``                    ``soc.execute()`` inside the model's ``execute``
``forEachAgent(soc, f)``                   ``for agent in soc: ...``
``soc:myFunction()`` (call on all agents)  ``soc.call("my_function")``
``Group{target = soc, select = f}``        ``Group(soc, select=f)``
``group:filter()``                         ``group.filter()``
``Trajectory{target = cs, select = f}``    ``Trajectory(cs, select=f)``
=========================================  ========================================

Execution order
---------------
``Society.execute()`` runs the agents alive at the start of the step, in
creation order (TerraME's default) or in a random order
(``order="random"``). An agent that dies during the step is skipped, and an
agent born during the step acts only from the next step.

TerraME's loop differs in one corner case: it runs as many iterations as
there were agents at the start, so when agents die, the same number of
newborns at the end of the list may still act in the step they were born.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import numpy as np

from . import _context
from .agent import Agent

if TYPE_CHECKING:  # pragma: no cover
    import geopandas as gpd

    from .cellspace import Cell, CellSpace

A = TypeVar("A", bound=Agent)


class Society(Generic[A]):
    """
    A collection of agents of one type.

    Parameters
    ----------
    agent_type : type[Agent]
        Class used for the initial agents and for ``add()`` without
        ``agent_type``.
    quantity : int
        Number of agents created now. Each gets ``attrs`` and then runs
        ``init()``.
    rng : numpy.random.Generator or int, optional
        Random generator. By default, the generator of the ``AgentModel``
        being set up, so one ``seed`` controls the whole run.
    **attrs
        Attributes given to each initial agent.
    """

    def __init__(
        self,
        agent_type: type[A] = Agent,  # type: ignore[assignment]
        quantity: int = 0,
        rng: Any = None,
        **attrs: Any,
    ) -> None:
        if not (isinstance(agent_type, type) and issubclass(agent_type, Agent)):
            raise TypeError(f"agent_type must be a subclass of Agent, got {agent_type!r}.")
        self.agent_type = agent_type
        self.rng: np.random.Generator = _context.resolve_rng(rng)
        self.model = _context.current_model()
        self._agents: dict[int, A] = {}
        self._next_id = 1
        for _ in range(int(quantity)):
            self.add(**attrs)

    # -- membership --------------------------------------------------------------

    def add(self, agent_type: type[A] | None = None, **attrs: Any) -> A:
        """Create an agent, add it, run its ``init()`` and return it."""
        cls = agent_type or self.agent_type
        agent = cls()
        agent.id = self._next_id
        agent.society = self
        self._next_id += 1
        self._agents[agent.id] = agent
        for name, value in attrs.items():
            setattr(agent, name, value)
        agent.init()
        return agent

    def remove(self, agent: A | Callable[[A], bool]) -> None:
        """
        Remove an agent, or every agent for which a predicate is true.

        Removing does not take the agent out of its cell; ``agent.die()``
        does both and is what models normally call.
        """
        if isinstance(agent, Agent):
            if self._agents.pop(agent.id, None) is None:  # type: ignore[arg-type]
                raise ValueError(f"Could not remove the agent (id = {agent.id!r}).")
            return
        if callable(agent):
            for ag in [a for a in self._agents.values() if agent(a)]:
                self._agents.pop(ag.id)  # type: ignore[arg-type]
            return
        raise TypeError(f"Expected an Agent or a function, got {type(agent).__name__}.")

    def __len__(self) -> int:
        return len(self._agents)

    def __iter__(self) -> Iterator[A]:
        """Agents in creation order; a copy, so agents may die or reproduce meanwhile."""
        return iter(list(self._agents.values()))

    def __contains__(self, agent: object) -> bool:
        return isinstance(agent, Agent) and self._agents.get(agent.id) is agent  # type: ignore[arg-type]

    def __getitem__(self, agent_id: int) -> A:
        return self._agents[agent_id]

    @property
    def agents(self) -> list[A]:
        return list(self._agents.values())

    # -- queries -------------------------------------------------------------------

    def sample(self) -> A | None:
        """An agent chosen at random, or ``None`` if the society is empty."""
        if not self._agents:
            return None
        agents = list(self._agents.values())
        return agents[int(self.rng.integers(len(agents)))]

    def select(self, predicate: Callable[[A], bool]) -> list[A]:
        return [a for a in self._agents.values() if predicate(a)]

    def count(self, predicate: Callable[[A], bool] | None = None) -> int:
        if predicate is None:
            return len(self._agents)
        return sum(1 for a in self._agents.values() if predicate(a))

    # -- execution --------------------------------------------------------------------

    def execute(self, order: str = "insertion") -> None:
        """
        Run ``execute()`` of each agent alive at the start of the step.

        Parameters
        ----------
        order : {"insertion", "random"}
            Creation order (TerraME's default) or a new random permutation
            drawn from the society's generator at every call.
        """
        agents = list(self._agents.values())
        if order == "random":
            agents = [agents[i] for i in self.rng.permutation(len(agents))]
        elif order != "insertion":
            raise ValueError(f"Unknown order {order!r}; use 'insertion' or 'random'.")
        for agent in agents:
            if agent.is_alive and agent.society is self:
                agent.execute()

    def call(self, method: str, *args: Any, **kwargs: Any) -> list[Any]:
        """Call ``method`` on every agent, in creation order, and return the results."""
        return [getattr(a, method)(*args, **kwargs) for a in list(self._agents.values())]

    # -- export -------------------------------------------------------------------------

    def to_geodataframe(self, attrs: Iterable[str] = ()) -> gpd.GeoDataFrame:
        """
        One point per agent, at the center of its cell.

        Uses the backend's affine ``transform`` and ``crs`` when present;
        otherwise ``(x + 0.5, y + 0.5)`` in cell units. Agents without a cell
        get an empty geometry.
        """
        import geopandas as gpd
        from shapely.geometry import Point

        attrs = list(attrs)
        rows = []
        crs = None
        for a in self._agents.values():
            geom = None
            if a.cell is not None:
                backend = a.cell.cellspace.backend
                crs = crs or backend.crs
                col, row = a.cell.x + 0.5, a.cell.y + 0.5
                if backend.transform is not None:
                    px, py = backend.transform * (col, row)
                else:
                    px, py = col, row
                geom = Point(px, py)
            rec = {"id": a.id, "type": type(a).__name__, "geometry": geom}
            rec.update({k: getattr(a, k, None) for k in attrs})
            rows.append(rec)
        return gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)

    def __repr__(self) -> str:
        return f"Society({self.agent_type.__name__}, n={len(self)})"


class Group(Generic[A]):
    """
    A filtered and optionally sorted view of a society (TerraME ``Group``).

    The view is computed when the group is created and again on each
    :meth:`filter` call, as in TerraME; it does not follow the society by itself.
    """

    def __init__(
        self,
        target: Society[A] | Group[A],
        select: Callable[[A], bool] | None = None,
        sort: Callable[[A], Any] | None = None,
        random: bool = False,
    ) -> None:
        self.target = target
        self.select = select
        self.sort_key = sort
        self.random = random
        self.rng: np.random.Generator = target.rng
        self.agents: list[A] = []
        self.filter()

    def filter(self) -> None:
        """Rebuild the group from its target (TerraME ``filter``/``rebuild``)."""
        agents = [a for a in self.target if self.select is None or self.select(a)]
        if self.sort_key is not None:
            agents.sort(key=self.sort_key)
        self.agents = agents
        if self.random:
            self.randomize()

    def randomize(self) -> None:
        self.agents = [self.agents[i] for i in self.rng.permutation(len(self.agents))]

    def sample(self) -> A | None:
        if not self.agents:
            return None
        return self.agents[int(self.rng.integers(len(self.agents)))]

    def execute(self) -> None:
        for agent in list(self.agents):
            if agent.is_alive:
                agent.execute()

    def __len__(self) -> int:
        return len(self.agents)

    def __iter__(self) -> Iterator[A]:
        return iter(list(self.agents))

    def __repr__(self) -> str:
        return f"Group(n={len(self)})"


class Trajectory:
    """
    A filtered and optionally sorted list of cells (TerraME ``Trajectory``).

    Rebuilt on creation and on each :meth:`filter` call.
    """

    def __init__(
        self,
        target: CellSpace,
        select: Callable[[Cell], bool] | None = None,
        sort: Callable[[Cell], Any] | None = None,
        random: bool = False,
    ) -> None:
        self.target = target
        self.select = select
        self.sort_key = sort
        self.random = random
        self.rng: np.random.Generator = target.rng
        self.cells: list[Cell] = []
        self.filter()

    def filter(self) -> None:
        cells = [c for c in self.target if self.select is None or self.select(c)]
        if self.sort_key is not None:
            cells.sort(key=self.sort_key)
        self.cells = cells
        if self.random:
            self.randomize()

    def randomize(self) -> None:
        self.cells = [self.cells[i] for i in self.rng.permutation(len(self.cells))]

    def sample(self) -> Cell | None:
        if not self.cells:
            return None
        return self.cells[int(self.rng.integers(len(self.cells)))]

    def __len__(self) -> int:
        return len(self.cells)

    def __iter__(self) -> Iterator[Cell]:
        return iter(list(self.cells))

    def __repr__(self) -> str:
        return f"Trajectory(n={len(self)})"


__all__ = ["Society", "Group", "Trajectory"]
