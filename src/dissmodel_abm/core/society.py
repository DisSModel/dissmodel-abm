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
``soc:createSocialNetwork{...}``           ``soc.create_social_network(...)``
``soc:synchronize()``                      ``soc.synchronize()`` (``execute()`` calls it)
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

Messages
--------
Messages sent with a delay wait in the sender's society. ``execute()`` first
calls ``synchronize()``, which subtracts one step from every waiting message
and delivers those that reach zero, and then runs the agents, as TerraME
does for ``Event{action = society}``. A message sent with ``delay=1`` during
step ``t`` is therefore received at the start of step ``t + 1``.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import numpy as np

from . import _context
from .agent import Agent
from .network import Message, SocialNetwork, deliver

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
        self._messages: list[Message] = []
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

    def execute(self, order: str = "insertion", period: float | None = None) -> None:
        """
        Deliver due messages, then run ``execute()`` of each agent alive at
        the start of the step.

        Parameters
        ----------
        order : {"insertion", "random"}
            Creation order (TerraME's default) or a new random permutation
            drawn from the society's generator at every call.
        period : float, optional
            Time subtracted from waiting messages (see :meth:`synchronize`).
            By default the step of the model that created the society, or 1.
        """
        self.synchronize(period)
        agents = list(self._agents.values())
        if order == "random":
            agents = [agents[i] for i in self.rng.permutation(len(agents))]
        elif order != "insertion":
            raise ValueError(f"Unknown order {order!r}; use 'insertion' or 'random'.")
        for agent in agents:
            if agent.is_alive and agent.society is self:
                agent.execute()

    def synchronize(self, delay: float | None = None) -> None:
        """
        Advance waiting messages by ``delay`` and deliver those that are due
        (TerraME ``Society:synchronize``).

        Messages sent while delivering wait for the next call. A message whose
        receiver died meanwhile is dropped, as TerraME warns and ignores it.
        """
        if delay is None:
            delay = getattr(self.model, "_step", 1) if self.model is not None else 1
        if delay <= 0:
            raise ValueError("Argument 'delay' must be positive.")
        pending = self._messages
        self._messages = []
        keep: list[Message] = []
        for msg in pending:
            msg.delay -= delay
            if msg.delay > 0:
                keep.append(msg)
            elif msg.receiver.is_alive:
                msg.delay = 0
                deliver(msg)
        # messages sent during delivery were appended to the new list
        self._messages = keep + self._messages

    @property
    def messages(self) -> list[Message]:
        """Messages waiting to be delivered."""
        return list(self._messages)

    # -- social networks -----------------------------------------------------------------

    def create_social_network(
        self,
        strategy: str | None = None,
        *,
        name: str = "1",
        quantity: int | None = None,
        probability: float | None = None,
        filter: Callable[[A, A], bool] | None = None,  # noqa: A002 - TerraME's name
        include_self: bool = False,
        start: int | None = None,
        inmemory: bool | None = None,
        symmetric: bool = False,
    ) -> None:
        """
        Connect the agents of the society (TerraME ``createSocialNetwork``).

        Parameters
        ----------
        strategy : str, optional
            How to connect. Inferred from the arguments when omitted, as in
            TerraME (``quantity`` → ``"quantity"``, ``probability`` →
            ``"probability"``, ``filter`` → ``"function"``):

            - ``"quantity"``: each agent gets ``quantity`` other agents drawn
              at random.
            - ``"probability"``: each other agent is connected with
              ``probability``.
            - ``"function"``: ``filter(agent, other)`` decides each pair.
            - ``"cell"``: the other agents in the same cell
              (``include_self`` adds the agent itself).
            - ``"neighbor"``: the agents in the neighboring cells.
            - ``"erdos"``: ``quantity`` random undirected links in total.
            - ``"barabasi"``: preferential attachment; the first ``start``
              agents seed it and each later agent makes ``quantity`` links.
            - ``"watts"``: ring of ``quantity`` neighbors per agent, each link
              rewired at random with ``probability``.
            - ``"void"``: empty networks, filled later with ``add``.
        name : str
            Name of the network (TerraME's default is ``"1"``).
        inmemory : bool, optional
            For the first five strategies: ``True`` (default) builds the
            networks now; ``False`` stores a function, so that every
            ``get_social_network`` builds a new network (new random contacts,
            current cell). The graph strategies are always in memory.
        symmetric : bool
            For ``"quantity"`` and ``"probability"`` in memory: when ``a`` is
            connected to ``b``, also connect ``b`` to ``a``.
        """
        if strategy is None:
            candidates = []
            if probability is not None:
                candidates.append("probability")
            if quantity is not None:
                candidates.append("quantity")
            if filter is not None:
                candidates.append("function")
            if len(candidates) != 1:
                raise ValueError(
                    "Could not infer argument 'strategy'; give it explicitly."
                    if candidates
                    else "It was not possible to infer a value for argument 'strategy'."
                )
            strategy = candidates[0]

        agents = list(self._agents.values())
        if not agents:
            raise ValueError("Cannot create a social network in an empty Society.")
        if any(name in a._networks() for a in agents):
            raise ValueError(f"SocialNetwork {name!r} already exists in the Society.")

        graph = strategy in ("erdos", "barabasi", "watts", "void")
        if graph and inmemory is not None and strategy != "void":
            raise ValueError(f"Argument 'inmemory' does not work with strategy {strategy!r}.")
        if inmemory is None:
            inmemory = True
        if symmetric and strategy not in ("quantity", "probability"):
            raise ValueError("Argument 'symmetric' works only with 'quantity' and 'probability'.")
        if symmetric and not inmemory:
            raise ValueError("Argument 'symmetric' needs inmemory=True.")

        rng = self.rng
        builder: Callable[[A], SocialNetwork]

        if strategy == "quantity":
            if quantity is None or quantity < 1:
                raise ValueError("Argument 'quantity' must be a positive integer.")
            if quantity > len(agents) - 1:
                raise ValueError(
                    f"It is not possible to connect such amount of agents ({quantity}). "
                    f"The Society only has {len(agents)} agents."
                )

            def builder(agent: A) -> SocialNetwork:
                pool = list(self._agents.values())
                net = SocialNetwork()
                while len(net) < quantity:
                    other = pool[int(rng.integers(len(pool)))]
                    if other is not agent and other not in net:
                        net.add(other)
                return net

        elif strategy == "probability":
            if probability is None or not 0 < probability <= 1:
                raise ValueError("Argument 'probability' must be a number in (0, 1].")

            def builder(agent: A) -> SocialNetwork:
                net = SocialNetwork()
                for other in list(self._agents.values()):
                    if other is not agent and rng.random() < probability:
                        net.add(other)
                return net

        elif strategy == "function":
            if filter is None:
                raise ValueError("Strategy 'function' needs argument 'filter'.")

            def builder(agent: A) -> SocialNetwork:
                net = SocialNetwork()
                for other in list(self._agents.values()):
                    if filter(agent, other):
                        net.add(other)
                return net

        elif strategy in ("cell", "neighbor"):
            if any(a.cell is None for a in agents):
                raise ValueError("Society has no placement. Call model.place() first.")

            if strategy == "cell":

                def builder(agent: A) -> SocialNetwork:
                    net = SocialNetwork()
                    for other in list(agent.cell.agents):  # type: ignore[union-attr]
                        if other is not agent or include_self:
                            net.add(other)
                    return net

            else:

                def builder(agent: A) -> SocialNetwork:
                    net = SocialNetwork()
                    for neigh in agent.cell.neighborhood:  # type: ignore[union-attr]
                        for other in neigh.agents:
                            if other not in net:
                                net.add(other)
                    return net

        elif strategy == "void":
            for agent in agents:
                agent.add_social_network(SocialNetwork(), name)
            return

        elif strategy == "erdos":
            self._erdos(agents, name, quantity)
            return
        elif strategy == "barabasi":
            self._barabasi(agents, name, quantity, start)
            return
        elif strategy == "watts":
            self._watts(agents, name, quantity, probability)
            return
        else:
            raise ValueError(f"Unknown strategy {strategy!r}.")

        if inmemory:
            for agent in agents:
                agent.add_social_network(builder(agent), name)
        else:
            for agent in agents:
                agent.add_social_network(builder, name)  # type: ignore[arg-type]

        if symmetric:
            for agent in agents:
                for other in agent.connections(name):
                    net = other.get_social_network(name)
                    if net is not None and agent not in net:
                        net.add(agent)

    def _void(self, agents: list[A], name: str) -> list[SocialNetwork]:
        nets = [SocialNetwork() for _ in agents]
        for agent, net in zip(agents, nets, strict=True):
            agent.add_social_network(net, name)
        return nets

    def _erdos(self, agents: list[A], name: str, quantity: int | None) -> None:
        n = len(agents)
        if quantity is None or quantity < 1:
            raise ValueError("Argument 'quantity' must be a positive integer.")
        if quantity > n * (n - 1) // 2:
            raise ValueError("Argument 'quantity' is larger than the number of possible links.")
        nets = dict(zip(agents, self._void(agents, name), strict=True))
        for _ in range(quantity):
            a1 = agents[int(self.rng.integers(n))]
            a2 = a1
            while a2 is a1 or a2 in nets[a1]:
                a2 = agents[int(self.rng.integers(n))]
            nets[a1].add(a2)
            nets[a2].add(a1)

    def _barabasi(
        self, agents: list[A], name: str, quantity: int | None, start: int | None
    ) -> None:
        n = len(agents)
        if start is None or start < 1 or quantity is None or quantity < 1:
            raise ValueError("Strategy 'barabasi' needs positive 'start' and 'quantity'.")
        if start >= n:
            raise ValueError("Argument 'start' should be less than the total of Agents in the Society.")
        if quantity >= start:
            raise ValueError("Argument 'quantity' should be less than 'start'.")
        nets = self._void(agents, name)
        count = start
        for i in range(start, n):
            agent, net = agents[i], nets[i]
            while len(net) < quantity:
                value = int(self.rng.integers(1, count + 1))
                position = -1
                while value > 0:
                    position += 1
                    value -= len(nets[position]) + 1
                candidate = agents[position]
                if candidate not in net:
                    nets[position].add(agent)
                    net.add(candidate)
                    count += 1
            count += quantity + 1

    def _watts(
        self, agents: list[A], name: str, quantity: int | None, probability: float | None
    ) -> None:
        n = len(agents)
        if quantity is None or quantity < 1:
            raise ValueError("Argument 'quantity' must be a positive integer.")
        if probability is None or not 0 <= probability <= 1:
            raise ValueError("Argument 'probability' should be between 0 and 1.")
        nets = dict(zip(agents, self._void(agents, name), strict=True))
        for i, a1 in enumerate(agents):
            net = nets[a1]
            for dist in range(1, quantity + 1):
                a2 = a1 if self.rng.random() < probability else agents[(i + dist) % n]
                while a2 is a1 or a2 in net:
                    a2 = agents[int(self.rng.integers(n))]
                nets[a2].add(a1)
                net.add(a2)

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
    A filtered and optionally sorted subset of a society (TerraME ``Group``).

    As in TerraME, the group is built from its target when it is created and
    does not follow the target by itself afterwards:

    - :meth:`filter` applies ``select`` again to the *current members* only,
      so an agent that left the group does not come back (TerraME
      ``filter``);
    - :meth:`rebuild` builds the group again from the target, as if it were
      created again (TerraME ``rebuild``).
    """

    def __init__(
        self,
        target: Society[A] | Group[A],
        select: Callable[[A], bool] | None = None,
        sort: Callable[[A], Any] | None = None,
        random: bool = False,
    ) -> None:
        if sort is not None and random:
            raise ValueError("A Group cannot use both 'sort' and 'random'.")
        self.target = target
        self.select = select
        self.sort_key = sort
        self.random = random
        self.rng: np.random.Generator = target.rng
        self.agents: list[A] = []
        self.rebuild()

    def rebuild(self) -> None:
        """Build the group again from its target (TerraME ``rebuild``)."""
        self.agents = [a for a in self.target if self.select is None or self.select(a)]
        if self.random:
            self.randomize()
        elif self.sort_key is not None:
            self.sort()

    def filter(self) -> None:
        """
        Apply ``select`` again to the current members (TerraME ``filter``).

        Agents of the target that are not in the group are not looked at, and
        the order of the members is kept. Agents that died are dropped.
        """
        if self.select is None:
            raise ValueError("Cannot filter a Group without a 'select' function.")
        self.agents = [a for a in self.agents if a.is_alive and self.select(a)]

    def sort(self) -> None:
        """Sort the members by the ``sort`` key (TerraME ``sort``)."""
        if self.sort_key is None:
            raise ValueError("Cannot sort a Group without a 'sort' function.")
        self.agents.sort(key=self.sort_key)

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

    Built from the cell space when it is created. As in :class:`Group`,
    :meth:`filter` applies ``select`` again to the current cells only, and
    :meth:`rebuild` builds the trajectory again from the cell space.
    """

    def __init__(
        self,
        target: CellSpace,
        select: Callable[[Cell], bool] | None = None,
        sort: Callable[[Cell], Any] | None = None,
        random: bool = False,
    ) -> None:
        if sort is not None and random:
            raise ValueError("A Trajectory cannot use both 'sort' and 'random'.")
        self.target = target
        self.select = select
        self.sort_key = sort
        self.random = random
        self.rng: np.random.Generator = target.rng
        self.cells: list[Cell] = []
        self.rebuild()

    def rebuild(self) -> None:
        """Build the trajectory again from the cell space (TerraME ``rebuild``)."""
        self.cells = [c for c in self.target if self.select is None or self.select(c)]
        if self.random:
            self.randomize()
        elif self.sort_key is not None:
            self.sort()

    def filter(self) -> None:
        """Apply ``select`` again to the current cells (TerraME ``filter``)."""
        if self.select is None:
            raise ValueError("Cannot filter a Trajectory without a 'select' function.")
        self.cells = [c for c in self.cells if self.select(c)]

    def sort(self) -> None:
        """Sort the cells by the ``sort`` key (TerraME ``sort``)."""
        if self.sort_key is None:
            raise ValueError("Cannot sort a Trajectory without a 'sort' function.")
        self.cells.sort(key=self.sort_key)

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
