"""
TerraME-style agents.

An agent type is a Python class. Class attributes are the default state of
every new agent (TerraME's ``Agent{energy = 40}``), :meth:`Agent.init` runs
once when the agent is added to a society, and :meth:`Agent.execute` runs
once per step when the society executes::

    class Rabbit(Agent):
        energy = 40

        def execute(self):
            cell = self.cell.neighborhood.sample()
            if cell.is_empty():
                self.move(cell)
            self.energy -= 1
            if self.energy < 0:
                self.die()

Mapping to TerraME
------------------
====================================  =========================================
TerraME                               dissmodel-abm
====================================  =========================================
``Agent{...}``                        ``class MyAgent(Agent): ...``
``init = function(self)``             ``def init(self)``
``execute = function(self)``          ``def execute(self)``
``agent:getCell()``                   ``agent.cell``
``agent:enter(cell)``                 ``agent.enter(cell)``
``agent:leave()``                     ``agent.leave()``
``agent:move(cell)``                  ``agent.move(cell)``
``agent:walk()``                      ``agent.walk()``
``agent:walkIfEmpty()``               ``agent.walk_if_empty()``
``agent:walkToEmpty()``               ``agent.walk_to_empty()``
``agent:emptyNeighbor()``             ``agent.empty_neighbor()``
``agent:die()``                       ``agent.die()``
``agent:reproduce{...}``              ``agent.reproduce(...)``
``Random():number()``                 ``agent.rng.random()``
``agent:addSocialNetwork(sn, name)``  ``agent.add_social_network(sn, name)``
``agent:getSocialNetwork(name)``      ``agent.get_social_network(name)``
``forEachConnection(agent, f)``       ``for other in agent.connections(): ...``
``agent:message{receiver = b}``       ``agent.message(b)``
``on_message``, ``on_<subject>``      ``def on_message(self, msg)``, ``def on_<subject>``
====================================  =========================================

As in TerraME, ``move`` and ``enter`` do not check whether the target cell is
empty: models with at most one agent per cell test ``cell.is_empty()``
themselves.
"""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

from . import _context

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    from .cellspace import Cell
    from .network import SocialNetwork
    from .society import Society


class Agent:
    """
    Base class for agents.

    Attributes
    ----------
    id : int or None
        Identifier given by the society (1, 2, 3, ... in creation order).
    society : Society or None
        The society the agent belongs to.
    cell : Cell or None
        The cell the agent is in, or ``None`` when it has no location.
    """

    id: int | None = None
    society: Society | None = None
    cell: Cell | None = None

    if TYPE_CHECKING:
        # Agents carry the attributes their model gives them (energy, state,
        # age...), as in TerraME; tell type checkers to accept any of them.
        def __getattr__(self, name: str) -> Any: ...
        def __setattr__(self, name: str, value: Any) -> None: ...

    def __init__(self, **attrs: Any) -> None:
        self._alive = True
        for name, value in attrs.items():
            setattr(self, name, value)

    # -- hooks -------------------------------------------------------------------

    def init(self) -> None:
        """Called once, after the agent joins a society (TerraME ``init``)."""

    def execute(self) -> None:
        """Called once per step by ``Society.execute()`` (TerraME ``execute``)."""

    # -- context -------------------------------------------------------------------

    @property
    def rng(self) -> np.random.Generator:
        """The random generator of the agent's society (and thus of its model)."""
        if self.society is not None:
            return self.society.rng
        return _context.resolve_rng()

    @property
    def model(self) -> Any:
        """The ``AgentModel`` whose setup created the agent's society, if any."""
        return self.society.model if self.society is not None else None

    @property
    def is_alive(self) -> bool:
        return self._alive

    @property
    def has_location(self) -> bool:
        return self.cell is not None

    # -- placement -----------------------------------------------------------------

    def enter(self, cell: Cell) -> None:
        """
        Put the agent in ``cell`` (TerraME ``enter``).

        An agent already in a cell is moved, with a warning, as TerraME warns
        and recommends ``move`` in that case.
        """
        if cell is None:
            raise ValueError("Cannot enter a cell that does not exist (got None).")
        if self.cell is not None:
            warnings.warn(
                "Agent is already inside a cell. Use agent.move() instead.",
                stacklevel=2,
            )
            self.leave()
        self.cell = cell
        cell.cellspace._add_agent(cell, self)

    def leave(self) -> None:
        """Remove the agent from its cell, keeping it in its society (TerraME ``leave``)."""
        cell = self.cell
        if cell is None:
            raise RuntimeError("Agent should belong to a cell in order to leave().")
        cell.cellspace._remove_agent(cell, self)
        self.cell = None

    def move(self, cell: Cell) -> None:
        """Move the agent to ``cell`` (TerraME ``move``); it must already have a cell."""
        if self.cell is None:
            raise RuntimeError("Agent should belong to a cell in order to move().")
        if cell is None:
            raise ValueError("Cannot move to a cell that does not exist (got None).")
        self.leave()
        self.enter(cell)

    def _require_cell(self, action: str) -> Cell:
        if self.cell is None:
            raise RuntimeError(f"Agent {self.id!r} has no cell; cannot {action}.")
        return self.cell

    def walk(self) -> None:
        """Move to a random neighbor, empty or not (TerraME ``walk``)."""
        cell = self._require_cell("walk")
        self.move(cell.neighborhood.sample())

    def walk_if_empty(self) -> bool:
        """Pick a random neighbor and move there only if it is empty (TerraME ``walkIfEmpty``)."""
        cell = self._require_cell("walk")
        target = cell.neighborhood.sample()
        if target.is_empty():
            self.move(target)
            return True
        return False

    def empty_neighbor(self) -> Cell | None:
        """A random empty neighbor, or ``None`` (TerraME ``emptyNeighbor``)."""
        cell = self._require_cell("look for an empty neighbor")
        empty = cell.neighborhood.empty()
        if not empty:
            return None
        return empty[int(self.rng.integers(len(empty)))]

    def walk_to_empty(self) -> bool:
        """Move to a random empty neighbor, if there is one (TerraME ``walkToEmpty``)."""
        target = self.empty_neighbor()
        if target is None:
            return False
        self.move(target)
        return True

    def neighbor_agents(self) -> list[Agent]:
        """Agents in the neighboring cells (TerraME ``forEachNeighborAgent``)."""
        cell = self._require_cell("look for neighbor agents")
        return [ag for neigh in cell.neighborhood for ag in neigh.agents]

    # -- life cycle -----------------------------------------------------------------

    def die(self) -> None:
        """
        Remove the agent from its cell and its society (TerraME ``die``).

        A dead agent is skipped by the rest of the current ``Society.execute()``.
        """
        if not self._alive:
            return
        if self.cell is not None:
            self.leave()
        if self.society is not None:
            self.society.remove(self)
        self._alive = False

    def reproduce(self, **attrs: Any) -> Agent:
        """
        Create a new agent of the same type in the same society (TerraME ``reproduce``).

        As in TerraME, the child starts from the type's defaults (class
        attributes and ``init()``), not from a copy of the parent; ``attrs``
        override them. The child enters the parent's cell, if the parent has
        one, even when that leaves two agents in the cell: one-agent-per-cell
        models move the child next, as in ``child.move(empty_cell)``.
        """
        if self.society is None:
            raise RuntimeError("Agent should belong to a society to be able to reproduce.")
        child = self.society.add(agent_type=type(self), **attrs)
        if self.cell is not None:
            child.enter(self.cell)
        return child

    # -- social networks (TerraME addSocialNetwork / getSocialNetwork) ----------------

    def _networks(self) -> dict[str, Any]:
        nets = self.__dict__.get("_social_networks")
        if nets is None:
            nets = {}
            self.__dict__["_social_networks"] = nets
        return nets

    def add_social_network(
        self,
        network: SocialNetwork | Callable[[Agent], SocialNetwork],
        name: str = "1",
    ) -> None:
        """
        Give the agent a social network (TerraME ``addSocialNetwork``).

        ``network`` may be a ``SocialNetwork`` or a function of the agent that
        builds one; a function is called again at every
        :meth:`get_social_network`, as TerraME's ``inmemory = false``.
        """
        self._networks()[name] = network

    def get_social_network(self, name: str = "1") -> SocialNetwork | None:
        """The social network called ``name``, or ``None`` (TerraME ``getSocialNetwork``)."""
        net = self._networks().get(name)
        if callable(net):
            return net(self)
        return net

    def connections(self, name: str = "1") -> list[Agent]:
        """The agents this agent is connected to (TerraME ``forEachConnection``)."""
        net = self.get_social_network(name)
        if net is None:
            raise ValueError(f"Agent does not have a SocialNetwork named {name!r}.")
        return list(net)

    # -- messages (TerraME message / on_message) ---------------------------------------

    def message(
        self,
        receiver: Agent,
        delay: float = 0,
        subject: str | None = None,
        **content: Any,
    ) -> Any:
        """
        Send a message to ``receiver`` (TerraME ``message``).

        With ``delay = 0`` the receiver's handler runs now and its return value
        is returned: ``on_<subject>(msg)`` when ``subject`` is given, otherwise
        ``on_message(msg)``. With ``delay > 0`` the message waits in this
        agent's society and is delivered by ``Society.synchronize()``, which
        ``Society.execute()`` calls before running the agents, after
        ``delay`` steps. Extra keyword arguments travel in ``msg.content`` and
        can be read as attributes (``msg.value``).
        """
        from .network import Message, deliver

        if not isinstance(receiver, Agent):
            raise TypeError(f"receiver must be an Agent, got {type(receiver).__name__}.")
        if delay < 0:
            raise ValueError("Argument 'delay' must be non-negative.")
        msg = Message(sender=self, receiver=receiver, delay=delay, subject=subject, content=content)
        if delay == 0:
            if not receiver.is_alive:
                warnings.warn("Trying to send a message to a dead Agent.", stacklevel=2)
                return None
            return deliver(msg)
        if self.society is None:
            raise RuntimeError("Agent must be within a Society to send messages with delay.")
        self.society._messages.append(msg)
        return None

    def on_message(self, msg: Any) -> Any:
        """Handle a message without subject. Override in agents that receive messages."""
        raise NotImplementedError(
            f"{type(self).__name__} receives messages but does not implement on_message()."
        )

    def __repr__(self) -> str:
        where = f"({self.cell.x}, {self.cell.y})" if self.cell is not None else "no cell"
        return f"{type(self).__name__}(id={self.id}, {where})"


__all__ = ["Agent"]
