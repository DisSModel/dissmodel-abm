"""
Social networks and messages between agents, as in TerraME.

Mapping to TerraME
------------------
==============================================  ===============================================
TerraME                                         dissmodel-abm
==============================================  ===============================================
``SocialNetwork()``                             ``SocialNetwork()``
``sn:add(agent, weight)``                       ``sn.add(agent, weight)``
``sn:isConnection(agent)``                      ``sn.is_connection(agent)``
``sn:getWeight(agent)``, ``setWeight``          ``sn.get_weight(agent)``, ``set_weight``
``sn:sample()``                                 ``sn.sample()``
``soc:createSocialNetwork{quantity = 6}``       ``soc.create_social_network(quantity=6)``
``agent:addSocialNetwork(sn, name)``            ``agent.add_social_network(sn, name)``
``agent:getSocialNetwork(name)``                ``agent.get_social_network(name)``
``forEachConnection(agent, f)``                 ``for other in agent.connections(): ...``
``agent:message{receiver = b, delay = 1}``      ``agent.message(b, delay=1)``
``on_message = function(self, msg)``            ``def on_message(self, msg)``
``on_<subject>``, ``message{subject = ...}``    ``def on_<subject>``, ``message(b, subject=...)``
==============================================  ===============================================

A network created with ``inmemory=False`` is a function: each call to
``get_social_network`` builds a new one. TerraME's ``Disease`` model relies
on this to draw new contacts at every step.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from .agent import Agent


class SocialNetwork:
    """
    Weighted, directed connections from one agent to others.

    Connections are kept in the order they were added; iterating yields the
    connected agents. Agents are keyed by identity, not by ``id``, so a
    network may join agents of different societies (whose ids start at 1
    each). TerraME iterates by id as a string ("1", "10", "2", ...); the order
    only changes the order in which messages are queued, not which are sent.
    """

    __slots__ = ("_conn", "_rng")

    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self._conn: dict[Agent, float] = {}
        self._rng = rng

    def add(self, agent: Agent, weight: float = 1.0) -> None:
        if agent in self._conn:
            raise ValueError(f"Agent {agent.id!r} already belongs to the SocialNetwork.")
        self._conn[agent] = weight

    def remove(self, agent: Agent) -> None:
        if agent not in self._conn:
            raise ValueError(f"Agent {agent.id!r} is not in the SocialNetwork.")
        del self._conn[agent]

    def clear(self) -> None:
        self._conn.clear()

    def is_connection(self, agent: Agent) -> bool:
        return agent in self._conn

    def get_weight(self, agent: Agent) -> float:
        return self._conn[agent]

    def set_weight(self, agent: Agent, weight: float) -> None:
        if agent not in self._conn:
            raise ValueError(f"Agent {agent.id!r} is not in the SocialNetwork.")
        self._conn[agent] = weight

    def is_empty(self) -> bool:
        return not self._conn

    def sample(self) -> Agent:
        """A connected agent chosen at random."""
        if not self._conn:
            raise ValueError("Cannot sample from an empty SocialNetwork.")
        agents = list(self._conn)
        rng = self._rng if self._rng is not None else agents[0].rng
        return agents[int(rng.integers(len(agents)))]

    def items(self) -> list[tuple[Agent, float]]:
        """``(agent, weight)`` pairs, in the order they were added."""
        return list(self._conn.items())

    def __iter__(self) -> Iterator[Agent]:
        return iter(list(self._conn))

    def __len__(self) -> int:
        return len(self._conn)

    def __contains__(self, agent: object) -> bool:
        return agent in self._conn

    def __repr__(self) -> str:
        return f"SocialNetwork(ids={[a.id for a in self._conn]})"


@dataclass
class Message:
    """
    A message between agents.

    ``content`` holds the extra keyword arguments given to ``agent.message``.
    ``delay`` counts down while the message waits in the sender's society.
    """

    sender: Agent
    receiver: Agent
    delay: float = 0
    subject: str | None = None
    content: dict[str, Any] = field(default_factory=dict)

    def __getattr__(self, name: str) -> Any:
        content = self.__dict__.get("content", {})
        if name in content:
            return content[name]
        raise AttributeError(name)


def deliver(msg: Message) -> Any:
    """Call ``on_<subject>`` (or ``on_message``) of the receiver."""
    receiver = msg.receiver
    if msg.subject is not None:
        handler = getattr(type(receiver), f"on_{msg.subject}", None)
        if handler is None:
            raise AttributeError(
                f"Receiver (id = {receiver.id!r}) does not implement on_{msg.subject}()."
            )
        return getattr(receiver, f"on_{msg.subject}")(msg)
    return receiver.on_message(msg)


__all__ = ["Message", "SocialNetwork", "deliver"]
