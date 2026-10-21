"""Tests of social networks and messages."""
from __future__ import annotations

import numpy as np
import pytest

from dissmodel.core import Environment
from dissmodel_abm import Agent, AgentModel, CellSpace, Message, Society, SocialNetwork


def rng(seed=0):
    return np.random.default_rng(seed)


class Inbox(Agent):
    def init(self):
        self.inbox = []

    def on_message(self, msg):
        self.inbox.append(msg)
        return "ok"

    def on_ping(self, msg):
        self.inbox.append(("ping", msg.value))


# -- SocialNetwork ---------------------------------------------------------------------


def test_social_network_basics():
    soc = Society(Agent, quantity=3, rng=rng())
    a, b, c = soc.agents
    net = SocialNetwork()
    net.add(b)
    net.add(c, 0.5)
    assert list(net) == [b, c] and len(net) == 2
    assert net.is_connection(b) and b in net and a not in net
    assert net.get_weight(c) == 0.5
    net.set_weight(c, 2.0)
    assert net.items() == [(b, 1.0), (c, 2.0)]
    with pytest.raises(ValueError, match="already"):
        net.add(b)
    net.remove(b)
    assert list(net) == [c]
    assert net.sample() is c


def test_agents_of_two_societies_do_not_collide():
    s1 = Society(Agent, quantity=1, rng=rng())
    s2 = Society(Agent, quantity=1, rng=rng())
    net = SocialNetwork()
    net.add(s1[1])
    net.add(s2[1])  # same id, different agent
    assert len(net) == 2


def test_add_and_get_social_network():
    soc = Society(Agent, quantity=3, rng=rng())
    a, b, c = soc.agents
    net = SocialNetwork()
    net.add(b)
    a.add_social_network(net)
    a.add_social_network(lambda ag: SocialNetwork(), "dynamic")
    assert a.get_social_network() is net
    assert a.connections() == [b]
    assert a.get_social_network("dynamic") is not a.get_social_network("dynamic")
    assert a.get_social_network("missing") is None
    with pytest.raises(ValueError, match="missing"):
        a.connections("missing")


# -- create_social_network ----------------------------------------------------------------


def test_quantity_in_memory_and_symmetric():
    soc = Society(Agent, quantity=20, rng=rng())
    soc.create_social_network(quantity=3)
    for agent in soc:
        others = agent.connections()
        assert len(others) == 3 and agent not in others
        assert agent.get_social_network() is agent.get_social_network()
    soc2 = Society(Agent, quantity=20, rng=rng())
    soc2.create_social_network(quantity=3, symmetric=True)
    for agent in soc2:
        for other in agent.connections():
            assert agent in other.get_social_network()


def test_quantity_not_in_memory_draws_new_contacts():
    soc = Society(Agent, quantity=50, rng=rng())
    soc.create_social_network(quantity=5, inmemory=False)
    agent = soc[1]
    draws = [tuple(a.id for a in agent.connections()) for _ in range(5)]
    assert all(len(d) == 5 for d in draws)
    assert len(set(draws)) > 1


def test_strategy_inference_and_errors():
    soc = Society(Agent, quantity=5, rng=rng())
    with pytest.raises(ValueError, match="infer"):
        soc.create_social_network()
    with pytest.raises(ValueError, match="infer"):
        soc.create_social_network(quantity=2, probability=0.5)
    with pytest.raises(ValueError, match="amount"):
        soc.create_social_network(quantity=5)
    soc.create_social_network(quantity=2)
    with pytest.raises(ValueError, match="already exists"):
        soc.create_social_network(quantity=2)
    with pytest.raises(ValueError, match="inmemory"):
        soc.create_social_network("erdos", name="e", quantity=2, inmemory=False)
    with pytest.raises(ValueError, match="symmetric"):
        soc.create_social_network(quantity=2, name="s", symmetric=True, inmemory=False)


def test_probability_and_function():
    soc = Society(Agent, quantity=30, rng=rng())
    soc.create_social_network(probability=1.0, name="all")
    assert all(len(a.connections("all")) == 29 for a in soc)
    soc.create_social_network(filter=lambda a, b: b.id == a.id + 1, name="next")
    assert [o.id for o in soc[4].connections("next")] == [5]
    assert soc[30].connections("next") == []


def test_cell_and_neighbor():
    Environment(start_time=1, end_time=1)

    class M(AgentModel):
        def setup(self):
            self.cs = CellSpace.grid(3)
            self.cs.create_neighborhood()
            self.soc = Society(Agent, quantity=4)
            self.place(self.soc, cellspace=self.cs, strategy="void")
            a, b, c, d = self.soc.agents
            a.enter(self.cs.get(1, 1))
            b.enter(self.cs.get(1, 1))
            c.enter(self.cs.get(0, 0))
            d.enter(self.cs.get(2, 2))

    m = M(seed=0)
    a, b, c, d = m.soc.agents
    m.soc.create_social_network("cell", name="cell")
    m.soc.create_social_network("neighbor", name="neigh")
    assert a.connections("cell") == [b]
    assert {o.id for o in a.connections("neigh")} == {c.id, d.id}
    assert {o.id for o in c.connections("neigh")} == {a.id, b.id}


def test_graph_strategies():
    soc = Society(Agent, quantity=40, rng=rng(1))
    soc.create_social_network("erdos", name="erdos", quantity=60)
    degrees = [len(a.connections("erdos")) for a in soc]
    assert sum(degrees) == 120
    for a in soc:
        for o in a.connections("erdos"):
            assert a in o.get_social_network("erdos")

    soc.create_social_network("barabasi", name="ba", start=5, quantity=3)
    later = soc.agents[5:]
    assert all(len(a.connections("ba")) >= 3 for a in later)

    soc.create_social_network("watts", name="ring", quantity=2, probability=0.0)
    assert [o.id for o in soc[1].connections("ring")][:2] == [2, 3]

    soc.create_social_network("void", name="empty")
    assert all(a.get_social_network("empty").is_empty() for a in soc)


# -- messages -----------------------------------------------------------------------------


def test_immediate_message_and_subject():
    soc = Society(Inbox, quantity=2, rng=rng())
    a, b = soc.agents
    assert a.message(b) == "ok"
    msg = b.inbox[0]
    assert isinstance(msg, Message) and msg.sender is a and msg.receiver is b
    a.message(b, subject="ping", value=7)
    assert b.inbox[1] == ("ping", 7)
    with pytest.raises(AttributeError, match="on_pong"):
        a.message(b, subject="pong")


def test_default_on_message_raises():
    soc = Society(Agent, quantity=2, rng=rng())
    with pytest.raises(NotImplementedError):
        soc[1].message(soc[2])


def test_delayed_message_arrives_at_the_next_step():
    order = []

    class Talker(Inbox):
        def execute(self):
            order.append(("exec", self.id, len(self.inbox)))
            if self.id == 1:
                self.message(self.society[2], delay=1)

    soc = Society(Talker, quantity=2, rng=rng())
    soc.execute()
    assert soc[2].inbox == [] and len(soc.messages) == 1
    soc.execute()  # delivered before the agents run
    assert order[-1] == ("exec", 2, 1)


def test_delay_counts_steps_and_dead_receivers_are_skipped():
    soc = Society(Inbox, quantity=3, rng=rng())
    a, b, c = soc.agents
    a.message(b, delay=2)
    a.message(c, delay=1)
    c.die()
    soc.synchronize()
    assert b.inbox == [] and c.inbox == [] and len(soc.messages) == 1
    soc.synchronize()
    assert len(b.inbox) == 1 and soc.messages == []
    with pytest.warns(UserWarning, match="dead"):
        a.message(c)


def test_delayed_message_needs_a_society():
    lone = Inbox()
    other = Society(Inbox, quantity=1, rng=rng())[1]
    with pytest.raises(RuntimeError, match="Society"):
        lone.message(other, delay=1)
