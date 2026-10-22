"""
A SIR epidemic with agents that meet through a social network.

Port of ``Disease`` from TerraME's ``logo`` package (version 0.5.6,
``lua/Disease.lua``). The original describes it as an example of the
decisions a modeler must take to go from system dynamics to agents; there is
no space, only a society and a social network.

Rules, as in the original
-------------------------
- ``quantity`` agents, all susceptible; one, drawn at random, falls sick
  before the first step.
- Each agent has ``contacts`` other agents drawn at random, **again every
  time the network is read** (``createSocialNetwork{quantity = contacts,
  inmemory = false}``).
- In each step, an infected agent sends a message with ``delay = 1`` to each
  of its contacts; with ``counter > duration`` it recovers; then its counter
  grows by one. So an agent infects during ``duration + 2`` steps.
- A susceptible agent that receives a message falls sick with probability
  ``probability``. Messages sent in step ``t`` arrive at the start of step
  ``t + 1``, before the agents act, and an agent infected then already
  sends messages in that step.

Defaults: ``quantity = 1000``, ``duration = 2``, ``contacts = 6``,
``probability = 0.25``, ``finalTime = 30``.

Comparison with TerraME
-----------------------
``log/Disease-chart-1.bmp`` (July 2017, same rules as the current code)
ends with about 12 to 17 agents never infected, where the port ends with
about 2. That chart is a single run: over 50 seeds, TerraME itself ends
with 2.7 never infected (0 to 7), as the port does and as the final-size
equation of the SIR model gives for R0 = 6 (four rounds of six messages at
probability 0.25). ``tests/test_goldens.py`` compares the port with those
TerraME runs.
"""
from __future__ import annotations

from dissmodel.visualization import track_plot
from dissmodel_abm.core import Agent, AgentModel, Society


class Person(Agent):
    state: str = "susceptible"
    counter: int = 0

    def sick(self) -> None:
        self.state = "infected"
        self.counter = 0
        self.model.count["infected"] += 1
        self.model.count["susceptible"] -= 1

    def on_message(self, msg) -> None:
        if self.state == "susceptible" and self.rng.random() < self.model.probability:
            self.sick()

    def execute(self) -> None:
        if self.state == "infected":
            for other in self.connections():
                self.message(other, delay=1)

            if self.counter > self.model.duration:
                self.state = "recovered"
                self.model.count["infected"] -= 1
                self.model.count["recovered"] += 1

            self.counter += 1


@track_plot("Susceptible", "tab:blue")
@track_plot("Infected", "tab:red")
@track_plot("Recovered", "tab:green")
class Disease(AgentModel):
    """
    Parameters (``setup``)
    ----------------------
    quantity : int
        Number of agents. Default 1000.
    duration : int
        Disease duration (see the rules). Default 2.
    contacts : int
        Contacts drawn for each agent every time it sends messages. Default 6.
    probability : float
        Probability that a message infects a susceptible agent. Default 0.25.
    """

    susceptible: int = 0
    infected: int = 0
    recovered: int = 0

    def setup(
        self,
        quantity: int = 1000,
        duration: int = 2,
        contacts: int = 6,
        probability: float = 0.25,
    ) -> None:
        self.duration = duration
        self.contacts = contacts
        self.probability = probability

        self.society = Society(Person, quantity=quantity)
        # Agents change these counters many times per step; the tracked
        # attributes below copy them once per step, so Chart gets one point
        # per step (the original's model.susceptible, infected, recovered).
        self.count = {"susceptible": len(self.society), "infected": 0, "recovered": 0}

        self.society.sample().sick()
        self.society.create_social_network(quantity=contacts, inmemory=False)
        self._publish()

    def execute(self) -> None:
        self.society.execute()
        self._publish()

    def _publish(self) -> None:
        self.susceptible = self.count["susceptible"]
        self.infected = self.count["infected"]
        self.recovered = self.count["recovered"]


__all__ = ["Disease", "Person"]
