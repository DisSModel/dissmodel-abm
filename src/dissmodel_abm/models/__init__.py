"""Models ported from TerraME's ``logo`` package (version 0.5.6)."""
from .ants import Ants
from .disease import Disease
from .life_cycle import LifeCycle
from .predator_prey import PredatorPrey
from .schelling import Schelling
from .single_agent import SingleAgent

__all__ = ["Ants", "Disease", "LifeCycle", "PredatorPrey", "Schelling", "SingleAgent"]
