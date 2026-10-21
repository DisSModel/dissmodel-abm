"""
dissmodel-abm: TerraME-style agent-based modeling for DisSModel.

Agents are Python objects (:class:`Agent`), grouped in societies
(:class:`Society`) and placed in a cellular space (:class:`CellSpace`) whose
cell attributes are the bands of a dissmodel ``RasterBackend``. The model
(:class:`AgentModel`) is a regular dissmodel ``Model``: it runs in a
``dissmodel.core.Environment`` and works with ``RasterMap``, ``Chart`` and
``@track_plot``.
"""
from .core import (
    OCCUPANCY_BAND,
    Agent,
    AgentModel,
    Cell,
    CellSpace,
    Group,
    Message,
    Neighborhood,
    Society,
    SocialNetwork,
    Trajectory,
)

__version__ = "0.2.0"

__all__ = [
    "Agent",
    "AgentModel",
    "Cell",
    "CellSpace",
    "Group",
    "Message",
    "Neighborhood",
    "OCCUPANCY_BAND",
    "Society",
    "SocialNetwork",
    "Trajectory",
    "__version__",
]
