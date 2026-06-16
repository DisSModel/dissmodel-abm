"""
dissmodel_abm
=============
Agent-based modeling (ABM) extension for `dissmodel`, inspired by
TerraME's Agent/Society API (https://www.terrame.org/base/types/agent/).

This package does not modify dissmodel's core. It provides:

- ``dissmodel_abm.core.Society`` / ``Agent`` — the protective layer
  between the modeler and the underlying spatial substrate (vector
  GeoDataFrame today; raster planned). Models read and write agents as
  objects (``agent.energy``, ``agent.die()``, ``agent.reproduce()``)
  instead of touching ``self.gdf`` directly.
- ``dissmodel_abm.core.AgentModel`` — base class (subclass of
  ``dissmodel.geo.vector.SpatialModel``) exposing ``self.society``, plus
  legacy batch/vectorized operations kept for backward compatibility:
  ``walk``, ``move_to``, ``die_if``, ``reproduce_if``,
  ``neighbors_within``, ``all_neighbors_within``.
- ``dissmodel_abm.models`` — example models (``RandomWalkModel``,
  ``PredatorPreyModel``, ``SchellingModel``), all written against
  ``self.society``.
"""
from .core import AgentModel, Society, Agent
from .models import RandomWalkModel, PredatorPreyModel, SchellingModel

__all__ = [
    "AgentModel",
    "Society",
    "Agent",
    "RandomWalkModel",
    "PredatorPreyModel",
    "SchellingModel",
]

__version__ = "0.2.0"
