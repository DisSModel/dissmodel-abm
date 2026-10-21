"""
Which :class:`~dissmodel_abm.core.model.AgentModel` is being set up.

``Society`` and ``CellSpace`` objects created inside ``AgentModel.setup()``
take the model's random generator from here, so a single ``seed`` controls
every random draw of a run. This mirrors how ``dissmodel.core.Environment``
tracks the active environment (``Environment._current``).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from .model import AgentModel

_stack: list[AgentModel] = []


def push(model: AgentModel) -> None:
    _stack.append(model)


def pop(model: AgentModel) -> None:
    if _stack and _stack[-1] is model:
        _stack.pop()


def current_model() -> AgentModel | None:
    """Return the model whose ``setup()`` is running, or ``None``."""
    return _stack[-1] if _stack else None


def resolve_rng(rng: Any = None) -> np.random.Generator:
    """
    Return the generator to use.

    Priority: an explicit ``rng`` (a ``Generator`` or an integer seed), then
    the generator of the model being set up, then a fresh unseeded generator.
    The last case is meant for quick experiments outside a model; it is not
    reproducible.
    """
    if isinstance(rng, np.random.Generator):
        return rng
    if rng is not None:
        return np.random.default_rng(rng)
    model = current_model()
    if model is not None:
        return model.rng
    return np.random.default_rng()
