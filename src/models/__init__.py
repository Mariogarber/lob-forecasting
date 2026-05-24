"""Model registry.

Adding a new model is a 3-step recipe:

  1. Create a class that subclasses ``ClassicalModel`` or ``SequenceModel``
     (see ``models.base``).
  2. Decorate it with ``@register_model("my-model-name", kind="classical" | "sequence")``.
  3. Import the module from ``models/classical/__init__.py`` or
     ``models/sequence/__init__.py`` so the decorator fires at import time.

The pipeline CLI looks the model up by name in ``MODEL_REGISTRY`` and
hands it a config dict; the trainer dispatches on ``model.kind``.
"""

from __future__ import annotations

from typing import Callable, Type

from models.base import BaseModel, ClassicalModel, SequenceModel

# name -> (class, kind)
MODEL_REGISTRY: dict[str, tuple[Type[BaseModel], str]] = {}


def register_model(name: str, *, kind: str) -> Callable[[Type[BaseModel]], Type[BaseModel]]:
    if kind not in {"classical", "sequence"}:
        raise ValueError(f"kind must be 'classical' or 'sequence', got {kind!r}")

    def wrapper(cls: Type[BaseModel]) -> Type[BaseModel]:
        if name in MODEL_REGISTRY:
            raise KeyError(f"Model '{name}' already registered.")
        MODEL_REGISTRY[name] = (cls, kind)
        cls._registered_name = name
        cls._registered_kind = kind
        return cls

    return wrapper


def get_model_class(name: str) -> tuple[Type[BaseModel], str]:
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model: {name!r}. Available: {sorted(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[name]


def list_models(kind: str | None = None) -> list[str]:
    if kind is None:
        return sorted(MODEL_REGISTRY)
    return sorted(n for n, (_cls, k) in MODEL_REGISTRY.items() if k == kind)


# Eagerly import all model modules so their decorators register.
from models.classical import (  # noqa: E402,F401
    linear, ridge, random_forest, lightgbm_model,
)
from models.sequence import (  # noqa: E402,F401
    gru, lstm, transformer, deeplob, mamba2, tcn,
)


__all__ = [
    "BaseModel",
    "ClassicalModel",
    "SequenceModel",
    "MODEL_REGISTRY",
    "register_model",
    "get_model_class",
    "list_models",
]
