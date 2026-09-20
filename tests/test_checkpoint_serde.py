from __future__ import annotations

import typing

from pydantic import BaseModel

from services.memory import _CHECKPOINT_ALLOWED_TYPES, _CHECKPOINT_SERDE
from state.state import PlanItem, SpecialistResult, SupervisorState, TaskRecord


def _base_models(annotation) -> set[type[BaseModel]]:
    found: set[type[BaseModel]] = set()
    for arg in typing.get_args(annotation) or (annotation,):
        if isinstance(arg, type) and issubclass(arg, BaseModel):
            found.add(arg)
        elif typing.get_args(arg):
            found |= _base_models(arg)
    return found


def test_allowlist_covers_every_checkpointed_state_model():
    """
    Every project-defined model that can reach a checkpointed SupervisorState
    channel must be in the serde allowlist, otherwise it deserializes silently
    as a plain dict and fails further downstream.
    """
    models: set[type[BaseModel]] = set()
    for field in SupervisorState.model_fields.values():
        models |= _base_models(field.annotation)

    # Only project models: langchain_core messages have native serde support
    # and are deliberately not part of the msgpack allowlist.
    models = {m for m in models if m.__module__.startswith("state.")}
    models.discard(SupervisorState)

    assert models, "expected to discover checkpointed custom models"
    missing = models - set(_CHECKPOINT_ALLOWED_TYPES)
    assert not missing, f"checkpointed models missing from allowlist: {missing}"


def test_typed_roundtrip_preserves_custom_models():
    values = [
        SpecialistResult(source="sql", summary="2 employees", status="done"),
        TaskRecord(
            turn=1,
            route="sql",
            task="count employees",
            status="done",
            result_summary="2",
        ),
        [
            TaskRecord(
                turn=1,
                route="rag",
                task="policy",
                status="done",
                result_summary="found policy text",
            )
        ],
        PlanItem(
            route="sql",
            task="count employees",
            status="done",
            result_summary="2",
        ),
    ]

    for value in values:
        restored = _CHECKPOINT_SERDE.loads_typed(_CHECKPOINT_SERDE.dumps_typed(value))
        assert type(restored) is type(value)
        assert restored == value
