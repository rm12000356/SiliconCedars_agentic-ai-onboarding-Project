import math
import os
from pathlib import Path
from typing import Literal
from uuid import uuid4

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from state.state import SupervisorState, SpecialistResult, ChartSpec
from services.llm import llm

OUTPUT_DIR = Path("outputs")

CHART_EXTRACTION_PROMPT = """
Extract a chart specification from the user's request.

Rules:
- Use ONLY numeric values explicitly present in the request.
- NEVER invent, estimate, calculate, or infer missing numbers.
- Preserve the provided numbers exactly.
- Preserve the provided labels exactly when possible.
- If the request does not contain enough concrete data to build a chart,
  return empty labels and values.
- Choose bar, line, or pie based on the user's requested visualization.
- Do not use any external knowledge.
"""


def Visualization(state: SupervisorState) -> dict:
    """
    Builds a chart from specialist structured data or, when unavailable,
    extracts explicit chart data from the current task.

    Important architectural rule:
        Existing structured data is never sent through the LLM again.
    """

    if not state.current_task:
        return _failed_result(
            "Visualization could not start because no task was provided.",
            "missing_current_task",
        )

    print(f"[VISU] current_task={state.current_task!r}")

    try:

        if (
            state.last_result is not None
            and state.last_result.structured_data
        ):
            print(
                "[VISU] using structured_data from "
                f"{state.last_result.source}"
            )

            spec = _extract_structured_rows(
                state.last_result.structured_data
            )

            spec = spec.model_copy(
                update={
                    "title": _title_from_task(state.current_task),
                    "chart_type": _chart_type_from_task(
                        state.current_task
                    ),
                }
            )

        else:
            print("[VISU] no structured_data, using LLM extraction")

            spec = _extract_from_task(state.current_task)



        _validate_spec(spec)

        print(f"[VISU] chart_spec={spec!r}")


        filepath = _render_chart(spec)

        print(f"[VISU] saved chart to {filepath}")

        return {
            "last_result": SpecialistResult(
                source="visu",
                summary=(
                    f"Chart '{spec.title}' created successfully "
                    f"and saved to {filepath}."
                ),
                status="done",
            )
        }

    except ValueError as exc:
        print(f"[VISU] validation failure: {exc}")

        return _failed_result(
            "The visualization request did not contain valid chart data.",
            "invalid_chart_spec",
        )

    except Exception as exc:
        print(f"[VISU] rendering/extraction failure: {exc}")

        return _failed_result(
            "The visualization could not be generated.",
            "visualization_failed",
        )


def _failed_result(summary: str, issue: str) -> dict:
    return {
        "last_result": SpecialistResult(
            source="visu",
            summary=summary,
            status="failed",
            issue=issue,
        )
    }


def _extract_structured_rows(rows: list[dict]) -> ChartSpec:
    """
    Convert already-structured specialist data into ChartSpec.

    This path is intentionally deterministic. No LLM is involved.
    """

    if not rows:
        raise ValueError("No structured rows were provided.")

    labels: list[str] = []
    values: list[float] = []

    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Structured row {index} is not an object.")

        if "label" not in row or "value" not in row:
            raise ValueError(
                f"Structured row {index} must contain 'label' and 'value'."
            )

        label = str(row["label"]).strip()

        try:
            value = float(row["value"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Structured row {index} contains a non-numeric value."
            ) from exc

        if not label:
            raise ValueError(f"Structured row {index} has an empty label.")

        if not math.isfinite(value):
            raise ValueError(
                f"Structured row {index} contains a non-finite value."
            )

        labels.append(label)
        values.append(value)

    if len(labels) != len(values):
        raise ValueError("Labels and values must have the same length.")

    return ChartSpec(
        chart_type="bar",
        title="Chart",
        labels=labels,
        values=values,
    )


def _extract_from_task(task: str) -> ChartSpec:
    """
    LLM fallback used only when structured specialist data is unavailable.

    The LLM extracts intent and explicit numbers from the user's text.
    It does not call tools and does not generate underlying data.
    """

    model = llm().with_structured_output(ChartSpec)

    spec = model.invoke(
        [
            {
                "role": "system",
                "content": CHART_EXTRACTION_PROMPT,
            },
            {
                "role": "user",
                "content": task,
            },
        ]
    )

    if not isinstance(spec, ChartSpec):
        spec = ChartSpec.model_validate(spec)

    return spec


def _validate_spec(spec: ChartSpec) -> None:
    if not spec.labels or not spec.values:
        raise ValueError("No chart data was provided.")

    if len(spec.labels) != len(spec.values):
        raise ValueError("Number of labels does not match number of values.")

    if spec.chart_type == "pie":
        if any(value < 0 for value in spec.values):
            raise ValueError("Pie charts cannot contain negative values.")

        if all(value == 0 for value in spec.values):
            raise ValueError("Pie charts cannot contain only zero values.")


def _safe_title(title: str) -> str:
    cleaned = "".join(
        character if character.isalnum() else "_"
        for character in title
    )

    cleaned = cleaned.strip("_")

    return cleaned[:80] or "chart"


def _render_chart(spec: ChartSpec) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    filename = f"{_safe_title(spec.title)}_{uuid4().hex[:8]}.png"
    filepath = OUTPUT_DIR / filename

    fig, ax = plt.subplots(figsize=(10, 6))

    try:
        if spec.chart_type == "bar":
            ax.bar(spec.labels, spec.values)
            ax.tick_params(axis="x", rotation=45)

        elif spec.chart_type == "line":
            ax.plot(
                spec.labels,
                spec.values,
                marker="o",
            )
            ax.tick_params(axis="x", rotation=45)

        elif spec.chart_type == "pie":
            ax.pie(
                spec.values,
                labels=spec.labels,
                autopct="%1.1f%%",
            )

        ax.set_title(spec.title)

        if spec.chart_type != "pie":
            fig.tight_layout()

        fig.savefig(
            str(filepath),
            format="png",
            dpi=150,
            bbox_inches="tight",
        )

    finally:
        plt.close(fig)

    return filepath


def _title_from_task(task: str) -> str:
    """
    Deterministic title fallback.

    We deliberately do not ask another LLM call just to produce a title.
    """

    cleaned = " ".join(task.split())

    if len(cleaned) <= 80:
        return cleaned

    return cleaned[:77].rstrip() + "..."


def _chart_type_from_task(task: str) -> Literal["bar", "line", "pie"]:
    """
    Deterministic chart-type selection for structured specialist data.

    Explicit user intent wins. Bar is the safe default.
    """

    text = task.lower()

    if "pie chart" in text or "pie graph" in text:
        return "pie"

    if "line chart" in text or "line graph" in text:
        return "line"

    return "bar"