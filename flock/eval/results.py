"""The versioned results file of an eval run, and the per-doc predictions beside it."""

import dataclasses
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from flock.eval.metrics import ChoiceScores
from flock.eval.provenance import Provenance
from flock.eval.requests import Scores
from flock.eval.tasks.base import MCDoc, Truncation
from flock.eval.tasks.perplexity import TextDoc


@dataclass(frozen=True, slots=True)
class TaskResult:
    """
    One task's scores.

    Args:
        task (`str`):
            The task's registry name.
        task_hash (`str`):
            sha256 of exactly what was scored, as [Results files](results#the-task-hash) defines it. Equal hashes mean the same requests were scored.
        primary_metric (`str`):
            The metric to read first.
        metrics (`dict[str, float]`):
            Each metric over the task's docs: a mean over docs for multiple choice, and a ratio of sums over all docs for `bits_per_byte`.
        num_docs (`int`):
            How many docs were scored.
        limit (`int`, *optional*):
            The doc limit the run used, or `None` for the whole split.
        seconds (`float`):
            Wall time of scoring, excluding data and model loading.
        truncation (`Truncation`):
            How many requests and docs had their contexts cut from the left to fit the model.
    """

    task: str
    task_hash: str
    primary_metric: str
    metrics: dict[str, float]
    num_docs: int
    limit: int | None
    seconds: float
    truncation: Truncation

    @property
    def docs_per_second(self) -> float:
        """Scoring throughput."""
        return self.num_docs / self.seconds if self.seconds > 0 else 0.0


@dataclass(frozen=True, slots=True)
class SuiteResult:
    """
    One suite's score.

    Args:
        suite (`str`):
            The suite's registry name.
        members (`list[str]`):
            Its members' names, tasks and suites, each with an entry of its own in the results.
        metric (`str`, *optional*):
            The metric read from each member task, or `None` for each task's primary metric.
        score (`float`, *optional*):
            The unweighted mean of the members' scores, or `None` for a suite that only groups others.
    """

    suite: str
    members: list[str]
    metric: str | None
    score: float | None


@dataclass(frozen=True, slots=True)
class Results:
    """
    Every task's scores from one run, and its provenance.

    Args:
        provenance (`Provenance`):
            What produced the run.
        tasks (`list[TaskResult]`):
            Each task's scores, in run order.
        suites (`list[SuiteResult]`):
            Each suite's score, every suite after the suites it holds.
        schema_version (`int`, *optional*, defaults to 4):
            The version of the file's layout, bumped when a field changes.
    """

    provenance: Provenance
    tasks: list[TaskResult]
    suites: list[SuiteResult]
    schema_version: int = 4

    def to_dict(self) -> dict[str, Any]:
        """Returns the results as plain JSON-ready data, with each task's `docs_per_second` added."""
        data = dataclasses.asdict(self)
        for task, entry in zip(self.tasks, data["tasks"], strict=True):
            entry["docs_per_second"] = task.docs_per_second

        return data

    def write(self, out: Path) -> Path:
        """Writes `results.json` into `out`, creating it if needed, and returns its path."""
        out.mkdir(parents=True, exist_ok=True)
        path = out / "results.json"
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        return path


def write_predictions(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Writes prediction rows to `path` as JSON lines, creating its directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def prediction_rows(docs: Sequence[MCDoc], scores: ChoiceScores) -> list[dict[str, Any]]:
    """Returns one row per doc with its id, gold index and each choice's raw scores, dropping the padding past each doc's last choice."""
    # One host copy per field, not per doc.
    logprob = scores.logprob.tolist()
    num_tokens = scores.num_tokens.tolist()
    logprob_uncond = scores.logprob_uncond.tolist() if scores.logprob_uncond is not None else None

    rows = []
    for i, doc in enumerate(docs):
        count = len(doc.choices)
        row: dict[str, Any] = {
            "id": doc.id,
            "gold": doc.gold,
            "logprob": logprob[i][:count],
            "num_tokens": num_tokens[i][:count],
        }
        if logprob_uncond is not None:
            row["logprob_uncond"] = logprob_uncond[i][:count]
        rows.append(row)

    return rows


def perplexity_prediction_rows(
    docs: Sequence[TextDoc], scores: Scores, num_bytes: torch.Tensor, window: int
) -> list[dict[str, Any]]:
    """Returns one row per doc with its id, summed log-probability, token and UTF-8 byte counts and number of windows, so any domain's `bits_per_byte` can be recomputed."""
    # One host copy per field, not per doc.
    logprob = scores.logprob.tolist()
    num_tokens = scores.num_tokens.tolist()
    doc_bytes = num_bytes.tolist()

    rows = []
    for i, doc in enumerate(docs):
        rows.append(
            {
                "id": doc.id,
                "logprob": logprob[i],
                "num_tokens": num_tokens[i],
                "num_bytes": doc_bytes[i],
                # Rounded up: the last window holds what is left.
                "windows": -(-num_tokens[i] // window),
            }
        )

    return rows
