"""Run tasks against a model: the one entry point the CLI and job runners call."""

import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch

from flock.eval.metrics import bits_per_byte, multiple_choice
from flock.eval.model import DEFAULT_BATCH_TOKENS, ScoringModel
from flock.eval.provenance import Provenance
from flock.eval.results import (
    Results,
    SuiteResult,
    TaskResult,
    perplexity_prediction_rows,
    prediction_rows,
    write_predictions,
)
from flock.eval.tasks import SUITES, TASKS, PerplexityTask, Suite, Task, Truncation
from flock.eval.tasks.suite import by_name


@dataclass(frozen=True, slots=True)
class EvalConfig:
    """
    One eval run. Plain data, so it pickles to a remote worker.

    Args:
        model (`str`):
            The model as `repo@revision`.
        tasks (`tuple[str, ...]`, *optional*):
            Registry names of the tasks to run, in order.
        out (`Path`, *optional*):
            Where to write `results.json` and `predictions/<task>.jsonl`. Nothing is written if not set.
        limit (`int`, *optional*):
            Score only each task's first `limit` docs, for quick checks.
        batch_tokens (`int`, *optional*):
            Passed to [`~eval.model.ScoringModel`].
        device (`str`, *optional*):
            Where to run. Defaults to CUDA when available.
        wandb_project (`str`, *optional*):
            Also log the summary to this tracker project through the `wandb` SDK.
        command (`str`, *optional*, defaults to `"python"`):
            The command line recorded in the provenance. The default marks a run started from the Python API. The CLI passes its own command line.
        suites (`tuple[str, ...]`, *optional*):
            Registry names of the suites to run. Their tasks run after `tasks`, each task once, and each suite and the suites it holds are scored.

    Raises:
        `ValueError`: if no task or suite is given, a task or suite repeats, or `limit` or `batch_tokens` is less than 1, before any model loads.
    """

    model: str
    tasks: tuple[str, ...] = ()
    out: Path | None = None
    limit: int | None = None
    batch_tokens: int = DEFAULT_BATCH_TOKENS
    device: str | None = None
    wandb_project: str | None = None
    command: str = "python"
    suites: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.tasks and not self.suites:
            raise ValueError("give at least one task or suite")
        if len(set(self.tasks)) != len(self.tasks):
            raise ValueError(f"tasks repeat: {self.tasks}")
        if len(set(self.suites)) != len(self.suites):
            raise ValueError(f"suites repeat: {self.suites}")
        if self.limit is not None and self.limit < 1:
            raise ValueError(f"limit must be at least 1, got {self.limit}")
        if self.batch_tokens < 1:
            raise ValueError(f"batch_tokens must be at least 1, got {self.batch_tokens}")


def evaluate(config: EvalConfig) -> Results:
    """
    Loads the model, scores every task and suite and writes the results, then logs them to the tracker if `wandb_project` is set.

    Raises:
        `KeyError`: if a task or suite name isn't in its registry.
        `ValueError`: if two different tasks, or two different suites, to run share a name.
        `FileExistsError`: if `config.out` exists and isn't an empty directory.
    """
    _check_names("tasks", config.tasks, TASKS)
    _check_names("suites", config.suites, SUITES)
    suites = [SUITES[name] for name in config.suites]
    tasks = _tasks_to_run(config.tasks, suites)
    _check_out(config.out)

    model = ScoringModel.from_pretrained(
        config.model, device=config.device, batch_tokens=config.batch_tokens
    )
    task_results = [score_task(model, task, config.limit, config.out) for task in tasks]

    provenance = Provenance.collect(
        model=model.name,
        command=config.command,
        device=model.device,
        dtype=model.dtype,
        batch_tokens=model.batch_tokens,
        max_length=model.max_length,
        attention=model.attention,
    )
    results = Results(provenance, task_results, _suite_results(suites, task_results))

    if config.out is not None:
        results.write(config.out)
    if config.wandb_project is not None:
        # Imported here so importing this module, as training loops do for `score_task`, doesn't
        # import wandb.
        from flock.eval.tracking import log_to_tracker

        log_to_tracker(results, config.wandb_project)

    return results


def _check_names(kind: str, names: tuple[str, ...], registry: Mapping[str, object]) -> None:
    """Raises `KeyError` if a name isn't in its registry, before any model loads."""
    # Checked here, not in `EvalConfig`: a config is plain data that may be built where the
    # registry differs from the worker's.
    unknown = [name for name in names if name not in registry]
    if unknown:
        raise KeyError(f"unknown {kind} {unknown}; known: {sorted(registry)}")


def _tasks_to_run(names: tuple[str, ...], suites: list[Suite]) -> list[Task]:
    """Returns the named tasks, then each suite's tasks not yet listed, each name once."""
    tasks = [TASKS[name] for name in names]
    for suite in suites:
        tasks.extend(suite.tasks)

    return list(by_name(tasks).values())


def _suite_results(suites: list[Suite], task_results: list[TaskResult]) -> list[SuiteResult]:
    """Scores each suite and the suites it holds, each after its members and each once."""
    metrics = {result.task: result.metrics for result in task_results}
    held = []
    for suite in suites:
        held.extend(suite.suites)

    results = []
    for suite in by_name(held).values():
        score = suite.score(metrics) if suite.scored else None
        members = [member.name for member in suite.members]
        results.append(SuiteResult(suite.name, members, suite.metric, score))

    return results


def _check_out(out: Path | None) -> None:
    """Raises `FileExistsError` if `out` exists and isn't an empty directory."""
    # Old predictions next to a new results.json would mix two runs.
    if out is not None and out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f"{out} is not an empty directory")


def score_task(
    model: ScoringModel, task: Task, limit: int | None = None, out: Path | None = None
) -> TaskResult:
    """
    Scores one task with an already loaded model, as training loops do between steps.

    Args:
        model (`ScoringModel`):
            The model to score.
        task (`MultipleChoiceTask` or `PerplexityTask`):
            The task.
        limit (`int`, *optional*):
            Score only the first `limit` docs.
        out (`Path`, *optional*):
            Write per-doc scores to `out/predictions/<task>.jsonl`, replacing an existing file. No `results.json` is written: that is [`evaluate`]'s job.
    """
    if isinstance(task, PerplexityTask):
        return _score_perplexity(model, task, limit, out)

    docs, shots = task.load(limit)
    prepared = task.requests(docs, shots)

    start = time.perf_counter()
    request_scores = model.loglikelihood(prepared.requests)
    scores = prepared.choice_scores(request_scores)
    reported = task.reported_metrics
    per_doc = {name: values for name, values in multiple_choice(scores).items() if name in reported}
    # One host copy for every metric's mean. It also waits for the GPU, so `seconds` times the
    # scoring.
    means = torch.stack(list(per_doc.values())).mean(dim=1).tolist()
    seconds = time.perf_counter() - start
    # `means` follows the order of `per_doc`'s keys.
    metrics = dict(zip(per_doc, means, strict=True))
    truncation = prepared.truncation(request_scores)

    if out is not None:
        rows = prediction_rows(prepared.docs, scores)
        write_predictions(out / "predictions" / f"{task.name}.jsonl", rows)

    return TaskResult(
        task=task.name,
        task_hash=prepared.hash,
        primary_metric=task.primary_metric,
        metrics=metrics,
        num_docs=len(prepared.docs),
        limit=limit,
        seconds=seconds,
        truncation=truncation,
    )


def _score_perplexity(
    model: ScoringModel, task: PerplexityTask, limit: int | None, out: Path | None
) -> TaskResult:
    """[`score_task`] for a perplexity task."""
    docs = task.load(limit)
    texts = [doc.text for doc in docs]
    # Each whole text's UTF-8 bytes, as OLMES's `num_bytes` (`loglikelihood_rolling_verbose` in
    # `oe_eval/models/eleuther_huggingface.py`).
    num_bytes = torch.tensor([len(text.encode()) for text in texts], dtype=torch.int64)

    start = time.perf_counter()
    scores = model.rolling_loglikelihood(texts, task.window)
    # Summed over all of the task's docs before dividing, as Paloma defines bits per byte
    # (App. B) and OLMES computes it per domain (`PerplexityMetric`), so a doc counts by its
    # bytes. `.item()` is the task's one host copy, and it waits for the GPU, so `seconds`
    # times the scoring.
    metric = bits_per_byte(scores.logprob.sum(), num_bytes.sum()).item()
    seconds = time.perf_counter() - start

    if out is not None:
        rows = perplexity_prediction_rows(docs, scores, num_bytes, task.window)
        write_predictions(out / "predictions" / f"{task.name}.jsonl", rows)

    return TaskResult(
        task=task.name,
        task_hash=task.task_hash(docs),
        primary_metric=task.primary_metric,
        metrics={"bits_per_byte": metric},
        num_docs=len(docs),
        limit=limit,
        seconds=seconds,
        # Every token is predicted once, in windows that fit the model, so nothing is cut.
        truncation=Truncation(requests=0, docs=0, max_tokens_cut=0),
    )
