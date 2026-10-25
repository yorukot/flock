"""Perplexity tasks as data: a pinned dataset of plain texts, each scored whole."""

import dataclasses
import functools
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from flock.eval.tasks.base import Dataset


@dataclass(frozen=True, slots=True)
class TextDoc:
    """
    One text a perplexity task scores whole.

    Args:
        id (`str`):
            The text's id in its dataset, unique within a split. The predictions file records it.
        text (`str`):
            Exactly the scored text.
    """

    id: str
    text: str


@dataclass(frozen=True, slots=True)
class PerplexityTask:
    """
    A task that scores how well a model fits plain text, as Paloma does (Magnusson et al. 2023, arXiv 2312.10523): each doc whole, in disjoint windows (see [`~eval.model.ScoringModel.rolling_loglikelihood`]), reported as `bits_per_byte` over all of the task's docs.

    Rows have Paloma's format: `id` and `text`, plus `subdomain` in a source with more than one domain.

    Args:
        name (`str`):
            The task's registry name.
        dataset (`Dataset`):
            Where the scored rows come from.
        domain (`str`, *optional*):
            Score only the rows whose `subdomain` is this. Leave it unset only for a source with one domain, whose rows have no `subdomain`.
        window (`int`, *optional*, defaults to 2048):
            The most tokens in one model input. It is part of the task, not the model, so every model is scored with the same context. A model whose `max_length` is smaller raises.

    Raises:
        `ValueError`: if `window` is less than 1.
    """

    name: str
    dataset: Dataset
    domain: str | None = None
    window: int = 2048

    def __post_init__(self) -> None:
        if self.window < 1:
            raise ValueError(f"{self.name}: window must be at least 1, got {self.window}")

    @property
    def primary_metric(self) -> str:
        """The metric reported first, `bits_per_byte`."""
        return "bits_per_byte"

    @property
    def reported_metrics(self) -> tuple[str, ...]:
        """The metrics the task reports: `bits_per_byte` only."""
        return ("bits_per_byte",)

    def load(self, limit: int | None = None) -> list[TextDoc]:
        """
        Loads the scored split's docs in the task's domain, keeping the first `limit` (all if `None`).

        Raises:
            `ValueError`: if `limit` is less than 1, no row is in the domain, or the rows have domains and the task sets none, which would pool the domains into one number.
        """
        if limit is not None and limit < 1:
            raise ValueError(f"limit must be at least 1, got {limit}")

        rows = _split_rows(self.dataset)
        # A null `subdomain` counts as none: `datasets` fills a column a row lacks with None.
        if self.domain is None and any(row.get("subdomain") is not None for row in rows):
            raise ValueError(f"{self.name}: the rows have domains; give the task its domain")
        if self.domain is not None:
            rows = [row for row in rows if row.get("subdomain") == self.domain]
        if not rows:
            raise ValueError(f"{self.name}: no rows in domain {self.domain!r}")

        return [TextDoc(row["id"], row["text"]) for row in rows[:limit]]

    def task_hash(self, docs: Sequence[TextDoc]) -> str:
        """Returns the sha256 of the definition and every doc's id and text, as [Results files](results#the-task-hash) defines it."""
        digest = hashlib.sha256(json.dumps(dataclasses.asdict(self), sort_keys=True).encode())
        for doc in docs:
            digest.update(json.dumps([doc.id, doc.text]).encode())

        return digest.hexdigest()


# One split at a time: the domains of a source run one after another and share its split, which
# takes about 2 seconds to load for the 100-domain sources, so loading it once per domain would
# add minutes to a run. A bound of one keeps a training process from holding every source.
@functools.lru_cache(maxsize=1)
def _split_rows(dataset: Dataset) -> list[dict[str, Any]]:
    """The scored split's rows, shared by every task on the same `Dataset`. Callers must not change them."""
    return dataset.load()
