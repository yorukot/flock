"""Suites: named groups of tasks, scored as the unweighted mean of their members."""

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass

from flock.eval.tasks.base import MultipleChoiceTask
from flock.eval.tasks.perplexity import PerplexityTask

# Every kind of task a suite can hold and a run can score.
type Task = MultipleChoiceTask | PerplexityTask


def by_name[T: Task | Suite](items: Iterable[T]) -> dict[str, T]:
    """
    Returns `items` by name, in order, each once.

    Raises:
        `ValueError`: if two different items share a name. Results are keyed by name, so they would score as one.
    """
    named: dict[str, T] = {}
    for item in items:
        if named.setdefault(item.name, item) != item:
            raise ValueError(f"two different definitions are named {item.name}")

    return named


@dataclass(frozen=True, slots=True)
class Suite:
    """
    A named group of tasks and suites. A scored suite's score is the unweighted mean of its members' scores, as OLMES's `macro` aggregate (`oe_eval/tasks/aggregate_tasks.py`).

    A member task's score is its `metric`, or its primary metric if `metric` isn't set. A member suite's score is its own score, under its own `metric`. A scored suite's members share one scale: all accuracies, all multiple-choice `bpb`, or all perplexity `bits_per_byte`.

    Args:
        name (`str`):
            The suite's registry name.
        members (`tuple[MultipleChoiceTask | PerplexityTask | Suite, ...]`):
            The tasks and suites it groups, each name once.
        metric (`str`, *optional*):
            The metric read from every member task. Each task's primary metric if not set. Member suites keep their own, so a suite with a `metric` holds only tasks.
        scored (`bool`, *optional*, defaults to `True`):
            Whether the suite has a score. An unscored suite only groups suites whose scores don't average, such as an accuracy and a `bpb`, so that one name runs them all.

    Raises:
        `ValueError`: if `members` is empty or repeats a name, a member task doesn't report the metric read from it, members mix scales, `metric` is set next to a member suite, or `scored` is `False` with a `metric` or a direct member task. Also if a scored suite holds an unscored one, or two different tasks, or two different suites, under it share a name.
    """

    name: str
    members: tuple[Task | Suite, ...]
    metric: str | None = None
    scored: bool = True

    def __post_init__(self) -> None:
        names = [member.name for member in self.members]
        if not names:
            raise ValueError(f"{self.name}: a suite needs members")
        if len(set(names)) != len(names):
            raise ValueError(f"{self.name}: members repeat a name: {names}")

        if self.scored:
            self._check_scored_members()
            self._check_one_scale()
        elif self.metric is not None or any(
            not isinstance(member, Suite) for member in self.members
        ):
            raise ValueError(f"{self.name}: an unscored suite groups only suites and has no metric")

        try:
            by_name(self._walk_tasks())
            by_name(self._walk_suites())
        except ValueError as error:
            raise ValueError(f"{self.name}: {error}") from None

    def _check_scored_members(self) -> None:
        """Raises `ValueError` if a member can't give a score under this suite."""
        for member in self.members:
            if isinstance(member, Suite):
                # A member suite is scored under its own metric, so this one's would misname the
                # score.
                if self.metric is not None:
                    raise ValueError(
                        f"{self.name}: metric reads member tasks only, not {member.name}"
                    )
                if not member.scored:
                    raise ValueError(f"{self.name}: member {member.name} has no score")
            elif (metric := self.metric or member.primary_metric) not in member.reported_metrics:
                raise ValueError(f"{self.name}: member {member.name} doesn't report {metric}")

    def _check_one_scale(self) -> None:
        """Raises `ValueError` if the members' scores are on different scales."""
        # OLMES doesn't check this: its suites keep `rc` and `bpb` apart by name only. A mean of
        # an accuracy (higher is better) and a `bpb` (lower is better) has no meaning. A `bpb`
        # (a mean over docs) and a `bits_per_byte` (a ratio over all bytes) weight docs
        # differently, so they don't average either.
        scales = {self._scale(member) for member in self.members}
        if len(scales) > 1:
            raise ValueError(f"{self.name}: members mix {sorted(scales)}, which don't average")

    def _scale(self, member: Task | Suite) -> str:
        """The scale of the member's score under this suite: `accuracy`, `bpb` or `bits_per_byte`."""
        if isinstance(member, Suite):
            # A suite's members share one scale, checked when it was built, so its first member
            # stands for all of them.
            return member._scale(member.members[0])

        metric = self.metric or member.primary_metric
        return metric if metric in ("bpb", "bits_per_byte") else "accuracy"

    @property
    def tasks(self) -> tuple[Task, ...]:
        """Every task under the suite, depth first, each once."""
        return tuple(by_name(self._walk_tasks()).values())

    @property
    def suites(self) -> tuple[Suite, ...]:
        """This suite and every suite under it, each after its own members and each once."""
        return tuple(by_name(self._walk_suites()).values())

    def score(self, metrics: Mapping[str, Mapping[str, float]]) -> float:
        """
        Returns the mean of the members' scores, given each task's metrics by task name.

        Raises:
            `ValueError`: if the suite is unscored.
            `KeyError`: if `metrics` lacks a member task or the metric read from it.
        """
        if not self.scored:
            raise ValueError(f"{self.name}: an unscored suite has no score")

        member_scores = []
        for member in self.members:
            if isinstance(member, Suite):
                member_scores.append(member.score(metrics))
            else:
                member_scores.append(metrics[member.name][self.metric or member.primary_metric])

        return sum(member_scores) / len(member_scores)

    def _walk_tasks(self) -> Iterator[Task]:
        """Yields every task under the suite, depth first, repeats included."""
        for member in self.members:
            if isinstance(member, Suite):
                yield from member._walk_tasks()
            else:
                yield member

    def _walk_suites(self) -> Iterator[Suite]:
        """Yields every suite under this one, members first, then this one, repeats included."""
        for member in self.members:
            if isinstance(member, Suite):
                yield from member._walk_suites()
        yield self
