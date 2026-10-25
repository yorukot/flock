import dataclasses

import pytest

from flock.eval.tasks import SUITES, TASKS, Suite

from .toy import TASK

OTHER = dataclasses.replace(TASK, name="other", primary_metric="acc_raw")
THIRD = dataclasses.replace(TASK, name="third", primary_metric="acc_raw")
GOLD_ONLY = dataclasses.replace(TASK, name="gold", metrics=("bpb",), primary_metric="bpb")

METRICS = {
    "toy": {"acc_per_char": 0.5, "acc_raw": 0.4, "bpb": 1.0},
    "other": {"acc_per_char": 0.3, "acc_raw": 0.9, "bpb": 2.0},
    "third": {"acc_raw": 0.7},
    "gold": {"bpb": 4.0},
}


def test_score_reads_each_tasks_primary_metric_unless_metric_is_set():
    assert Suite("rc", (TASK, OTHER)).score(METRICS) == pytest.approx((0.5 + 0.9) / 2)
    assert Suite("bpb", (TASK, OTHER), metric="bpb").score(METRICS) == pytest.approx(1.5)


def test_nested_suites_count_as_one_member_under_their_own_metric():
    inner = Suite("inner", (OTHER, THIRD), metric="acc_raw")
    outer = Suite("outer", (TASK, inner))

    # The inner mean (0.9 + 0.7) / 2 weighs as much as the one task beside it.
    assert outer.score(METRICS) == pytest.approx((0.5 + 0.8) / 2)
    bpb = Suite("bpb", (GOLD_ONLY, Suite("inner", (TASK, OTHER), metric="bpb")))
    assert bpb.score(METRICS) == pytest.approx((4.0 + 1.5) / 2)


def test_a_suite_may_share_a_name_with_a_task():
    # Tasks and suites are separate namespaces in the registries and in results.json.
    assert Suite("toy", (TASK,)).tasks == (TASK,)


def test_two_suites_over_one_task_list_share_their_tasks():
    rc = Suite("rc", (TASK, OTHER))
    bpb = Suite("bpb", (TASK, OTHER), metric="bpb")
    group = Suite("group", (rc, bpb), scored=False)

    assert group.tasks == (TASK, OTHER)
    assert [suite.name for suite in group.suites] == ["rc", "bpb", "group"]


def test_unscored_suite_has_no_score():
    group = Suite("group", (Suite("rc", (TASK,)),), scored=False)
    with pytest.raises(ValueError, match="no score"):
        group.score(METRICS)


def test_score_needs_every_member_tasks_metric():
    with pytest.raises(KeyError, match="other"):
        Suite("rc", (TASK, OTHER)).score({"toy": METRICS["toy"]})


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        pytest.param({"members": ()}, "needs members", id="empty"),
        pytest.param({"members": (TASK, TASK)}, "repeat", id="repeat"),
        pytest.param(
            {"members": (TASK, GOLD_ONLY), "metric": "acc_raw"},
            "gold doesn't report acc_raw",
            id="metric",
        ),
        pytest.param(
            {"members": (TASK, GOLD_ONLY)}, r"mix \['accuracy', 'bpb'\]", id="mixed-tasks"
        ),
        pytest.param(
            {"members": (TASK, Suite("inner", (OTHER,), metric="bpb"))},
            r"mix \['accuracy', 'bpb'\]",
            id="mixed-suites",
        ),
        pytest.param(
            {"members": (Suite("rc", (TASK,)),), "metric": "bpb"},
            "metric reads member tasks only",
            id="metric-over-suite",
        ),
        pytest.param(
            {"members": (Suite("a", (Suite("x", (TASK,)),)), Suite("b", (Suite("x", (OTHER,)),)))},
            "two different definitions are named x",
            id="suite-name-clash",
        ),
        pytest.param(
            {"members": (Suite("g", (Suite("rc", (TASK,)),), scored=False),)},
            "member g has no score",
            id="holds-unscored",
        ),
        pytest.param(
            {"members": (Suite("rc", (TASK,)),), "scored": False, "metric": "bpb"},
            "unscored",
            id="unscored-with-metric",
        ),
        pytest.param({"members": (TASK,), "scored": False}, "unscored", id="unscored-with-task"),
        pytest.param(
            {"members": (TASK, Suite("inner", (dataclasses.replace(TASK, description="x"),)))},
            "two different definitions are named toy",
            id="task-name-clash",
        ),
    ],
)
def test_suite_rejects_bad_definitions(kwargs, match):
    with pytest.raises(ValueError, match=match):
        Suite("s", **kwargs)


def test_registry_suites_hold_registered_tasks():
    assert SUITES["base_easy"].tasks == (TASKS["arc_easy"],)
    assert SUITES["base_easy:qa_bpb"].metric == "bpb"
    for suite in SUITES.values():
        assert all(TASKS[task.name] is task for task in suite.tasks)
