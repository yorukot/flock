"""Every task and suite `flock-eval` can run, by name."""

from flock.eval.tasks.arc import ARC_EASY
from flock.eval.tasks.base_easy import BASE_EASY
from flock.eval.tasks.paloma import PALOMA
from flock.eval.tasks.suite import Suite, Task, by_name

SUITES: dict[str, Suite] = by_name((*BASE_EASY.suites, *PALOMA.suites))

# Every task in a suite is registered, so `--task` runs what `--suite` does.
TASKS: dict[str, Task] = by_name((ARC_EASY, *BASE_EASY.tasks, *PALOMA.tasks))
