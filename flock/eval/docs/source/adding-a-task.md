# Adding a task

A multiple-choice task is a definition and one registry line. You don't write any scoring code.

## How `arc_easy` is defined

This is the whole task, `flock/eval/tasks/arc.py`:

<literalinclude>{"path": "../../tasks/arc.py", "language": "python"}</literalinclude>

## Steps

1. **Pin the dataset** to a commit hash. Find it on the dataset's **Files and versions** tab, or with `huggingface_hub.HfApi().dataset_info(repo).sha`.
2. **Write a function** that turns one dataset row into an [`~eval.tasks.MCDoc`]. It renders each choice as the model sees it: [`~eval.tasks.cloze`] gives OLMES's `Question: ... Answer:` format, as `arc_doc` uses. [The prompt format](#the-prompt-format) covers other formats.
3. **Choose the few-shot examples** by id. Use the curated OLMES examples when the task has them, so results compare with published numbers. Examples that aren't rows of the dataset go in `Fewshot.rows`, in the dataset's row format.
4. **Choose the primary metric.** Follow OLMES or OLMo 3 for the task. [Metrics](metrics) lists the options.
5. **Register the task.** Add it to its suite, such as `_QA_TASKS` in `flock/eval/tasks/base_easy.py` for a Base Easy QA task (see [Suites](tasks#suites)): every task in a suite is registered. A task in no suite goes in the tuple that builds `TASKS` in `flock/eval/tasks/registry.py`. `flock-eval run --task` accepts it from then on.
6. **Test the row function.** Add a test in `tests/eval/test_tasks.py` that maps a hand-written row to the expected [`~eval.tasks.MCDoc`], as `test_arc_doc_maps_letter_and_digit_keys` does. Tests never download: `tests/eval/conftest.py` makes every task read the toy rows in `tests/eval/toy.py`.
7. **List the task** in the table of [Tasks](tasks#available-tasks).

To check a new task, print the first doc's prompt:

```python
from flock.eval.tasks import TASKS

task = TASKS["arc_easy"]
docs, shots = task.load(limit=1)
print(task.requests(docs, shots).requests[0])
```

Then score a few docs with a small model:

```bash
uv run flock-eval run \
  --model allenai/OLMo-2-0425-1B@a1847dff35000b4271fa70afc5db10fd29fedbdf \
  --task arc_easy \
  --limit 20 \
  --out results/arc-easy-check
```

## The prompt format

Each [`~eval.tasks.Choice`] of a doc is one request: its `continuation` is scored after the task's `description`, the few-shot examples and its `context`. Each example is its gold choice's context and continuation, followed by a blank line. With [`~eval.tasks.cloze`], every choice shares the context `Question: <question>\nAnswer:` and its continuation is `" <choice>"`:

```text
Question: <few-shot question 1>
Answer: <its gold choice>

...

Question: <question>
Answer:
```

For pointwise mutual information (PMI, `acc_uncond`), each choice is scored a second time after the task's `unconditional_context` alone (`Answer:` by default).

A task with another format writes its own choices in `to_doc`:
- **Other prefixes:** `cloze(goal, answers, question_prefix="Goal: ")`.
- **A context per choice:** give each `Choice` its own `context` and the same `continuation`, as WinoGrande does.
- **A gold text only**, such as a math solution: one `Choice` with `gold=0`, and `metrics=("bpb",)` on the task, since an accuracy over one choice is always 1.
- **Text before the examples**, such as a subject header: the task's `description`.
- **Examples from the scored split:** set `Fewshot.num_shots` and list one id more than it. Each doc is shown the first `num_shots` examples that aren't the doc itself. The ids must come from the split the task scores, since an id identifies a doc within one split; the task rejects any other.

A changed prompt is a different task, and its numbers don't compare with the old ones, so give it a new name.

## A perplexity task

A [`~eval.tasks.PerplexityTask`] needs no row function: it reads each row's `id` and `text`, and keeps the rows whose `subdomain` is its `domain`. A Paloma source is one line in `DOMAINS` in `flock/eval/tasks/paloma.py`, with the domains `flock/eval/scripts/paloma_census.py` prints for it. `source_suite` makes one task per domain and the source's suite, and `PALOMA` registers them.

## The task hash

Each result records the task's [hash](results#the-task-hash), so results from before and after a change to the task never share one.
