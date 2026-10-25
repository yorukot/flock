# Results files

A run with an output directory writes `results.json` and one predictions file per task:

```text
<out>/
  results.json
  predictions/<task>.jsonl
```

## `results.json`

```text
{
  "provenance": { ... },
  "tasks": [
    {
      "task": "arc_easy",
      "task_hash": "<sha256>",
      "primary_metric": "acc_per_char",
      "metrics": {"acc_raw": ..., "acc_per_token": ..., "acc_per_char": ..., "acc_per_byte": ..., "acc_uncond": ..., "bpb": ...},
      "num_docs": 2376,
      "limit": null,
      "seconds": ...,
      "truncation": {"requests": 0, "docs": 0, "max_tokens_cut": 0},
      "docs_per_second": ...
    }
  ],
  "suites": [
    {"suite": "base_easy:qa_rc", "members": ["arc_easy"], "metric": null, "score": ...},
    {"suite": "base_easy:qa_bpb", "members": ["arc_easy"], "metric": "bpb", "score": ...},
    {"suite": "base_easy", "members": ["base_easy:qa_rc", "base_easy:qa_bpb"], "metric": null, "score": null}
  ],
  "schema_version": 4
}
```

The file is a [`~eval.results.Results`]. Each entry of `tasks` is a [`~eval.results.TaskResult`] plus its `docs_per_second`, each entry of `suites` is a [`~eval.results.SuiteResult`], and `provenance` is described below. The class docstrings at the end of this page define every field.

### The task hash

`task_hash` is the sha256 of the task's definition (every field but `to_doc`, so its name and primary metric too), every scored context and continuation, and every gold index. Changing the dataset revision, the few-shot examples, the task's name, description, metrics or primary metric, or a choice's context or continuation or the gold index that `to_doc` returns changes it. Doc ids aren't hashed. Two results with the same hash scored the same requests under the same task definition.

A perplexity task's hash covers its definition (name, dataset, domain and window) and every doc's id and text.

### Suites

`suites` holds the suites the run was given and every suite they hold, each after its members, and is empty for a run of tasks only. Each entry lists its members by name, and every member has its own entry in `tasks` or `suites`, so you can recompute the score: the mean of each member task's `metric` (its `primary_metric` when `metric` is `null`) and each member suite's `score`. A suite that only groups others has a `null` score. A suite's members can grow between versions of `flock-eval`, so compare two suite scores only when their `members` match, every member task has the same `task_hash`, and neither run used a `limit`.

### Truncation

A context too long for the model loses tokens from the left, and the continuation is kept whole (see [How scoring works](how-scoring-works)). `truncation` says how often that happened in the task: how many requests were cut, how many docs had at least one cut request, and the most tokens cut from one request. Metrics on cut contexts can differ from metrics on whole ones, so check `docs` before comparing a task across models: a different `max_length` or tokenizer cuts different requests. `max_tokens_cut` counts the model's own tokens. All three are 0 when nothing was cut.

## Provenance

`provenance` records what produced the run, so a number can be traced to the exact model, code and settings: [`~eval.provenance.Provenance`] at the end of this page defines each field. `created_at` is taken after scoring, when `results.json` is assembled.

## Predictions

Each line of `predictions/<task>.jsonl` is one doc:

| Field | Meaning |
|---|---|
| `id` | The doc's id in the dataset. |
| `gold` | The index of the correct choice. |
| `logprob` | Each choice's summed log-probability. |
| `num_tokens` | Each choice's token count. |
| `logprob_uncond` | Each choice's log-probability after the unconditional context, for tasks that score pointwise mutual information (PMI). |

A perplexity task's line has other fields, enough to recompute `bits_per_byte` over any set of docs:

| Field | Meaning |
|---|---|
| `id` | The doc's id in the dataset. |
| `logprob` | The text's summed log-probability over all of its windows. |
| `num_tokens` | The text's token count. |
| `num_bytes` | The text's UTF-8 byte count. |
| `windows` | How many windows the text was scored in. |

## Classes

[[autodoc]] eval.results.Results

[[autodoc]] eval.results.TaskResult

[[autodoc]] eval.results.SuiteResult

[[autodoc]] eval.tasks.Truncation

[[autodoc]] eval.provenance.Provenance
