# Scoring during training

You can score a model you already hold in memory, such as a checkpoint between training steps, without saving it or loading it again.

## Score a task

Wrap the model and its tokenizer in a [`~eval.model.ScoringModel`], then call [`~eval.run.score_task`]:

```python
from flock.eval.model import ScoringModel
from flock.eval.run import score_task
from flock.eval.tasks import TASKS

scorer = ScoringModel(model, tokenizer, name="llm-pretrain@step-2000")
result = score_task(scorer, TASKS["arc_easy"])
print(result.metrics["bpb"])
```

`name` labels the model in log messages, such as the truncation warning. Use the run and step, so you can tell checkpoints apart. The training loop records the step with the metrics itself.

`score_task` returns a [`~eval.results.TaskResult`] with each metric over the task's docs: a mean over docs for a multiple-choice metric, and a ratio of sums for a perplexity task's `bits_per_byte` (see [Metrics](metrics#perplexity)). It writes nothing unless you pass `out`, and then only `out/predictions/<task>.jsonl`, replacing an existing file. It never writes `results.json`.

## What it does to the model

- **Modes are restored.** Scoring switches the model to eval mode and puts each module back in the mode it was in, so training continues as before.
- **No gradients.** Scoring runs under `torch.no_grad()`.
- **The model stays where it is.** Scoring uses the model's device and dtype as they are. Scores stay on that device until the task's means are computed.

## Keep runs comparable

- Score with the same `batch_tokens` at every step. In bfloat16, scores depend slightly on which requests share a batch. See [Batching and precision](how-scoring-works#batching-and-precision).
- Score full tasks for numbers you compare with other runs. A `limit` is fine for a quick signal, but its results aren't comparable to full runs.

## Logging to the tracker

Log the metrics with the training run's own `wandb` run, as any other metric. Use the `<task>/<metric>` keys that `flock-eval run --wandb-project` writes to its run summary, so one metric has one name everywhere:

```python
run.log({f"arc_easy/{name}": value for name, value in result.metrics.items()}, step=step)
```

The curves then show up in Flock next to the training loss. See [Logging from training code](https://docs.cerulean.works/flock/latest/guide/logging).
