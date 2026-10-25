# Tasks

A task is a [`~eval.tasks.MultipleChoiceTask`] or a [`~eval.tasks.PerplexityTask`] registered by name in `TASKS`, and a suite is a [`~eval.tasks.Suite`] of tasks registered in `SUITES`. [Adding a task](adding-a-task) shows how to write one.

## Available tasks

| Name | Benchmark | Split | Docs | Few-shot | Primary metric |
|---|---|---|---|---|---|
| `arc_easy` | [ARC-Easy](https://huggingface.co/datasets/allenai/ai2_arc) | `test`, all of it, as OLMo 3's Base Easy scores it | 2376 | 5, the curated OLMES examples | `acc_per_char` |

### Perplexity tasks

A perplexity task scores each doc's text whole and reports `bits_per_byte` over all of its docs (see [Metrics](metrics#perplexity)). Each [Paloma](https://arxiv.org/abs/2312.10523) domain is one task, scored on its whole `test` split in windows of 2048 tokens.

| Name | Source | Split | Docs | Window | Primary metric |
|---|---|---|---|---|---|
| `paloma_wikitext_103` | [Paloma](https://huggingface.co/datasets/allenai/paloma)'s `wikitext_103` | `test`, all of it | 61 | 2048 | `bits_per_byte` |

Paloma is gated: accept its terms on the dataset page before the first run.

## Suites

A suite's score is the unweighted mean of its members' scores, as OLMES's `macro` aggregate. A member task gives the suite's `metric`, or its own primary metric if the suite sets none. A member suite counts as one member, whatever its size. A scored suite's members share one scale: all accuracies, all `bpb`, or all perplexity `bits_per_byte`, since no two of them average. `base_easy` has no score of its own: it groups suites whose scores don't average, so that one name runs them all.

| Name | Members | Metric | OLMES suite |
|---|---|---|---|
| `base_easy:qa_rc` | `arc_easy` | each task's primary metric | `olmo3:base_easy:qa_rc` |
| `base_easy:qa_bpb` | `arc_easy` | `bpb` | `olmo3:base_easy:qa_bpb` |
| `base_easy` | `base_easy:qa_rc`, `base_easy:qa_bpb` | unscored | none |
| `paloma:wikitext_103` | `paloma_wikitext_103` | `bits_per_byte` | none; OLMES's task `paloma_wikitext_103::paloma` |
| `paloma` | `paloma:wikitext_103` | unscored | none |

A Paloma source is a suite of its domains, scored as their unweighted mean, as the paper averages a source's domains. OLMES instead reports a source's bits per byte over all of its docs together, so a source with several domains differs from OLMES's number. `paloma` has no score, since Paloma defines none over all sources.

The Base Easy suites hold only the tasks that exist so far, so their scores change as tasks are added. See [Suites in results](results#suites) for when two suite scores compare.

## Classes

[[autodoc]] eval.tasks.MultipleChoiceTask

[[autodoc]] eval.tasks.Dataset

[[autodoc]] eval.tasks.Fewshot

[[autodoc]] eval.tasks.MCDoc

[[autodoc]] eval.tasks.Choice

[[autodoc]] eval.tasks.cloze

[[autodoc]] eval.tasks.MCRequests

[[autodoc]] eval.tasks.PerplexityTask

[[autodoc]] eval.tasks.TextDoc

[[autodoc]] eval.tasks.Suite
