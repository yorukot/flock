# Tasks

A task is a [`~eval.tasks.MultipleChoiceTask`] or a [`~eval.tasks.PerplexityTask`] registered by name in `TASKS`, and a suite is a [`~eval.tasks.Suite`] of tasks registered in `SUITES`. [Adding a task](adding-a-task) shows how to write one.

## Available tasks

| Name | Benchmark | Split | Docs | Few-shot | Primary metric |
|---|---|---|---|---|---|
| `arc_easy` | [ARC-Easy](https://huggingface.co/datasets/allenai/ai2_arc) | `test`, all of it, as OLMo 3's Base Easy scores it | 2376 | 5, the curated OLMES examples | `acc_per_char` |

### Perplexity tasks

A perplexity task scores each doc's text whole and reports `bits_per_byte` over all of its docs (see [Metrics](metrics#perplexity)). Each [Paloma](https://arxiv.org/abs/2312.10523) domain is one task, scored on its whole `test` split in windows of 2048 tokens.

The first version has 9 of [Paloma](https://huggingface.co/datasets/allenai/paloma)'s 16 sources: those licensed under AI2's ImpACT Low Risk license or CC BY-SA, without the fringe sources. A source with one domain has one task, `paloma_<source>`. A source with several has one task per domain, `paloma_<source>__<domain>`, and `flock/eval/tasks/paloma.py` lists the domains.

| Source | Tasks | Domains | Docs | Licence |
|---|---|---|---|---|
| `c4_en` | `paloma_c4_en` | 1 | 2,124 | ImpACT LR |
| `mc4` | `paloma_mc4` | 1 | 1,202 | ImpACT LR |
| `wikitext_103` | `paloma_wikitext_103` | 1 | 61 | CC BY-SA |
| `ptb` | `paloma_ptb` | 1 | 1 | ImpACT LR |
| `falcon-refinedweb` | `paloma_falcon-refinedweb` | 1 | 1,711 | ImpACT LR |
| `dolma-v1_5` | `paloma_dolma-v1_5__<domain>` | 5 | 4,481 | ImpACT LR |
| `c4_100_domains` | `paloma_c4_100_domains__<domain>` | 99 | 14,131 | ImpACT LR |
| `dolma_100_subreddits` | `paloma_dolma_100_subreddits__<domain>` | 100 | 46,203 | ImpACT LR |
| `dolma_100_programing_languages` | `paloma_dolma_100_programing_languages__<domain>` | 100 | 7,832 | ImpACT LR |

Every task scores its domain's whole `test` split in windows of 2048 tokens, with `bits_per_byte` as its primary metric. `ptb` is one document of 98,844 OLMo 2 tokens. Paloma is gated: accept its terms on the dataset page before the first run.

## Suites

A suite's score is the unweighted mean of its members' scores, as OLMES's `macro` aggregate. A member task gives the suite's `metric`, or its own primary metric if the suite sets none. A member suite counts as one member, whatever its size. A scored suite's members share one scale: all accuracies, all `bpb`, or all perplexity `bits_per_byte`, since no two of them average. `base_easy` has no score of its own: it groups suites whose scores don't average, so that one name runs them all.

| Name | Members | Metric | OLMES suite |
|---|---|---|---|
| `base_easy:qa_rc` | `arc_easy` | each task's primary metric | `olmo3:base_easy:qa_rc` |
| `base_easy:qa_bpb` | `arc_easy` | `bpb` | `olmo3:base_easy:qa_bpb` |
| `base_easy` | `base_easy:qa_rc`, `base_easy:qa_bpb` | unscored | none |
| `paloma:<source>` | the source's tasks | `bits_per_byte` | OLMES's task `paloma_<source>::paloma`: the same number for a source with one domain, and its docs pooled instead of a mean of domains for a source with several |
| `paloma` | the 9 `paloma:<source>` suites | unscored | none |

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
