# Metrics

Every multiple-choice task reports the metrics of [OLMES](https://arxiv.org/abs/2406.08446) (sec. 3.3) and the bits per byte of [OLMo 3](https://arxiv.org/abs/2512.13961) (sec. 3.3.2). [`~eval.metrics.multiple_choice`] below lists each metric and how it scores a choice.

`bpb` doesn't depend on the tokenizer, which makes it the fairest comparison between models with different vocabularies. Lower is better.

## Perplexity

A [`~eval.tasks.PerplexityTask`] reports one metric, `bits_per_byte`: the summed log-probability of all of the task's docs, negated and divided by their total UTF-8 bytes, in base 2 (Paloma, App. B). It is a ratio of sums, not a mean over docs, so a long doc counts by its length. Multiple-choice `bpb` is a mean over docs, which is why the two have different names and never share a suite. Lower is better.

## Functions

[[autodoc]] eval.metrics.multiple_choice

[[autodoc]] eval.metrics.accuracy

[[autodoc]] eval.metrics.bits_per_byte

[[autodoc]] eval.metrics.ChoiceScores
