# How scoring works

This page explains how a request becomes a score, and which settings can move a score. It follows the reference code of [OLMES](https://github.com/allenai/olmes), so numbers can be checked against it.

## Tokenizing a request

A request is a context and a continuation, and its score is the summed log-probability of the continuation's tokens. The boundary between them is drawn the way OLMES draws it:

1. Whitespace at the end of the context moves to the start of the continuation.
2. The context and the continuation are tokenized together. The continuation's tokens are the ones after the context's own tokenization, so a token that spans the boundary counts as continuation.
3. No special tokens are added. An empty context becomes the beginning-of-sequence (BOS) token, or the end-of-sequence (EOS) token if the tokenizer has no BOS. A whitespace-only context counts as empty once its whitespace has moved. lm-evaluation-harness checks for an empty context before moving whitespace, so this one case differs from it.
4. An input longer than the model's `max_length` loses tokens from the left, so the context is cut and the continuation is scored in full. The results file counts the cuts per task ([Truncation](results#truncation)).

A continuation with no tokens raises an error, since scoring it as 0 would rank it above every real choice. So does a continuation longer than `max_length` on its own.

## Scoring a whole text

A [`~eval.tasks.PerplexityTask`] scores each doc's text whole with [`~eval.model.ScoringModel.rolling_loglikelihood`], as lm-evaluation-harness's `loglikelihood_rolling` does:

1. The text is tokenized with no special tokens.
2. It is split into windows of at most the task's `window` tokens, and every token is predicted exactly once. The first window predicts the first `window` tokens after one prefix token: BOS, or EOS if the tokenizer has no BOS. Each later window predicts the next `window` tokens from an input of exactly `window` tokens. The last window predicts what is left, and its input reaches back into the previous window.
3. The windows of all texts are batched together like requests, and each text's windows are summed in float64. Every position of a window is scored, so a batch of windows reaches the full `batch_tokens` times the vocabulary in logits. Lower `batch_tokens` if that doesn't fit beside a training run.

The window belongs to the task, not the model, so every model sees the same context. A model whose `max_length` is below the window raises an error. Nothing is cut, so a perplexity task's truncation is always 0.

OLMES starts the first window from EOS, whatever the tokenizer. That is the same token as BOS for OLMo 2, but not for Llama 3.2.

## Batching and precision

Requests are sorted longest first and packed into batches of at most `batch_tokens` padded tokens ([Command line](cli) gives the default). Rows are left-padded, so every continuation ends at the last position. The model then computes logits only for the continuation positions, never for the few-shot context, which keeps memory at about `batch_tokens` times the vocabulary size.

Log-softmax runs in float32, whatever the weights' dtype. In float32, a batched score matches the same request scored alone within 2e-4 nats. In bfloat16, the default for [`~eval.model.ScoringModel.from_pretrained`], a score depends slightly on which requests share its batch: about 0.05 nats on average and 0.49 at most, measured with OLMo 2 1B on 300 ARC-Easy docs at budgets of 32768 and 512 tokens.

So compare runs only when they used the same:
- `batch_tokens` and `limit`, which decide the batches
- dtype and attention implementation, which change bfloat16 arithmetic
- `max_length`, which decides where long inputs are cut

`results.json` records all of them: `limit` with each task, and the rest in its provenance. See [Results files](results#provenance).

## Metrics on the device

Scores stay on the GPU while a task runs. The metrics are tensor operations over every doc's choices at once, and nothing is copied to the host until the task's means are computed. [Metrics](metrics) defines each one.
