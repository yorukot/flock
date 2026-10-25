# 006: Paloma perplexity, bits per byte per domain

| | |
| --- | --- |
| Status | built in two PRs: the core with `wikitext_103`, then the other 8 sources and the parity table |
| Date | 2026-10-24 |
| Builds on | [001](001-eval-core.md) (scoring, `bits_per_byte`), [003](003-base-easy.md) (`Suite`, D18) |
| Keeps | 001's tokenization, batching and float32 log-softmax; 002's `truncation` field (always 0 here); D18's unweighted suite means |
| Result | 309 domains on OLMo 2 1B in 424 seconds of scoring; checks 1 and 3's source means pass; check 2's per-document rule misses on 137 of 77,746 documents (section 7) |

Decision numbers are provisional (`006.D1`, ...) and take the next free global numbers when the PR is opened.

## 1. Goal

1. **A perplexity task.** Score how well a model fits plain text, as bits per byte, for the domains of Paloma (Magnusson et al. 2023, arXiv 2312.10523). Perplexity moves smoothly from 60M parameters up, where multiple-choice accuracy sits at chance.
2. **Two levels of results.** Each domain's bits per byte, and each source's unweighted mean over its domains, as Paloma reports it.
3. **Parity.** Per-domain numbers on `allenai/OLMo-2-0425-1B` match OLMES's Paloma tasks within a tolerance fixed before the full comparison runs.

**Out of scope:**
- decontaminating our pretraining corpus against Paloma (Paloma's G1);
- Traditional Chinese domains, which are a separate task with their own name (section 6);
- Paloma's fringe sources and the sources not licensed for our use (section 2.3);
- sliding-window scoring, and perplexity per token, character or word.

---

## 2. What Paloma is

### 2.1 The paper's rules

Paloma is 16 sources (existing corpora) split into 546 domains (fine-grained parts of a source, such as one subreddit or one C4 URL domain), each with a `val` and a `test` split. Its evaluation rules (sec. 3, App. C.2):

- **G3, subsampling.** Each domain is sampled to about 100 thousand tokens per split and each source to at least 1 million, "to tolerate variance equally for each domain" (sec. 3, App. C.2.1). We score the splits whole.
- **G4, vocabulary.** Models with different tokenizers are compared in bits per byte, `BPB = -l / (B ln 2)` for a summed log-likelihood `l` over `B` UTF-8 bytes (App. B), following the Pile.
- **G5, evaluation format.** "Documents are evaluated individually, e.g., `<BOS>document 1` then `<BOS>document 2`", not packed together, and "documents longer than maximum sequence length are split into disjoint inputs", which Press et al. call nonoverlapping inference (App. C.2.3).
- **Aggregation.** Per-source results are "a macro average of perplexity over domains", so that every domain has an equal say (sec. 4, Fig. 2). Within a domain, the likelihood is summed over all of its tokens.

### 2.2 How OLMES implements it

Read from allenai/olmes at `5a51f502d463b8cdc4a2dcad7d7096c41ff1197e`, the commit 003's parity used.

- **Tasks.** `paloma_<source>` for each of the 16 sources (`oe_eval/tasks/oe_eval_tasks/paloma.py`, `oe_eval/datasets_extra/paloma_sources.py`), configured as `paloma_<source>::paloma` with split `test` and primary metric `bits_per_byte` (`oe_eval/configs/tasks.py:5239-5246`). The suite `paloma::paloma` is the unweighted mean over the 16 sources (`oe_eval/configs/task_suites.py:687-690`).
- **Requests.** Each document is one `loglikelihood_rolling` request over its whole `text` (`oe_eval/tasks/base_task.py:329-343`). The model splits it with lm-evaluation-harness's `get_rolling_token_windows(..., prefix_token=self.eot_token_id, max_seq_len=self.max_length, context_len=1)` and `make_disjoint_window` (`oe_eval/models/eleuther_huggingface.py:178-260`; `lm_eval/utils.py:204-248` at lm-eval 0.4.3), then sums the windows' log-probabilities per document.
- **Windows.** With `context_len=1`, the first window predicts the first `min(L, n)` tokens after one prefix token, where `L` is `max_length` and `n` the document's length. Each later window predicts the next `L` tokens and conditions on the one token before them, so the model's input is exactly `L` tokens. The last window predicts what is left, and its input is the `L` tokens before its last target, so it conditions on more of the previous window than the others do. Every token is predicted exactly once.
- **Metric.** `PerplexityMetric` (`oe_eval/metrics/perplexity.py`) keys each document by `<source>__<subdomain>` and computes, per domain, `-sum(logprob) / (sum(bytes) * ln 2)` over the domain's documents, where `bytes` is the UTF-8 length of the document's text. The task's primary `bits_per_byte` is the same sum over every document of the source.

**Where OLMES and the paper differ:**
- OLMES's per-source number sums over all of the source's documents (a micro average), while the paper averages over domains (a macro average). Per-domain numbers are the same in both.
- OLMES's first window starts from EOS (`eot_token_id`), while the paper writes `<BOS>document` and lm-evaluation-harness's own `loglikelihood_rolling` uses BOS, falling back to EOS (`prefix_token_id` in `lm_eval/models/huggingface.py:360-366, 872`). For OLMo 2 these are the same token, `<|endoftext|>`. For Llama 3.2 they are not.

### 2.3 Dataset and licences

`allenai/paloma` at `65cd6fc59dba021b21db414fa5e8d7765ffbe5e6` (2024-06-06, the head on 2026-10-24). It is gated: each user accepts the AI2 ImpACT License, Low Risk Artifacts (LR), on the dataset page. Files are `<source>/<split>/*.jsonl.gz`, and each row has `text` and `id`, plus `subdomain` in sources with more than one domain. The dataset card licenses every source under LR except those it lists.

| Source | Domains | Tokens per split (paper, Table 1) | Licence (card) | First version |
| --- | --- | --- | --- | --- |
| `c4_en` | 1 | 1,000,000 | LR | yes |
| `mc4` | 1 | 1,000,000 | LR | yes |
| `wikitext_103` | 1 | 265,552 | CC BY-SA | yes |
| `ptb` | 1 | 95,868 | LR | yes |
| `falcon-refinedweb` | 1 | 1,000,000 | LR | yes |
| `dolma-v1_5` | 6 (5 in `test`) | about 3,000,000 | LR | yes |
| `c4_100_domains` | 99 | about 9,800,000 | LR | yes |
| `dolma_100_subreddits` | 100 | about 9,680,000 | LR | yes |
| `dolma_100_programing_languages` | 100 | about 10,000,000 | LR | yes |
| `redpajama` | 7 | about 700,000 | "see license details" | no: mixed licences |
| `m2d2_s2orc_unsplit` | 167 | about 16,700,000 | CC BY-NC | no: non-commercial |
| `m2d2_wikipedia_unsplit` | 49 | about 4,890,000 | CC BY-NC | no: non-commercial |
| `twitterAAE_HELM_fixed` | 2 | about 720,000 | research only | no |
| `manosphere_meta_sep`, `gab`, `4chan_meta_sep` | 9, 1, 1 | about 1,000,000 each | LR | no: fringe sources |

Token counts are the paper's, with the GPT-NeoX-20B tokenizer, half of its val-plus-test column. The paper leaves the three fringe sources out of its headline perplexity too (sec. 4.1).

**The first version's `test` split, counted at the pinned revision** with OLMo 2's tokenizer (`allenai/OLMo-2-0425-1B` at `a1847dff`, no special tokens):

| Source | Domains | Docs | OLMo 2 tokens | Docs over 2048 tokens | Windows of 2048 |
| --- | --- | --- | --- | --- | --- |
| `c4_en` | 1 | 2,124 | 964,544 | 50 | 2,200 |
| `mc4` | 1 | 1,202 | 955,824 | 59 | 1,335 |
| `wikitext_103` | 1 | 61 | 285,336 | 39 | 170 |
| `ptb` | 1 | 1 | 98,844 | 1 | 49 |
| `falcon-refinedweb` | 1 | 1,711 | 966,185 | 72 | 1,821 |
| `dolma-v1_5` | 5 | 4,481 | 2,372,348 | 171 | 4,916 |
| `c4_100_domains` | 99 | 14,131 | 9,448,808 | 370 | 14,686 |
| `dolma_100_subreddits` | 100 | 46,203 | 9,302,083 | 20 | 46,223 |
| `dolma_100_programing_languages` | 100 | 7,832 | 8,573,391 | 763 | 10,361 |
| **Total** | **309** | **77,746** | **32,967,363** | **1,545** | **81,761** |

- `dolma-v1_5`'s `books` file is empty in `test` (`test_books.jsonl.gz`), so the source has 5 domains there: `common-crawl`, `pes2o`, `reddit_uniform`, `stack_uniform`, `wiki`.
- No text is empty, and no id repeats within a source, so a doc's id is its `id` field.
- Domain names carry Paloma's rank prefix (`00_AskReddit`, `97_archives.lib.state.ma.us`) and use only letters, digits and `_.+#-`.
- The smallest domain is `c4_100_domains`' `97_archives.lib.state.ma.us`: 1 doc, 464 tokens. It is kept, as Paloma keeps it (four of C4's 100 domains fall short of the 100 thousand target, sec. 2, fn. 4), but it is the noisiest member of its source's mean.
- `ptb` is one document of 98,844 tokens, scored as 49 windows.

---

## 3. Design decisions

### 006.D1. A domain is a task, and a source is a suite

Each Paloma domain is one task, and each source is a `Suite` of its domains with `metric="bits_per_byte"`. D18 already gives a suite's score as the unweighted mean of its members and writes each suite to the `suites` section of `results.json`, so the paper's two levels need no new aggregation: domains are in `tasks`, sources in `suites`.

- **Names.** A domain's task is `paloma_<source>__<subdomain>`, with `<source>` the dataset configuration. A source with one domain has the task `paloma_<source>`. OLMES keys a domain the same way but with each row's `source` field (`PerplexityMetric`), which differs from the configuration in 5 of the 9 sources (`Dolma-v1_5`, `Dolma_Reddit`, `dolma-100-programing-languages`, `PTB`, `falcon-refinedweb/data`), so parity maps documents by id, not by name. A source's suite is `paloma:<source>`. `paloma` is an unscored suite over the 9 sources, so one name runs them all.
- **No score over all sources.** The paper defines none (it gives "ordinary perplexity" over pooled tokens and per-source macro averages), and OLMES's `paloma::paloma` averages micro per-source numbers we don't report. A mean over sources can be added later, under its own name.
- **Domains are data.** The domain names are frozen in the task module, read once from the pinned revision, and the tasks are generated from that list, as MMLU's 57 subjects are.
- **Cost:** 309 entries in `tasks` for one `--suite paloma` run. That is the per-domain record the paper asks for. A compact CLI summary is a separate change.

The alternative, one task per source holding its domains inside, would need a nested per-domain field in `TaskResult` and its own mean over domains, which repeats `Suite`.

### 006.D2. `PerplexityTask` is data

```python
@dataclass(frozen=True, slots=True)
class PerplexityTask:
    name: str
    dataset: Dataset  # 003's pinned `Dataset`; rows have `text` and `id`
    domain: str | None = None  # keep rows whose `subdomain` is this; every row if None
    window: int = 2048  # the most tokens in one model input (006.D4)
```

- `primary_metric` is `"bits_per_byte"` and `reported_metrics` is `("bits_per_byte",)`, both properties, so `Suite` reads a perplexity task as it reads a multiple-choice one.
- A doc is the row's `id` and `text`. Ids are unique within every source at the pinned revision (section 2.3), so no row index is needed.
- Loading filters the source's rows on `subdomain`. 100 domains of one source would then load the same split 100 times, so the build measures that cost and, if it matters, caches a pinned split's rows for the run.
- The task hash covers the definition and every document's text, as D7 hashes what is scored.

Only Paloma's row format is supported. A later dataset with other field names adds the fields it needs then.

### 006.D3. A rolling request: a document scored whole, in disjoint windows

`ScoringModel` gains one public method:

```python
def rolling_loglikelihood(self, texts: Sequence[str], window: int) -> Scores: ...
```

- **Tokens.** Each text is tokenized whole, with no special tokens, as OLMES's `tok_encode` does for OLMo 2 (`add_bos_token` is false for it).
- **Windows.** The windows are exactly lm-evaluation-harness's `get_rolling_token_windows(context_len=1)` followed by `make_disjoint_window` (section 2.2), with `max_seq_len=window`. Each window is a (context ids, continuation ids) pair, which is what `loglikelihood` scores after `_encode`, so windows go through the same `_batches` and `_score`. Sorting longest first and the token budget work across documents unchanged.
- **The first window's prefix** is `_prefix_token`: BOS, else EOS (Harry, 2026-10-24). This follows the paper's `<BOS>document` and lm-evaluation-harness, and differs from OLMES only for tokenizers whose BOS and EOS differ. Parity can't see the difference, since OLMo 2's two are one token.
- **Per document.** The windows' log-probabilities are laid out one row per text on the device and summed along each row, so the sum is the same in every run (the stage file first said `index_add_`, which adds in any order on CUDA). The sum is float64 (Harry, 2026-10-25): a 100,000-token text sums to about 3e5 nats, which float32 resolves only to about 0.03, close to check 2's budget for `ptb` (49 windows, 0.049 nats). OLMES adds windows in Python floats. `Scores.logprob` is float64 from this method only. `Scores` keeps its three fields, one entry per text. `tokens_cut` is all zeros, since every token is predicted once and nothing is cut, so the task's `truncation` is always 0.
- **The only change to the existing path** is moving the batching loop of `loglikelihood` into a private method both callers share. It is a move with no behaviour change, checked by comparing every multiple-choice score on the tiny model before and after on the CPU.
- **Raises** if `window` is below 1 or above `max_length`, and if a text has no tokens, which would give 0 bits over 0 bytes (as D11 refuses an empty continuation).

**Why not reuse 003's gold-only docs** (one `Choice` with an empty context): `loglikelihood` raises on a continuation longer than `max_length` (D2), and making it split would change what a multiple-choice request does; the window would be the model's `max_length`, not a task setting (006.D4); and the metric reduces differently (006.D5). A separate method keeps each path one job.

### 006.D4. The window is 2048 tokens for every model

A longer window gives every model more context, so lower bits per byte. Using each model's own `max_length` would score OLMo 2 (4096) and Llama 3.2 (131,072) under different settings, against Principle 3. So `window` is part of the task, 2048 by default (Harry, 2026-10-24), which is the sequence length of Paloma's own 1B baselines (App. G) and fits a model trained at 2048. A model whose `max_length` is below the task's window raises rather than scoring a different task. A task at another window is another task with its own name.

The parity run passes `max_length` 2048 to OLMES through its model arguments, since OLMES's windows are its `max_length`.

### 006.D5. Bits per byte summed over a domain

A domain's `bits_per_byte` is `-sum(logprob) / (sum(bytes) * ln 2)` over its documents, where `bytes` is the UTF-8 length of a document's whole text. This is Paloma's definition (App. B) and OLMES's per-domain number (`compute_aggregate_metrics` in `oe_eval/metrics/perplexity.py`). It is computed on the device, one host copy per task.

It is not 001's `bpb`, which is a mean of per-doc values and so weights a 50-byte post like a 50,000-byte article. The two get different names, following OLMES, which calls the perplexity metric `bits_per_byte` and the multiple-choice one `bits_per_byte_corr`. `Suite`'s scale check (D18, members must share one scale) learns `bits_per_byte` as a third scale, so no suite averages it with a multiple-choice `bpb`.

The predictions file has one row per document: `id`, `logprob`, `num_tokens`, `num_bytes` and `windows`. A reader can recompute any domain's number, or OLMES's micro per-source number, from it.

### 006.D6. A source's score is the mean over its domains

`paloma:<source>` is the unweighted mean of its domains' `bits_per_byte` (006.D1), the paper's per-source macro average (sec. 4, Fig. 2). The paper averages perplexity, and we average bits per byte, which is the log of per-byte perplexity, so we average logs rather than perplexities. We have no precedent for this choice. It keeps one domain with a very high perplexity from dominating its source, which is the effect the paper reports (arXiv at 391,171 in RedPajama for its C4 baseline, sec. 4.1), and it is the scale every other `flock.eval` suite averages on. A source with one domain scores its domain's number.

### 006.D7. The first version: 9 sources, the full test split

- **Sources:** the 9 in section 2.3, licensed LR or CC BY-SA, with no fringe source (Harry, 2026-10-24): 309 domains in `test`. The others can be added later, each with its licence recorded.
- **Split:** `test`, as OLMES's `::paloma` configs (Harry, 2026-10-24).
- **Whole splits.** Paloma already sized each domain for its variance (G3), so a task scores its whole split. `--limit` caps docs per domain for smoke runs only.
- **Smoke scale.** `--suite paloma --limit 10` scores 3,020 documents. The full first version is 33.0 million OLMo 2 tokens in 81,761 windows. A 60M model should take a few H200-minutes and a 1B model about ten, which the first smoke run measures.

### 006.D8. How parity is measured

The setup is D22's: OLMES from its own environment, `allenai/OLMo-2-0425-1B` at `a1847dff35000b4271fa70afc5db10fd29fedbdf`, OLMES at its default batch size 1, plus `max_length` 2048 on both sides (006.D4), on the `test` split. Three checks per domain, in order:

1. **Same documents.** OLMES writes every request (`--save-raw-requests`). Each domain's document ids and texts match ours, in order, and every document's token count matches. A failure is a loading bug.
2. **Float32: the implementation.** Every document's summed log-probability is within 1e-3 nats per window of OLMES's, and every domain's `bits_per_byte` is within 1e-5.
3. **Bfloat16: what we report.** Every domain's `bits_per_byte`, and every source's mean, within a tolerance fixed before the full comparison.

**Calibration first,** as D22 was fixed. `c4_en`, `wikitext_103` and `ptb` run on both sides in both dtypes (four dev jobs). The bfloat16 tolerance is the larger of the largest observed per-domain gap (rounded up to one significant figure) and our same-commit run-to-run noise on the same three tasks, with D22's 3e-4 as the ceiling. It is written here and frozen, and only then does the full first version run.

**Calibrated and frozen (2026-10-25): check 3's tolerance is 2e-4 bits per byte**, per domain and per source. On `allenai/OLMo-2-0425-1B` at `a1847dff`, the per-domain bfloat16 gaps between flock and OLMES were 9.7e-7 (`c4_en`), 2.7e-5 (`wikitext_103`) and 1.2e-4 (`ptb`), so the largest rounds up to 2e-4. Two flock bfloat16 runs on one commit were bit-for-bit equal, so the run-to-run noise adds nothing. D22's 3e-4 ceiling is not reached. The same runs passed checks 1 and 2 on `wikitext_103` and `ptb`. On `c4_en`, 3 of 2,124 documents missed check 2's 1e-3 nats per window in float32 (1.46e-3, 1.40e-3 and 1.34e-3, each one window). Scored alone they are 4.9e-4, 1.2e-4 and 3.6e-4 nats from OLMES, so the gap is our left-padded batching in float32, not the windows; the domain's bits per byte is within 2.5e-8. The rule is not changed.

**The verdict rests on checks 1 and 2.** Check 3 is reported pass or fail for every domain, next to each side's bfloat16 drift from OLMES's float32 run, but a fail there doesn't keep a domain from being used. OLMES sums each window's bfloat16 log-probabilities in bfloat16 (`_loglikelihood_tokens_verbose` in `oe_eval/models/eleuther_huggingface.py`, the multiple-choice path), so its bfloat16 numbers are rounded values, while the two float32 runs share one answer. Our own bfloat16 drift from float32 is a domain's noise floor: a difference between two checkpoints smaller than it is not a result.

---

## 4. Plan

Each PR is 200 to 400 lines of code and one idea. Builds on `feat/eval-suites` (D18), since `Suite` carries the per-source mean.

1. **Core (006.D2 to 006.D6).** `rolling_loglikelihood`, `PerplexityTask`, `score_task` for a perplexity task, its predictions rows, `bits_per_byte` as a suite scale, and `Suite` and `evaluate` taking either kind of task. One source, `wikitext_103`, lands as the example. Check: every multiple-choice score and hash unchanged on the CPU.
2. **Sources (006.D7).** The other 8 sources and their domain lists, as data, with the parity table (006.D8) in the PR body and the user docs.

---

## 5. Tests

All on CPU with the tiny random model, no downloads.

- **Windows:** a document shorter than the window, exactly one window long, one token longer, and over two windows. Each gives the same summed log-probability as explicit unbatched forward passes over the windows section 2.2 defines, and every token is scored exactly once.
- **Batching:** documents of mixed lengths score the same at several `batch_tokens` budgets, and in any order.
- **Bytes:** multi-byte UTF-8 text (Chinese, emoji) counts bytes, not characters.
- **Metric:** `bits_per_byte` on hand values is the summed ratio, not a mean of per-doc ratios.
- **Suites:** a source's score is the mean of its domains; a suite mixing `bits_per_byte` and `bpb` raises.
- **Raises:** empty text, a window above `max_length`, a window below 1.
- **Domains:** `domain` keeps only its rows, and `None` keeps all of a source with one domain (the build also raises on `None` for rows that have domains, so they can't be pooled).

---

## 6. Open questions

- **Traditional Chinese.** A Traditional Chinese domain set is its own task with its own name, built later. Candidates, none checked yet for licence or for overlap with our pretraining corpus: held-out zh-TW Wikipedia articles (CC BY-SA, but our corpus includes zh-Wikipedia), DRCD's passages (CC BY-SA 3.0, Wikipedia-derived, already used by the DRCD gold-answer task), and Taiwan government open-data text (Open Government Data License). Any converted or translated text gets a name of its own.
- **A second model for rankings** (as D23): worth running on Paloma, or are per-domain levels enough?
- **The other 7 sources**, once their licences are settled for our use.

---

## 7. How it was actually built

### PR 1: the core and `wikitext_103`

- **`ScoringModel.rolling_loglikelihood`** (`model.py`) builds the windows in one loop over each text's ids: the first window is `([prefix], ids[:window])`, and each later window starting at `start` is `(ids[end - window - 1 : start], ids[start:end])`, which is what `make_disjoint_window` makes of `get_rolling_token_windows(context_len=1)`. A test compares the windows with a copy of lm-evaluation-harness's two functions for windows of 1 to 64 tokens. The windows go through `_logprobs`, the batching loop moved out of `loglikelihood` unchanged (every multiple-choice score, both task hashes and every output file are bit-for-bit equal before and after the move, on the tiny model on the CPU). Per-text sums are float64 (006.D3).
- **`PerplexityTask`** and **`TextDoc`** (`tasks/perplexity.py`), as 006.D2. `load` raises when a domain has no rows, so a misspelt domain can't score 0 bits over 0 bytes. The hash is `task_hash(docs)`, over the definition and every doc's id and text.
- **`score_task`** sends a perplexity task to `_score_perplexity` in `run.py`, which divides the summed log-probability by the summed bytes on the device and copies one number to the host. `truncation` is all zeros.
- **Predictions.** `write_predictions(path, rows)` now takes rows, so one writer serves both kinds of task: `prediction_rows` for multiple choice and `perplexity_prediction_rows` for perplexity (`id`, `logprob`, `num_tokens`, `num_bytes`, `windows`). This and the `Task` alias below were approved as public-surface changes (Harry, 2026-10-25).
- **`Suite`** takes `Task | Suite` members, where `Task = MultipleChoiceTask | PerplexityTask` (`tasks/suite.py`), and its scale check names three scales: `accuracy`, `bpb` and `bits_per_byte`.
- **Paloma** (`tasks/paloma.py`): `DOMAINS` maps each source to its domains, and `source_suite` builds a source's tasks and suite. PR 1 lists `wikitext_103` only.
- **The census** (`scripts/paloma_census.py`) reads the raw files at the pinned revision with OLMo 2's tokenizer, writes the counts of section 2.3, and prints each source's domains in `DOMAINS`'s format. Its output equals the design census in every count, for all 16 sources.
- **Loading cost** of one split per domain (006.D2) is measured in PR 2, which adds the sources with 99 and 100 domains.

**Checks on `allenai/OLMo-2-0425-1B` at `a1847dff`**, OLMES at batch size 1 and `max_length` 2048, both on `test` at `65cd6fc5`:

| `paloma_wikitext_103` | flock | OLMES |
| --- | --- | --- |
| docs, windows | 61, 170 | 61, 170 |
| check 1 (ids in order, texts, tokens, windows, bytes) | equal | |
| float32 bits per byte | 0.7064185 | 0.7064184 |
| check 2: most nats per window, bits per byte difference | 7.3e-4, 3.5e-8 (pass) | |
| bfloat16 bits per byte | 0.7065046 | 0.7064772 |
| bfloat16 drift from OLMES's float32 | 8.6e-5 | 5.9e-5 |

The bfloat16 run took 5.2 seconds for 285,336 tokens on one H200. Check 3's tolerance is set by PR 2's calibration.

### PR 2: the other 8 sources and parity

- **Data.** `DOMAINS` lists all 9 sources and their 309 domains, in Paloma's rank order, as the census prints them.
- **Loading.** One split takes 1.6 to 2.6 seconds to load from the cache for the 100-domain sources, so a run that loaded it once for each of their 299 domains would spend about 8 to 13 minutes loading. `PerplexityTask.load` keeps the last split it loaded, grouped by domain as tuples of frozen docs (`functools.lru_cache(maxsize=1)` on the frozen `Dataset`), and the domains of a source run one after another, so each split loads once per run. A run that interleaves the domains of two sources loads their splits again each time; suites never do.
- **Speed.** The full first version, 77,746 documents in 81,761 windows, scores in 424 seconds on one H200 in bfloat16.

**Parity** (006.D8, all 309 domains; `allenai/OLMo-2-0425-1B` at `a1847dff`, OLMES at `5a51f50`, batch size 1, `max_length` 2048, `test` at `65cd6fc5`):

| Source | Domains | float32, flock and OLMES (mean of domains) | bfloat16, flock / OLMES | Check 3 (source) | OLMES's own number (all docs pooled), float32 |
| --- | --- | --- | --- | --- | --- |
| `c4_en` | 1 | 0.852881 | 0.852939 / 0.852940 | pass | 0.852881 |
| `mc4` | 1 | 0.877680 | 0.877719 / 0.877656 | pass | 0.877680 |
| `wikitext_103` | 1 | 0.706418 | 0.706505 / 0.706477 | pass | 0.706418 |
| `ptb` | 1 | 0.867739 | 0.867690 / 0.867570 | pass | 0.867739 |
| `falcon-refinedweb` | 1 | 0.884158 | 0.884205 / 0.884115 | pass | 0.884158 |
| `dolma-v1_5` | 5 | 0.770683 | 0.770731 / 0.770726 | pass | 0.779772 |
| `c4_100_domains` | 99 | 0.803331 | 0.803370 / 0.803368 | pass | 0.799736 |
| `dolma_100_subreddits` | 100 | 0.988086 | 0.988198 / 0.988106 | pass | 0.987239 |
| `dolma_100_programing_languages` | 100 | 0.610011 | 0.610088 / 0.610049 | pass | 0.585758 |

1. **Same documents: pass in all 309 domains.** Ids in order, token counts, windows and bytes equal OLMES's everywhere, and the texts equal OLMES's saved requests in 8 sources. OLMES can't save requests for `dolma_100_programing_languages`: its `timestamp` column reaches OLMES as a `datetime`, which OLMES's JSON writer refuses, so there the texts aren't compared, and ids, token counts and bytes are.
2. **Float32.** Every domain's bits per byte is within 2.0e-7 of OLMES's (the rule is 1e-5). The per-document rule, 1e-3 nats per window, misses on 137 of 77,746 documents in 87 domains, at most 8.0e-3; 133 of the 137 are one window long. Four of the worst were scored alone, one window per forward pass as OLMES does. Two come within 1e-3 (`46_childfree` `1d3265fc`: 3.5e-3 batched, 7.9e-4 alone; `62_asktransgender` `6a4c48d3`: 4.3e-3, 3.7e-4), and one comes to 1.2e-3 (`16_www.ncbi.nlm.nih.gov` `9849bcc8`: 5.1e-3 batched), so most of the gap is our left-padded batching in float32, as the math and code parity found for one multiple-choice request. One document (`16_www.ncbi.nlm.nih.gov` `1b74b3bc`, 1,195 tokens, one window) stays 4.0e-3 nats from OLMES alone, so batching doesn't explain it, and it is recorded here unexplained. The rule is not changed.
3. **Bfloat16, tolerance 2e-4.** Every source's mean passes (at most 1.2e-4). 44 of the 309 domains don't (at most 1.3e-3), mostly small ones, and in 38 of those 44 OLMES's own bfloat16 result is further from the float32 answer than ours. The largest drift from float32 is 4.5e-4 for flock and 1.1e-3 for OLMES. A domain's own drift is its noise floor.

Where the paper and OLMES differ (section 2.2), the table shows it: for a source with several domains, OLMES's pooled number differs from the mean of the domains by up to 0.024 bits per byte (`dolma_100_programing_languages`).
