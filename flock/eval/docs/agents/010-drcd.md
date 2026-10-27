# 010: DRCD, Traditional Chinese reading comprehension as gold-answer BPB

| | |
| --- | --- |
| Status | designed; the build waits for the dataset mirror |
| Date | 2026-10-27 (design) |
| Builds on | [001](001-eval-core.md) (scoring core), [002](002-truncation-count.md) (truncation count), [003](003-base-easy.md) (SQuAD's passage format, D16, D17); 005 (the Chinese format, 005.D1, 005.D7) and 007 (`GoldOnlyTask`, 007.D2), both before this stage merges |
| Keeps | 001's scoring, batching, metrics and results file; 003's `Choice`, `MCDoc` and `Fewshot`; 005's `cloze` arguments and Chinese prefixes; 007's `GoldOnlyTask` |
| Result | (design) one task, `drcd_bpb`, 3,493 docs; no core change |

The file number and the decision numbers are provisional. The file takes the next free number when its PR is opened, and decisions `010.D1` to `010.D6` take the next free global numbers after every stage merged before it.

## 1. Goal

1. **A native Traditional Chinese reading-comprehension task that separates small models.** DRCD (Shao et al. 2018, arXiv 1806.00920) is extractive QA over Traditional Chinese Wikipedia passages, built the way SQuAD is. TMMLU+ accuracy sits near chance below about 1B (005, section 7), so this task gives a Chinese curve from the smallest checkpoints up.
2. **Scored as OLMo 3 scores SQuAD and DROP**: bits per byte of the gold answer after the passage and the question (OLMo 3, arXiv 2512.13961, Table 45: "SQuAD (2016) General QA 5 BPB" and "DROP (2019) Passage QA 5 BPB").
3. **No new code path.** The task is one `to_doc`, one `Dataset` and one `Fewshot` on top of 005's format and 007's `GoldOnlyTask`.

**Out of scope:** generative exact match or F1, a multiple-choice version (no gen2mc set exists for DRCD), decontamination against a pretraining corpus, and the training and dev splits as scored tasks.

---

## 2. The dataset

DRCD is published on GitHub only, as SQuAD-style JSON (`DRCKnowledgeTeam/DRCD` at `b944790de5af02c5fbb7cd9cb1473d27d169eebf`, 2020-04-21), under CC-BY-SA 3.0 (its README). Counted at that commit:

| | Articles | Paragraphs | Questions | Answers per question | Questions whose answers differ |
| --- | --- | --- | --- | --- | --- |
| `DRCD_training.json` | 1,960 | 8,014 | 26,936 | 1 | 0 |
| `DRCD_dev.json` | 383 | 1,000 | 3,524 | 2 | 774 |
| `DRCD_test.json` | 378 | 1,000 | 3,493 | 2 | 686 |

- **Answers are short spans.** The first answer has a median of 4 characters (at most 46 in test), always sits at its `answer_start` in the passage, and never has outer whitespace. Passages have a median of 408 characters in test, and no passage or question contains a newline. Two test passages carry outer whitespace.
- **Question ids are unique** and encode their place: `1172-3-2` is article 1172, paragraph 3, question 2.
- **The splits share articles, not paragraphs.** 255 of the 378 test articles also appear in dev and 286 in training, each time with other paragraphs. SQuAD's splits are article-disjoint; DRCD's are paragraph-disjoint only.
- **Titles are often in Simplified script.** They are Wikipedia page names (`国际动物命名法规`, `城市规划`, `耶稣会` in test): 173 of the 378 test titles change under OpenCC's `s2t`. Passages, questions and answers are Traditional Chinese.

---

## 3. Design decisions

### 010.D1. A pinned parquet mirror at `cerulean-works/drcd`

`Dataset` reads a Hugging Face repo at a revision, and DRCD has no canonical one. The three JSON files are mirrored to the public dataset repo `cerulean-works/drcd`, as `allenai/basic-skills` is mirrored to `cerulean-works/basic-skills` (006). CC-BY-SA 3.0 permits this with attribution and under the same licence. The mirror holds:

- One parquet file per split, `train` (`DRCD_training.json`), `validation` (`DRCD_dev.json`) and `test`, with one row per question in SQuAD's Hugging Face schema (`rajpurkar/squad`): `id`, `title`, `context`, `question`, and `answers` with `text` and `answer_start` lists in file order. DRCD's per-answer `id` is dropped: it is always the answer's position (`"1"`, `"2"`), which the list order keeps.
- The three original JSON files unchanged, with their sha256, under `original/`.
- A README with the attribution to Shao et al. 2018, the source repo and commit, the licence text, and the conversion script.

The task pins the mirror's commit. Flattening at upload keeps `Dataset` unchanged: loaded raw, the nested JSON gives one row per article, and splitting it into questions would need new core code for one dataset.

### 010.D2. The format: passage, question, answer, with 005's Chinese prefixes

A `drcd_bpb` request reads:

```text
文章：<passage>
問題：<question>
答案：<answer>
```

The continuation is the first gold answer with no leading space, and the context is everything before it. This is SQuAD's format in 003 (`Title: <title>\nPassage: <passage>\nQuestion: <question>\nAnswer:`, OLMES's `squad_mc`) with 005.D1's Chinese prefixes and its no-space rule, and one change of our own: the title line is dropped.

- **`文章：` for the passage** is the passage prefix of every DRCD template in OpenCompass (`configs/datasets/CLUE_DRCD/`, four configs, at `7562df99`: `文章：{context}`). OpenCompass's question lines are instructions in Simplified script (`根据上文，回答如下问题：`) for generation, so we keep 005's `問題：` and `答案：` and one format across our Chinese tasks.
- **No title.** Titles are page names, often in Simplified script (section 2), so a title line would put Simplified text into a Traditional Chinese prompt on 1,425 of the 3,493 docs. The `title` column stays in the mirror for anyone who wants it.
- **No leading space** keeps the context/continuation boundary clean. With `答案：` and no space, the context's tokens are a prefix of the joint tokenization for all 3,493 test docs under both `allenai/OLMo-2-0425-1B` (`a1847dff`) and `Qwen/Qwen2.5-0.5B` (`060db649`), as 005.D1 found on TMMLU+.
- **Outer whitespace** is stripped from the passage, the question and the answer, as 005.D1 strips TMMLU+'s.

**Code.** `drcd_doc` calls 005's `cloze(question, (answer,), question_prefix=f"文章：{passage}\n{CHINESE_QUESTION_PREFIX}", answer_prefix=CHINESE_ANSWER_PREFIX, separator="")`, the way 003's `squad_doc` puts its passage line in `question_prefix`.

### 010.D3. One example, not five

OLMo 3 shows SQuAD and DROP five examples. DRCD passages are long in tokens: OLMo 2's tokenizer reads 0.69 characters per token on DRCD's test requests (Qwen 2.5's reads 1.22), so each example costs about 416 tokens under OLMo 2 and 240 under Qwen. Rendered with the format above:

| Examples | OLMo 2, longest request | OLMo 2, requests over 2,048 tokens | Qwen 2.5, longest | Qwen 2.5, over 2,048 |
| --- | --- | --- | --- | --- |
| 0 | 1,489 | 0 | 881 | 0 |
| 1 | 1,905 | 0 | 1,121 | 0 |
| 5 | 3,941 | 3,493 | 2,205 | 105 |

The project's own checkpoints train at a 2,048-token context, as OLMo 1B's recipe does. With five examples, such a model would see every doc cut from the left, and how much is cut depends on its tokenizer, so two models in one comparison would read different prompts. With one example, every request fits in 2,048 tokens under both tokenizers, and the shot still fixes the format for a base model. This departs from OLMo 3's five. 002's truncation count reports any model whose tokenizer still overflows.

### 010.D4. The example: one dev question from an article test doesn't share

`Fewshot(split="validation", ids=("1172-3-2",))`. The rule that picks it: in dev file order, skip articles that also appear in test, and take the first question whose two answers agree. That gives article 1172 (溫度), paragraph 3, question 2, a 263-character passage, whose answer is `攝氏溫標`.

- **From dev, not test or training**, so no scored doc is its own example and the example has two annotators. OLMES's SQuAD examples come from SQuAD's training split, but DRCD's training split has one annotator per question.
- **An article test doesn't share**, so no doc sees another paragraph of its own article as the example (section 2). 128 of the 383 dev articles qualify.
- **Answers that agree.** The first question in that article's first paragraph has two different answers (`加布里埃爾·華倫海特` and `奧勒·羅默`), and an example whose gold is disputed teaches the wrong span.

005.D7's rule (an example from another split isn't shown to a doc whose gold choice equals its own) never fires here: dev and test share no paragraph, so no rendered gold choice can match.

### 010.D5. The first gold answer, scored by `bpb` on `GoldOnlyTask`

Each test question has two annotated answers, and they differ on 686 of the 3,493. The doc scores the first one, as SQuAD's gold-answer BPB scores one gold text per question. Taking the minimum or mean over both answers would make a doc's score depend on how many answers it has. The task is a `GoldOnlyTask` (007.D2), so `bpb` is the only metric, there is no unconditional context, and `bpb` divides by the UTF-8 bytes of the answer, 3 per CJK character. The doc id is the question id. Below about 300M parameters `bpb` is the metric to decide on (OLMo 3, sec. 3.3.2), which is why this task exists.

### 010.D6. Parity against lm-evaluation-harness, rendering our requests

OLMES has no DRCD task and no Chinese task (005.D6). As for TMMLU+, the reference is lm-evaluation-harness v0.4.3 from the locked environment that runs OLMES (never installed in or imported by flock), with a generated config that renders exactly our request: `doc_to_text` the context above with the example as a literal `description`, `doc_to_target` the first answer, `target_delimiter: ""`, `output_type: loglikelihood`. The rule, fixed before any run, is 005.D6's checks 1 and 2 with 003's D22 tolerances:

1. **Same requests.** lm-evaluation-harness's logged `arguments` equal ours string for string, in doc order.
2. **Float32.** Every request within 1e-3 nats, read with 007's correction to D22: a request over it that falls back under it when scored alone is our batching, and the note says which requests needed that. `bpb` within 1e-4, computed on its side from its logged log-likelihoods and the answer's UTF-8 bytes.

The model is `Qwen/Qwen2.5-0.5B` at `060db6499f32faf8b98477b0a26969ef7d8b9987` in float32, one job per side. The only new code is the rendering, so this checks the request strings above all.

---

## 4. Plan

One PR, after 005 and 007 are on `main` (it imports `cloze`'s Chinese arguments and `GoldOnlyTask`):

1. **Mirror** ([010.D1](#010d1-a-pinned-parquet-mirror-at-cerulean-worksdrcd)): convert and upload, check the row counts above, pin the commit.
2. **Task:** `flock/eval/tasks/drcd.py` with `drcd_doc` and `DRCD_BPB`, registered as `drcd_bpb`; an entry on the tasks docs page.
3. **Smoke** on `allenai/OLMo-2-0425-1B` and `Qwen/Qwen2.5-0.5B` (pins above), one dev-partition job each, bfloat16: 3,493 docs, no truncated request, and a lower `bpb` for Qwen, which was trained on far more Chinese.
4. **Parity** ([010.D6](#010d6-parity-against-lm-evaluation-harness-rendering-our-requests)): two float32 jobs, and the PR reports the result.

---

## 5. Tests

On CPU with tiny random models, as every eval test, with hand-written rows in the mirror's schema:

- **Rendering:** a row renders to exactly `文章：<passage>\n問題：<question>\n答案：` and the continuation `<answer>` with no space, after the one example and a blank line; outer whitespace is stripped from all three fields.
- **The first answer:** a row whose two answers differ scores the first.
- **Multi-byte answers:** `bpb` divides by UTF-8 bytes on an all-CJK answer, a mixed answer such as `18世紀`, and an ASCII one.
- **Long contexts:** a doc whose request is longer than the tiny model's `max_length` is cut from the left, counted by 002's truncation count, and keeps its full continuation; one that fits exactly is not counted.
- **Batch invariance:** `bpb` is the same at two `batch_tokens` settings, with left and right padding in the mix.

---

## 6. Open questions

- Whether a 5-shot variant earns a separate task name for models with a 4,096-token context or more (Principle 3: a different setting is a different task).
- Whether DRCD's dev split, 3,524 questions disjoint in paragraphs from test, should be scored as a second, development-only task so that tuning never reads test.
