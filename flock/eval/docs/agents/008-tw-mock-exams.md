# 008: Taiwanese mock exams, an internal Traditional Chinese task

| | |
| --- | --- |
| Status | design |
| Date | 2026-10-26 (design) |
| Builds on | [001](001-eval-core.md) (scoring core); 003 (`Choice`, `MCDoc`, `Fewshot`, `Suite`); [004](004-tmmluplus.md) (the Chinese format, `row_index`, the few-shot repeat rule); stacked on 003's suites PR |
| Keeps | 001's scoring, batching, metrics and results file; 004.D1's format and 004.D7's rule unchanged |
| Result | (after the build) 10 subject tasks, 3 suites, 18,970 docs |

The file number and the decision numbers are provisional. The file takes the next free number when its PR is opened, and decisions `008.D1` to `008.D7` take the next free global numbers after every stage merged before it.

## 1. Goal

1. **Native exam questions as data.** Single-answer multiple-choice items from Taiwanese senior-high mock exams for 學測 (GSAT) and 指考 / 分科 (AST), 2009 to 2025, scored as 004 scores TMMLU+: cloze rank classification, `acc_per_char` primary, gold-answer `bpb` beside it, 5-shot.
2. **Named for what it is.** The papers are regional, nationwide and publisher mock exams, not CEEC's official papers (286 of 27,744 items are CEEC's), so the tasks are `tw_mock_exams_*`, never `gsat`.
3. **Internal only.** The data is read from a private dataset repo, never a public one, and never stored in this repo. Anyone can read the task definition; only accounts with access to the dataset can run it.

**Out of scope:** multi-answer items (多選), English cloze and listening sections, items that need a figure, a public release, a reference implementation, and pretraining-corpus decontamination.

---

## 2. The data

The source is a shared folder of 2,416 PDFs (29,848 pages). Each page was transcribed with `Qwen/Qwen3.5-27B` (`fc05daec`) and structured into items with one schema-constrained call per paper. Answers come from the paper's own key or a paired answer document, parsed from the transcription rather than generated, and every field must occur in the paper's transcription. The character error rate is at most 0.98% on 103 digital pages checked against their text layer (91,305 Han characters). Groups whose text contains a picture placeholder were dropped whole. The result is 27,744 items from 983 papers.

| Field | Meaning |
| --- | --- |
| `hash` | a hash of the normalized item, unique, used as the doc id |
| `doc_subject` | 國文, 英文, 社會, 自然, 歷史, 地理, 公民 / 公民與社會, 物理, 化學, 生物 |
| `rights` | `official_exam` 286, `school_exam` 19,883, `publisher_copyright` 3,722, `unknown` 3,853 |
| `passage` | the shared text of a 題組, empty for a standalone item |
| `stem`, `options`, `answer` | the question, `[{label, text}]` with 4 or 5 options (up to 12 in English word banks), and the gold letter(s) |
| `type` | 單選 23,503, 多選 4,241 |
| `char_sensitive` | 245 items about 錯別字, 字音 or 字形, whose answer depends on characters an OCR model may silently correct |
| `section` | the paper's section heading, e.g. 一、詞彙題, 二、綜合測驗 |

Overlap with TMMLU+ (`ikala/tmmluplus` v1.1, 21,483 questions) is small: no stem of 20 or more characters matches exactly, and 6 items share at least half their character 13-grams with it, the test GPT-3 used (Brown et al. 2020, app. C). Mock exams are often posted online, so contamination with web pretraining data is likely and is not measured here.

---

## 3. Design decisions

### 008.D1. A private dataset repo, pinned, one config per subject

`Dataset` reads a Hugging Face repo at a revision, and a private repo needs no code change: the run's environment carries a token with read access. The repo holds every item, in ten configs (one per subject, section 3 of [008.D5](#008d5-ten-subject-tasks-and-three-suites)) with a `dev` split (the 5 few-shot items, [008.D4](#008d4-five-shots-the-shortest-self-contained-items)) and a `test` split (every other item, eligible or not). Nothing is deleted: which items are scored is the task's choice ([008.D2](#008d2-which-items-are-scored)), so a later task can score more of them under another name. The three flags of 008.D2 are added as fields when the repo is built, so the task reads them instead of re-deriving them.

A local-file source was the alternative. It needs a new `Dataset` kind pinned by a file hash, runs only where the files are, and is a public-surface change for one task. If the data moves to object storage later, that is a new `Dataset` source, and only the `dataset=` line of this task changes. CI never loads the data: the tests use rows written in the test.

### 008.D2. Which items are scored

A doc is scored when its row passes all of:

| Rule | Why | Items it removes, in this order |
| --- | --- | --- |
| `type == "單選"` | single-answer only, as TMLU (Chen et al. 2024, arXiv 2403.20180) and TMMLU+ | 4,241 |
| not `char_sensitive` | the answer may have been altered by the OCR | 204 |
| not `listening` | English 對答, 簡短對話 and 聽解 sections: the audio isn't in the paper | 380 |
| not `cloze` | English 綜合測驗, 文意選填, 篇章結構 and 克漏字 sections, and grouped English items whose stem is the item's own blank number: the answer fills a numbered blank inside the passage, which a question/answer prompt can't show | 3,494 |
| not `figure_ref` | the text refers to a figure (下圖, 如圖, 圖 3, ...) that a text-only model can't see, as TMLU excludes "unanswerable questions that require accompanied images" (sec. 3.3, p. 5) | 405 |

All four `rights` classes are scored. The rights field stays on every row so a version for outside use can filter on it.

That leaves 19,020 items, 18,970 after the 50 few-shot items. Cloze items could be scored later by substituting each option into the passage, as OLMES scores Winogrande; that is a separate task with its own name.

**Core change: `MultipleChoiceTask.keep`.** An optional `Callable[[dict[str, Any]], bool]`, applied to the scored split's rows before `limit` and `to_doc`. lm-evaluation-harness has the same hook (`process_docs` in its task config, v0.4.3, `lm_eval/api/task.py`). Doc ids are the rows' `hash`, so filtering never shifts an id. The predicate is a function like `to_doc`, so it isn't part of the task hash: what it selects is, since the hash covers every doc and request. Without `keep`, the repo would need a copy of each split already filtered, and a second task over the same data a second copy.

### 008.D3. The prompt: 004.D1 for Chinese subjects, OLMES's for English

The nine Chinese subjects use 004.D1 unchanged: the TMMLU+ header with the subject's Chinese name, `問題：` and `答案：` with no space before the answer, stripped text, and `答案：` as the unconditional context. A grouped item puts its passage on the line before the question, as OLMES's BoolQ does (`question_prefix=f"{passage}\nQuestion: "`, `oe_eval/tasks/oe_eval_tasks/boolq.py`, `5a51f502`):

```text
以下是關於國文考試單選題，請選出正確的答案。

<5 examples, each followed by a blank line>
<passage>
問題：<stem>
答案：<option text>
```

英文 is English text, so it uses OLMES's English format: the MMLU header ("The following are multiple choice questions (with answers) about English.\n\n", `oe_eval/tasks/oe_eval_tasks/mmlu.py:128`), `Question: ` and `Answer:` with the answer after a space, and `Answer:` as the unconditional context. 004.D1 drops the space because a space before a Chinese answer splits its first character into bytes; for English answers the space is how the words tokenize.

Options aren't shown in the prompt: in OLMES's cloze (RC) formulation each option's text is the continuation. Items whose options only make sense next to each other (以上皆是, or combinations such as 甲乙) are kept, as OLMES keeps them in MMLU's cloze form; the 5-option papers of the sciences stay 5-option.

### 008.D4. Five shots: the shortest self-contained items

TMLU holds out "a development set consisting of five instances" per subject (sec. 3.3, p. 5) and shows them as five shots (sec. 4.1); there is no dev split here, so we hold out five too. Which five matters more than usual: OLMo 2's tokenizer spends about 1.5 tokens per Han character, and 國文's items carry their reading inside the stem. With five arbitrary items (lowest `hash`), 國文's examples alone are about 1,900 tokens and 2,543 of its 2,836 docs exceed 2,048 tokens, so a 2,048-token model would see them cut from the left (002 counts this) and score a different prompt from a 4,096-token model.

So each subject's examples are its five **shortest self-contained items**: eligible under 008.D2, no passage, a stem that doesn't point outside itself (上文, 承上題, 承第 9 題, 資料甲, 表中, ...), shortest by stem plus gold answer, ties by `hash`. Three of the shortest were read and skipped by hand, because each refers to a figure or setup it doesn't carry (`22e4706b`, `4c861f04`, `c18c5661`). The 50 ids are written into the task module, in this order, as 004 writes TMMLU+'s dev rows. No prior work picks examples by length, so this is our choice. An example's job in OLMES's cloze format is to show the format, which a short item does as well as a long one, at the cost that the examples are easier and shorter than the docs.

| | 2,048-token prompts exceeded, OLMo 2 | Qwen 2.5 |
| --- | --- | --- |
| Five by `hash` | 2,543 國文, 278 地理, 119 other | 157 |
| Five shortest (this decision) | 56 國文, 22 other | 0 |

Counted on the longest choice of each doc, with the tokenizers of `allenai/OLMo-2-0425-1B` (`a1847dff`) and `Qwen/Qwen2.5-0.5B` (`060db649`). 004.D7's rule applies unchanged; no test doc repeats an example's question and answer.

### 008.D5. Ten subject tasks and three suites

One task per subject, `tw_mock_exams_<subject>`, generated from a table of `(name, config, Chinese name)` like 004's:

| Task | Subject | Docs | Passage groups | Options (most docs) |
| --- | --- | --- | --- | --- |
| `tw_mock_exams_english` | 英文 | 3,713 | 515 groups | 4 |
| `tw_mock_exams_social` | 社會 (學測, combined) | 3,660 | 773 | 4 |
| `tw_mock_exams_chinese` | 國文 | 2,836 | 716 | 4 |
| `tw_mock_exams_science` | 自然 (學測, combined) | 1,869 | 291 | 5 |
| `tw_mock_exams_history` | 歷史 | 1,767 | 95 | 4 |
| `tw_mock_exams_civics` | 公民與社會 (公民 is its short name) | 1,730 | 366 | 4 |
| `tw_mock_exams_biology` | 生物 | 1,138 | 200 | 4 |
| `tw_mock_exams_geography` | 地理 | 953 | 347 | 4 |
| `tw_mock_exams_physics` | 物理 | 729 | 32 | 5 |
| `tw_mock_exams_chemistry` | 化學 | 575 | 37 | 5 |

`tw_mock_exams` is the unweighted mean of the ten subjects' `acc_per_char`, as OLMES averages MMLU's subjects; `tw_mock_exams:bpb` is the same over `bpb`, the metric to decide on for small models (OLMo 3, sec. 3.3.2; 004's smoke found Qwen 2.5 0.5B one point above chance on TMMLU+ accuracy); `tw_mock_exams:all` groups the two and is unscored, as `tmmluplus:all`. The smallest subject has 575 docs, a standard error of about 2 points near 50%, so subjects are reported and the mean decides.

### 008.D6. Pin and scope

The dataset repo and its revision are written in the task module once the repo exists; its rows are the item set above, `test` split, all ten configs.

### 008.D7. Parity: no reference exists, so three self-checks

No implementation scores this data: it is new. The scoring path is 004's, which matches lm-evaluation-harness in float32 on TMMLU+; this stage adds a row filter, a passage line and an English variant of the format. What we can check:

1. **Requests, against an independent renderer.** lm-evaluation-harness v0.4.3 (outside flock, as for 004), with a config per subject: the same repo and revision, `process_docs` applying 008.D2's rule, our header as `description`, the passage line and prefixes in `doc_to_text`, the stripped option texts as `doc_to_choice`, `target_delimiter` `""` (Chinese) or `" "` (English), and our five examples. Its logged `arguments` must equal our requests, string for string, in doc order. No model is needed.
2. **Batching.** On `Qwen/Qwen2.5-0.5B` in float32, every request whose log-probability moves by more than 1e-3 nats between our batched run and a batch of one must land under 1e-3 when scored alone, the rule 004 and every later stage use.
3. **Smoke and noise floor.** `allenai/OLMo-2-0425-1B` and `Qwen/Qwen2.5-0.5B` score all ten tasks in bfloat16 and float32. The report gives each task's `acc_per_char`, `bpb`, truncated requests and the bfloat16-to-float32 drift, which is the task's noise floor: a difference between checkpoints smaller than that is not a result.

There is no tolerance against a reference number, so these checks show the task is implemented as specified, not that the specification measures what a real exam measures.

---

## 4. Plan

One PR, stacked on 003's suites PR, after 004's:

1. **Data.** Build the private repo from the item set: ten configs, `dev` and `test`, the three flags, a card that states the source, the rights classes and that the repo is internal. Pin its revision.
2. **Core:** `MultipleChoiceTask.keep` ([008.D2](#008d2-which-items-are-scored)).
3. **Task:** `flock/eval/tasks/tw_mock_exams.py`: the subject table, `keep`, `to_doc` for both formats, the 50 example ids, the suites, and their registration; a docs page entry saying the data is internal.
4. **Checks** ([008.D7](#008d7-parity-no-reference-exists-so-three-self-checks)): the request comparison on CPU, then two smoke jobs on one H200 each and the batch-of-one rescoring.

---

## 5. Tests

On CPU with tiny random models and rows written in the test:
- **Rendering:** a standalone and a grouped Chinese row, and an English row, render to the exact strings above; no space before a Chinese answer, one before an English answer.
- **Filter:** `keep` drops multi-answer, char-sensitive, listening, cloze and figure rows, keeps the rest whatever their `rights`, and applies before `limit`; ids are the rows' hashes.
- **Few-shot:** the examples appear in the listed order; a doc never sees an example from `dev` with its own question and answer.
- **Lengths:** `acc_per_char` divides by characters and `bpb` by UTF-8 bytes on a Chinese answer and on an English one with its leading space.
- **Suites:** `tw_mock_exams` is the plain mean of the ten subjects on hand values; the table has ten distinct subjects.

---

## 6. Open questions

- **Follow-up items without their setup.** 57 test docs have no passage and a stem that continues another question (承上題, 續前題, 承第 9 題): 16 in 自然, 15 in 物理, 7 each in 社會 and 公民. Some carry enough to answer, some don't. Leave them in (as now), or flag and drop them like figures.
- **學測 and 指考 apart.** Mock 學測 and mock 指考 differ in level; a split by `doc_exam` would give two suites. Not done until a decision needs it.
- **A reading suite.** 英文 and 國文's grouped items are reading comprehension (515 and 716 groups). They are inside their subjects now.
