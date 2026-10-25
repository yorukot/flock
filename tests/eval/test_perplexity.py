import dataclasses
import json
import math

import pytest

from flock.eval import run
from flock.eval.model import ScoringModel
from flock.eval.run import EvalConfig, evaluate, score_task
from flock.eval.tasks import SUITES, TASKS, PerplexityTask, Suite
from flock.eval.tasks.paloma import PALOMA, source_suite

from .toy import TASK, TEXT_ROWS, TEXT_TASK

NEWS = dataclasses.replace(TEXT_TASK, name="toy_text__news", domain="news")
EXAM = dataclasses.replace(TEXT_TASK, name="toy_text__exam", domain="exam")
SOURCE = Suite("toy_text:source", (NEWS, EXAM))


def test_load_keeps_the_domains_rows_in_order():
    assert [doc.id for doc in TEXT_TASK.load()] == ["t-1", "t-2", "t-3"]
    assert [doc.id for doc in NEWS.load()] == ["t-1", "t-3"]
    assert [doc.id for doc in NEWS.load(limit=1)] == ["t-1"]
    assert NEWS.load()[1].text == TEXT_ROWS[2]["text"]


def test_load_rejects_an_empty_domain_or_a_limit_below_one():
    with pytest.raises(ValueError, match="no rows in domain 'sports'"):
        dataclasses.replace(TEXT_TASK, domain="sports").load()
    with pytest.raises(ValueError, match="limit must be at least 1"):
        TEXT_TASK.load(limit=0)


def test_window_below_one_raises():
    with pytest.raises(ValueError, match="window must be at least 1"):
        dataclasses.replace(TEXT_TASK, window=0)


def test_task_hash_covers_the_definition_and_every_text():
    docs = TEXT_TASK.load()
    changed_text = [docs[0], dataclasses.replace(docs[1], text="玉山"), docs[2]]
    hashes = {
        TEXT_TASK.task_hash(docs),
        TEXT_TASK.task_hash(changed_text),
        dataclasses.replace(TEXT_TASK, window=16).task_hash(docs),
    }
    assert len(hashes) == 3
    assert TEXT_TASK.task_hash(TEXT_TASK.load()) == TEXT_TASK.task_hash(docs)


def test_bits_per_byte_is_a_ratio_of_sums_over_utf8_bytes(scorer):
    texts = [row["text"] for row in TEXT_ROWS]
    scores = scorer.rolling_loglikelihood(texts, TEXT_TASK.window)
    logprob = scores.logprob.tolist()
    num_bytes = [len(text.encode()) for text in texts]
    # Chinese and the emoji take more than one byte each.
    assert num_bytes[1] > len(texts[1])

    result = score_task(scorer, TEXT_TASK)

    expected = -sum(logprob) / (sum(num_bytes) * math.log(2))
    assert result.metrics == {"bits_per_byte": pytest.approx(expected, rel=1e-6)}
    mean_of_ratios = sum(-lp / (b * math.log(2)) for lp, b in zip(logprob, num_bytes)) / 3
    assert result.metrics["bits_per_byte"] != pytest.approx(mean_of_ratios, rel=1e-3)
    assert result.primary_metric == "bits_per_byte"
    assert result.num_docs == 3
    assert result.task_hash == TEXT_TASK.task_hash(TEXT_TASK.load())
    assert result.truncation.requests == 0


def test_score_task_does_not_depend_on_batching(scorer):
    one_batch = score_task(scorer, TEXT_TASK).metrics["bits_per_byte"]
    scorer.batch_tokens = 1
    assert score_task(scorer, TEXT_TASK).metrics["bits_per_byte"] == pytest.approx(one_batch)


def test_score_task_writes_one_prediction_row_per_doc(scorer, tmp_path):
    score_task(scorer, TEXT_TASK, out=tmp_path)

    lines = (tmp_path / "predictions" / "toy_text.jsonl").read_text().splitlines()
    rows = [json.loads(line) for line in lines]
    assert [row["id"] for row in rows] == ["t-1", "t-2", "t-3"]
    for row, source in zip(rows, TEXT_ROWS, strict=True):
        ids = scorer.tokenizer.encode(source["text"], add_special_tokens=False)
        assert row["num_tokens"] == len(ids)
        assert row["num_bytes"] == len(source["text"].encode())
        assert row["windows"] == math.ceil(len(ids) / TEXT_TASK.window)
        assert row["logprob"] < 0


def test_a_window_above_the_models_max_length_raises(lm, tokenizer):
    short = ScoringModel(lm, tokenizer, name="tiny@0", max_length=4)
    with pytest.raises(ValueError, match="window must be from 1 to max_length 4, got 8"):
        score_task(short, TEXT_TASK)


def test_a_source_scores_the_mean_of_its_domains(scorer):
    news = score_task(scorer, NEWS).metrics
    exam = score_task(scorer, EXAM).metrics
    metrics = {NEWS.name: news, EXAM.name: exam}
    expected = (news["bits_per_byte"] + exam["bits_per_byte"]) / 2
    assert SOURCE.score(metrics) == pytest.approx(expected)


GOLD_ONLY = dataclasses.replace(TASK, name="gold", metrics=("bpb",), primary_metric="bpb")


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        pytest.param({"members": (NEWS, GOLD_ONLY)}, r"\['bits_per_byte', 'bpb'\]", id="bpb"),
        pytest.param({"members": (NEWS, TASK)}, r"\['accuracy', 'bits_per_byte'\]", id="acc"),
        pytest.param({"members": (NEWS,), "scored": False}, "groups only suites", id="unscored"),
    ],
)
def test_suite_rejects_perplexity_tasks_beside_other_scales(kwargs, match):
    with pytest.raises(ValueError, match=match):
        Suite("mixed", **kwargs)


def test_evaluate_scores_perplexity_tasks_and_their_source(monkeypatch, scorer, tmp_path):
    group = Suite("toys", (SOURCE,), scored=False)
    monkeypatch.setitem(run.SUITES, group.name, group)
    monkeypatch.setattr(ScoringModel, "from_pretrained", lambda *args, **kwargs: scorer)
    results = evaluate(EvalConfig(model="tiny@0", suites=("toys",), out=tmp_path))

    data = json.loads((tmp_path / "results.json").read_text())
    assert [task["task"] for task in data["tasks"]] == [NEWS.name, EXAM.name]
    source, toys = data["suites"]
    assert source["suite"] == SOURCE.name
    assert source["score"] == pytest.approx(
        SOURCE.score({r.task: r.metrics for r in results.tasks})
    )
    assert toys["score"] is None


def test_paloma_is_registered_as_sources_of_domain_tasks():
    assert SUITES["paloma"] == PALOMA
    assert [suite.name for suite in PALOMA.members] == ["paloma:wikitext_103"]
    wikitext = TASKS["paloma_wikitext_103"]
    assert isinstance(wikitext, PerplexityTask)
    assert (wikitext.dataset.subset, wikitext.dataset.split) == ("wikitext_103", "test")
    assert (wikitext.domain, wikitext.window) == (None, 2048)


def test_source_suite_names_a_task_per_domain():
    suite = source_suite("c4_100_domains", ("00_a.org", "01_b.com"))
    assert suite.name == "paloma:c4_100_domains"
    names = [task.name for task in suite.tasks]
    assert names == ["paloma_c4_100_domains__00_a.org", "paloma_c4_100_domains__01_b.com"]
    domains = [task.domain for task in suite.tasks if isinstance(task, PerplexityTask)]
    assert domains == ["00_a.org", "01_b.com"]


def test_toy_texts_span_several_windows(tokenizer):
    lengths = [len(tokenizer.encode(row["text"])) for row in TEXT_ROWS]
    assert min(lengths) > TEXT_TASK.window
