import copy

import pytest
import torch

from flock.eval import model
from flock.eval.model import ModelRef, ScoringModel
from flock.eval.requests import Request

REQUESTS = [
    Request("Question: Which gas do plants take in?\nAnswer:", " carbon dioxide"),
    Request("Question: Which gas do plants take in?\nAnswer:", " oxygen"),
    Request("Question: 臺灣最高的山是哪一座？\nAnswer:", " 玉山"),
    Request("Answer:", " 繁體中文的評估。"),
    Request("The quick brown fox", " jumps over the lazy dog."),
    Request("", " fox"),
]


def reference(scorer: ScoringModel, request: Request) -> float:
    """The continuation's log-probability from one unbatched, unpadded forward pass."""
    context, continuation = scorer.encode(request)
    ids = (context + continuation)[-(scorer.max_length + 1) :]
    with torch.no_grad():
        logits = scorer.model(torch.tensor([ids[:-1]])).logits[0].float()
    logprobs = logits.log_softmax(-1)
    start = len(ids) - 1 - len(continuation)
    return sum(logprobs[start + i, t].item() for i, t in enumerate(continuation))


def test_model_ref_parse():
    ref = ModelRef.parse("allenai/OLMo-2-0425-1B@a1847dff")
    assert (ref.repo, ref.revision) == ("allenai/OLMo-2-0425-1B", "a1847dff")
    assert str(ref) == "allenai/OLMo-2-0425-1B@a1847dff"


@pytest.mark.parametrize("model_id", ["allenai/OLMo-2-0425-1B", "allenai/OLMo-2-0425-1B@", "@abc"])
def test_model_ref_requires_revision(model_id):
    with pytest.raises(ValueError, match="repo@revision"):
        ModelRef.parse(model_id)


class FakeHub:
    def __init__(self, sha):
        self.sha = sha

    def model_info(self, repo, revision):
        return self


def test_model_ref_resolve_keeps_commit_hash_offline(monkeypatch):
    monkeypatch.setattr(model, "HfApi", None)
    ref = ModelRef("allenai/OLMo-2-0425-1B", "a" * 40)
    assert ref.resolve() is ref


def test_model_ref_resolve_asks_hub_for_branch(monkeypatch):
    monkeypatch.setattr(model, "HfApi", lambda: FakeHub("b" * 40))
    assert ModelRef("org/repo", "main").resolve() == ModelRef("org/repo", "b" * 40)


def test_model_ref_resolve_requires_commit_hash(monkeypatch):
    monkeypatch.setattr(model, "HfApi", lambda: FakeHub(None))
    with pytest.raises(ValueError, match="no commit hash for org/repo@main"):
        ModelRef("org/repo", "main").resolve()


@pytest.mark.parametrize("batch_tokens", [0, -5])
def test_rejects_non_positive_batch_tokens(lm, tokenizer, batch_tokens):
    with pytest.raises(ValueError, match="batch_tokens must be at least 1"):
        ScoringModel(lm, tokenizer, name="tiny@0", batch_tokens=batch_tokens)


def test_encode_splits_at_context_tokens(scorer, tokenizer):
    request = Request("Question: Which gas do plants take in?\nAnswer:", " carbon dioxide")
    context, continuation = scorer.encode(request)
    whole = tokenizer.encode(request.context + request.continuation, add_special_tokens=False)
    assert context == tokenizer.encode(request.context, add_special_tokens=False)
    assert context + continuation == whole
    assert tokenizer.decode(continuation) == " carbon dioxide"


def test_encode_moves_trailing_whitespace_to_continuation(scorer):
    assert scorer.encode(Request("Answer: ", "fox")) == scorer.encode(Request("Answer:", " fox"))


def test_encode_empty_context_uses_eos_without_bos(scorer, tokenizer):
    assert tokenizer.bos_token_id is None
    context, continuation = scorer.encode(Request("", " fox"))
    assert context == [tokenizer.eos_token_id]
    assert continuation == tokenizer.encode(" fox", add_special_tokens=False)
    # A whitespace-only context is empty once its whitespace moves to the continuation.
    assert scorer.encode(Request("  ", "fox")) == (context, tokenizer.encode("  fox"))


def test_empty_context_scores_after_eos_without_bos(scorer, tokenizer):
    # Qwen's tokenizers, like this one, have no BOS token (Qwen3's `bos_token` is null).
    assert tokenizer.bos_token_id is None
    ids = [tokenizer.eos_token_id, *tokenizer.encode(" fox", add_special_tokens=False)]
    with torch.no_grad():
        logits = scorer.model(torch.tensor([ids[:-1]])).logits[0].float()
    expected = logits.log_softmax(-1)[range(len(ids) - 1), ids[1:]].sum().item()

    assert scorer.loglikelihood([Request("", " fox")]).logprob[0].item() == pytest.approx(
        expected, abs=1e-4
    )


def test_encode_empty_context_without_bos_or_eos_raises(lm, tokenizer):
    bare = copy.deepcopy(tokenizer)
    bare.eos_token = None
    with pytest.raises(ValueError, match="neither"):
        ScoringModel(lm, bare, name="tiny@0").encode(Request("", " fox"))


def test_loglikelihood_matches_unbatched_reference(scorer):
    scores = scorer.loglikelihood(REQUESTS)
    expected = torch.tensor([reference(scorer, r) for r in REQUESTS])
    torch.testing.assert_close(scores.logprob, expected, atol=1e-4, rtol=1e-5)
    assert scores.logprob.dtype == torch.float32
    assert scores.num_tokens.tolist() == [len(scorer.encode(r)[1]) for r in REQUESTS]


@pytest.mark.parametrize("batch_tokens", [1, 40, 100, 10_000])
def test_loglikelihood_is_batch_size_and_padding_invariant(scorer, batch_tokens):
    # The default budget puts every request in one batch. batch_tokens=1 scores each request
    # alone, and the budgets between left-pad short requests next to long ones.
    one_batch = scorer.loglikelihood(REQUESTS)
    scorer.batch_tokens = batch_tokens
    torch.testing.assert_close(
        scorer.loglikelihood(REQUESTS).logprob, one_batch.logprob, atol=1e-4, rtol=1e-5
    )


def test_loglikelihood_keeps_request_order(scorer):
    forward = scorer.loglikelihood(REQUESTS).logprob
    backward = scorer.loglikelihood(REQUESTS[::-1]).logprob
    torch.testing.assert_close(backward, forward.flip(0), atol=1e-4, rtol=1e-5)


def test_empty_continuation_raises(scorer):
    with pytest.raises(ValueError, match="no tokens"):
        scorer.loglikelihood([Request("Answer:", ""), REQUESTS[0]])


def test_loglikelihood_restores_each_module_mode(scorer):
    scorer.model.train()
    frozen = scorer.model.model.embed_tokens.eval()
    try:
        scorer.loglikelihood(REQUESTS)
        assert scorer.model.training
        assert not frozen.training
    finally:
        # The model is shared by the whole session, so later tests must get it back in eval mode.
        scorer.model.eval()


def test_score_rejects_a_model_that_ignores_logits_to_keep(scorer, monkeypatch):
    forward = scorer.model.forward
    monkeypatch.setattr(
        scorer.model, "forward", lambda **kwargs: forward(**kwargs | {"logits_to_keep": 0})
    )
    with pytest.raises(ValueError, match="ignored logits_to_keep"):
        scorer.loglikelihood(REQUESTS)


@pytest.mark.parametrize("max_length", [0, -1])
def test_scoring_model_rejects_max_length_below_one(lm, tokenizer, max_length):
    with pytest.raises(ValueError, match="max_length must be at least 1"):
        ScoringModel(lm, tokenizer, name="tiny@0", max_length=max_length)


def test_long_inputs_truncate_from_the_left(lm, tokenizer):
    scorer = ScoringModel(lm, tokenizer, name="tiny@0", max_length=8)
    request = Request("The quick brown fox jumps over the lazy dog.", " fox")
    assert sum(map(len, scorer.encode(request))) > 9
    assert scorer.loglikelihood([request]).logprob[0].item() == pytest.approx(
        reference(scorer, request), abs=1e-4
    )


def test_requests_that_fit_record_no_cut(scorer):
    assert scorer.loglikelihood(REQUESTS).tokens_cut.tolist() == [0] * len(REQUESTS)


def test_tokens_cut_counts_what_left_truncation_drops(lm, tokenizer):
    scorer = ScoringModel(lm, tokenizer, name="tiny@0", max_length=8)
    requests = [Request("The quick brown fox jumps over the lazy dog.", " fox"), Request("A", " b")]
    window = sum(map(len, scorer.encode(requests[0])))

    # The window keeps `max_length + 1` tokens: the model's input, then the last target.
    assert scorer.loglikelihood(requests).tokens_cut.tolist() == [window - 9, 0]


def test_truncation_is_logged(lm, tokenizer, caplog):
    scorer = ScoringModel(lm, tokenizer, name="tiny@0", max_length=8)
    requests = [Request("The quick brown fox jumps over the lazy dog.", " fox"), Request("A", " b")]

    scorer.loglikelihood(requests)

    assert "1 of 2 requests exceed 8 tokens" in caplog.text


def test_continuation_longer_than_max_length_raises(lm, tokenizer):
    scorer = ScoringModel(lm, tokenizer, name="tiny@0", max_length=2)
    with pytest.raises(ValueError, match="longer than 2 tokens"):
        scorer.loglikelihood([Request("Answer:", " jumps over the lazy dog.")])


@pytest.mark.parametrize("chunk_positions", [1, 5, 11, 4096])
def test_target_logprobs_is_the_same_in_any_chunks(chunk_positions):
    # 3 rows of 5 positions: chunks of 1 row, 1 row, 2 rows (uneven last chunk) and all rows.
    torch.manual_seed(0)
    logits = torch.randn(3, 5, 7, dtype=torch.bfloat16)
    targets = torch.randint(7, (3, 5))

    chunked = model._target_logprobs(logits, targets, chunk_positions)

    whole = logits.float().log_softmax(-1).gather(-1, targets[..., None]).squeeze(-1)
    assert chunked.dtype == torch.float32
    torch.testing.assert_close(chunked, whole)
    assert torch.equal(chunked, model._target_logprobs(logits, targets, 15))


def test_batched_rows_do_not_see_each_other(scorer, monkeypatch):
    # The gradient of one request's score reaches only its own row's real tokens, so nothing
    # leaks across rows or from padding through the attention mask or position ids.
    inputs = []
    forward = scorer.model.forward

    def embedded_forward(input_ids, attention_mask, **kwargs):
        embedded = scorer.model.get_input_embeddings()(input_ids).detach().requires_grad_()
        inputs.append((embedded, attention_mask))
        return forward(inputs_embeds=embedded, attention_mask=attention_mask, **kwargs)

    monkeypatch.setattr(scorer.model, "forward", embedded_forward)
    # `_score` runs one forward pass over all the requests, so there is exactly one input.
    logprob = scorer._score(scorer._encode(REQUESTS))
    [(embedded, attention_mask)] = inputs

    for i in range(len(REQUESTS)):
        [grad] = torch.autograd.grad(logprob[i], embedded, retain_graph=True)
        reaches = grad.ne(0).any(-1)
        own_tokens = torch.zeros_like(reaches)
        own_tokens[i] = attention_mask[i].bool()
        assert torch.equal(reaches, own_tokens)


# Long enough for several windows of `WINDOW` tokens, with Chinese and an emoji for multi-byte
# UTF-8.
TEXTS = [
    "The quick brown fox jumps over the lazy dog. 繁體中文的評估。 " * 2,
    "Question: 臺灣最高的山是哪一座？\nAnswer: 玉山 🏔",
    "fox",
]


def lm_eval_windows(token_list, prefix_token, max_seq_len):
    """lm-evaluation-harness 0.4.3's `get_rolling_token_windows` with `context_len=1`, each window passed through `make_disjoint_window` (`lm_eval/utils.py`), copied as the reference."""
    first_seq_len = min(max_seq_len, len(token_list))
    windows = [([prefix_token] + token_list[: first_seq_len - 1], token_list[:first_seq_len])]
    predicted = first_seq_len
    while predicted < len(token_list):
        window_pred_len = min(len(token_list) - predicted, max_seq_len)
        window_end = predicted + window_pred_len
        windows.append(
            (
                token_list[window_end - max_seq_len - 1 : window_end - 1],
                token_list[window_end - window_pred_len : window_end],
            )
        )
        predicted += window_pred_len

    return [(a[: len(a) - (len(b) - 1)], b) for a, b in windows]


def rolling_reference(scorer, text, window):
    """A text's summed log-probability over lm-evaluation-harness's windows, one unbatched forward pass each."""
    ids = scorer.tokenizer.encode(text, add_special_tokens=False)
    total = 0.0
    for context, continuation in lm_eval_windows(ids, scorer.tokenizer.eos_token_id, window):
        window_ids = context + continuation
        assert len(window_ids) - 1 <= window
        with torch.no_grad():
            logits = scorer.model(torch.tensor([window_ids[:-1]])).logits[0].float()
        logprobs = logits.log_softmax(-1)
        start = len(window_ids) - 1 - len(continuation)
        total += sum(logprobs[start + i, t].item() for i, t in enumerate(continuation))

    return total


def text_lengths(scorer):
    return [len(scorer.tokenizer.encode(text, add_special_tokens=False)) for text in TEXTS]


def recorded_windows(scorer, monkeypatch, texts, window):
    """The (context, continuation) windows `rolling_loglikelihood` scores, in order."""
    windows = []

    def record(encoded):
        windows.extend(encoded)
        return torch.zeros(len(encoded))

    monkeypatch.setattr(scorer, "_logprobs", record)
    scorer.rolling_loglikelihood(texts, window)

    return windows


@pytest.mark.parametrize("window", [1, 3, 7, 8, 9, 64])
def test_rolling_windows_are_lm_eval_windows(scorer, monkeypatch, window):
    expected = []
    for text in TEXTS:
        ids = scorer.tokenizer.encode(text, add_special_tokens=False)
        text_windows = lm_eval_windows(ids, scorer.tokenizer.eos_token_id, window)
        # Every token is predicted exactly once.
        assert [t for _, continuation in text_windows for t in continuation] == ids
        expected.extend(text_windows)

    assert recorded_windows(scorer, monkeypatch, TEXTS, window) == expected


@pytest.mark.parametrize("offset", [None, 3, 0, -1])
def test_rolling_loglikelihood_matches_unbatched_windows(scorer, offset):
    # The window is set from the longest text: a third of it (over two windows), or that text's
    # length plus 3 (shorter), plus 0 (exactly one window), or minus 1 (one token longer).
    longest = max(text_lengths(scorer))
    window = longest // 3 if offset is None else longest + offset
    scores = scorer.rolling_loglikelihood(TEXTS, window)

    expected = torch.tensor(
        [rolling_reference(scorer, text, window) for text in TEXTS], dtype=torch.float64
    )
    torch.testing.assert_close(scores.logprob, expected, atol=1e-4, rtol=1e-5)
    assert scores.logprob.dtype == torch.float64
    assert scores.num_tokens.tolist() == text_lengths(scorer)
    assert scores.tokens_cut.tolist() == [0, 0, 0]


@pytest.mark.parametrize("batch_tokens", [1, 40, 10_000])
def test_rolling_loglikelihood_is_batch_size_and_order_invariant(scorer, batch_tokens):
    window = max(text_lengths(scorer)) // 3
    one_batch = scorer.rolling_loglikelihood(TEXTS, window).logprob
    scorer.batch_tokens = batch_tokens
    backward = scorer.rolling_loglikelihood(TEXTS[::-1], window).logprob
    torch.testing.assert_close(backward, one_batch.flip(0), atol=1e-4, rtol=1e-5)


def test_rolling_loglikelihood_starts_from_bos_when_there_is_one(lm, tokenizer, monkeypatch):
    with_bos = copy.deepcopy(tokenizer)
    with_bos.bos_token = "<|pad|>"
    scorer = ScoringModel(lm, with_bos, name="tiny@0")
    windows = recorded_windows(scorer, monkeypatch, ["fox"], 8)
    assert windows == [([with_bos.bos_token_id], with_bos.encode("fox"))]


@pytest.mark.parametrize("window", [0, 129])
def test_rolling_loglikelihood_rejects_a_window_outside_max_length(scorer, window):
    with pytest.raises(ValueError, match="window must be from 1 to max_length 128"):
        scorer.rolling_loglikelihood(TEXTS, window)


def test_rolling_loglikelihood_rejects_an_empty_text(scorer):
    with pytest.raises(ValueError, match="text has no tokens"):
        scorer.rolling_loglikelihood(["fox", ""], 8)
