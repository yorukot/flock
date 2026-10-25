"""Batched loglikelihood scoring with a causal language model, kept on the model's device."""

import logging
import re
from collections.abc import Generator, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from huggingface_hub import HfApi
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from flock.eval.requests import Request, Scores

DEFAULT_BATCH_TOKENS = 32768

logger = logging.getLogger(__name__)

# One request's (context, continuation) token ids.
type _Encoded = tuple[list[int], list[int]]


@dataclass(frozen=True, slots=True)
class ModelRef:
    """A Hugging Face model repo pinned to a revision, written `repo@revision`."""

    repo: str
    revision: str

    @classmethod
    def parse(cls, model_id: str) -> ModelRef:
        """
        Parses `repo@revision`.

        Raises:
            `ValueError`: if the id has no revision, so an unpinned model can't be scored.
        """
        repo, _, revision = model_id.partition("@")
        if not repo or not revision:
            raise ValueError(f"model id {model_id!r} must be repo@revision")

        return cls(repo, revision)

    def __str__(self) -> str:
        return f"{self.repo}@{self.revision}"

    def resolve(self) -> ModelRef:
        """
        Returns this ref with its revision resolved to a commit hash.

        A full hash is returned as is, without asking the Hub, so pinned models load offline from the cache.

        Raises:
            `ValueError`: if the Hub doesn't report a commit hash for the revision.
        """
        # 40 lowercase hex characters is how the Hub writes a full commit hash.
        if re.fullmatch(r"[0-9a-f]{40}", self.revision):
            return self

        sha = HfApi().model_info(self.repo, revision=self.revision).sha
        if sha is None:
            raise ValueError(f"the Hub reports no commit hash for {self}")

        return ModelRef(self.repo, sha)


class ScoringModel:
    """
    A causal LM and its tokenizer that answer loglikelihood [`~eval.requests.Request`]s in batches.

    Requests are tokenized as OLMES does, then scored longest first in left-padded batches of at most `batch_tokens` padded tokens. Inputs longer than `max_length` lose tokens from the left.

    Args:
        model (`PreTrainedModel`):
            A causal LM, already on its device and in its dtype.
        tokenizer (`PreTrainedTokenizerBase`):
            The model's tokenizer.
        name (`str`):
            Labels the model in log messages. [`~eval.run.evaluate`] records it as the provenance's `model`, `repo@commit`.
        max_length (`int`, *optional*):
            The longest input the model sees. Defaults to the config's `max_position_embeddings`.
        batch_tokens (`int`, *optional*, defaults to 32768):
            The most padded tokens in one forward pass.

    Raises:
        `ValueError`: if `batch_tokens` or `max_length` is less than 1, or `max_length` isn't given and the config has no `max_position_embeddings`.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        *,
        name: str,
        max_length: int | None = None,
        batch_tokens: int = DEFAULT_BATCH_TOKENS,
    ) -> None:
        if batch_tokens < 1:
            raise ValueError(f"batch_tokens must be at least 1, got {batch_tokens}")
        if max_length is None:
            max_length = getattr(model.config, "max_position_embeddings", None)
        if max_length is None:
            raise ValueError(f"{name}: config has no max_position_embeddings; pass max_length")
        if max_length < 1:
            raise ValueError(f"max_length must be at least 1, got {max_length}")

        self.model = model
        self.tokenizer = tokenizer
        self.name = name
        self.max_length = max_length
        self.batch_tokens = batch_tokens

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
        batch_tokens: int = DEFAULT_BATCH_TOKENS,
    ) -> ScoringModel:
        """
        Loads `repo@revision` from the Hugging Face Hub.

        The revision is resolved to its commit hash, which becomes the model's `name`, so a branch or tag in `model_id` is recorded as the exact weights that were scored.

        Args:
            model_id (`str`):
                The model as `repo@revision`, for example `allenai/OLMo-2-0425-1B@main`. Pin a full 40-character commit hash to load offline from the cache.
            device (`str` or `torch.device`, *optional*):
                Where to put the weights. Defaults to CUDA when available, else the CPU.
            dtype (`torch.dtype`, *optional*, defaults to `torch.bfloat16`):
                The weights' dtype. Log-softmax always runs in float32.
            batch_tokens (`int`, *optional*):
                Passed to [`ScoringModel`].
        """
        ref = ModelRef.parse(model_id).resolve()
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        tokenizer = AutoTokenizer.from_pretrained(ref.repo, revision=ref.revision)
        assert tokenizer is not None  # The stubs allow None; loading raises instead.
        model = AutoModelForCausalLM.from_pretrained(
            ref.repo, revision=ref.revision, dtype=dtype, device_map=device
        )

        return cls(model, tokenizer, name=str(ref), batch_tokens=batch_tokens)

    @property
    def device(self) -> torch.device:
        """The device the weights and every returned tensor live on."""
        return self.model.device

    @property
    def dtype(self) -> torch.dtype:
        """The weights' dtype."""
        return self.model.dtype

    @property
    def attention(self) -> str:
        """The attention implementation (`sdpa`, `eager`, ...), which changes bfloat16 scores."""
        # transformers has no public accessor for it, only this private config attribute.
        return self.model.config._attn_implementation

    def encode(self, request: Request) -> tuple[list[int], list[int]]:
        """
        Returns the (context, continuation) token ids the model scores for `request`.

        An empty or whitespace-only context becomes one token: the tokenizer's beginning-of-sequence (BOS) token, or its end-of-sequence (EOS) token if it has no BOS.
        """
        return self._encode([request])[0]

    @torch.no_grad()  # Not inference_mode: buffers rebuilt here must stay usable in training.
    def loglikelihood(self, requests: Sequence[Request]) -> Scores:
        """
        Scores every request, in order, and counts the context tokens each lost to left truncation.

        The model runs in eval mode, and each module is put back in the mode it was in. The scores stay on the model's device: nothing here waits for the GPU.

        Raises:
            `ValueError`: if a continuation has no tokens, or is longer than `max_length` on its own.
        """
        encoded = self._encode(requests)
        self._check_continuations(requests, encoded)
        input_lengths = [_input_length(pair, self.max_length) for pair in encoded]
        tokens_cut = [
            _window_length(pair) - 1 - length for pair, length in zip(encoded, input_lengths)
        ]
        # Left truncation changes what is scored, so it is never silent.
        truncated = sum(cut > 0 for cut in tokens_cut)
        if truncated:
            logger.warning(
                "%s: %d of %d requests exceed %d tokens; their contexts are left-truncated",
                self.name,
                truncated,
                len(requests),
                self.max_length,
            )

        num_tokens = torch.tensor(
            [len(continuation) for _, continuation in encoded], dtype=torch.int64
        )
        return Scores(
            self._logprobs(encoded),
            num_tokens.to(self.device, non_blocking=True),
            torch.tensor(tokens_cut, dtype=torch.int64).to(self.device, non_blocking=True),
        )

    def _logprobs(self, encoded: list[_Encoded]) -> torch.Tensor:
        """Each encoded request's summed continuation log-probability, in order, as float32 on the model's device, scored longest first in batches of at most `batch_tokens` padded tokens."""
        input_lengths = [_input_length(pair, self.max_length) for pair in encoded]
        # Longest first, as lm-evaluation-harness sorts (`_collate` in `_loglikelihood_tokens`),
        # so the first request of each batch sets its padded width.
        order = sorted(range(len(encoded)), key=lambda i: -input_lengths[i])

        logprob = torch.zeros(len(encoded), dtype=torch.float32, device=self.device)
        with _eval_mode(self.model):
            for batch in self._batches(order, input_lengths):
                # `non_blocking` on these small CPU tensors: the copy is staged at once and the
                # host never waits for GPU work already queued.
                index = torch.tensor(batch).to(self.device, non_blocking=True)
                logprob[index] = self._score([encoded[i] for i in batch])

        return logprob

    def _encode(self, requests: Sequence[Request]) -> list[_Encoded]:
        """[`encode`] for many requests, with one tokenizer call for the distinct contexts and one for the joined texts."""
        pairs = [_move_whitespace(request) for request in requests]
        ids_by_context = self._tokenize_contexts([context for context, _ in pairs])
        joined_ids = self._tokenize([context + continuation for context, continuation in pairs])

        encoded = []
        for (context, _), joined in zip(pairs, joined_ids, strict=True):
            if not context:
                encoded.append(([self._prefix_token()], joined))
                continue

            # As lm-evaluation-harness's `_encode_pair`: the continuation's ids are the joint ids
            # after as many leading ids as the context has when tokenized alone.
            context_ids = ids_by_context[context]
            encoded.append((context_ids, joined[len(context_ids) :]))

        return encoded

    def _tokenize(self, texts: list[str]) -> list[list[int]]:
        """
        Each text's token ids, with no special tokens.

        OLMES adds none (`add_bos_token: False` in `oe_eval/default_configs.py`) except for models configured with one, such as Gemma.
        """
        if not texts:
            return []

        return self.tokenizer(texts, add_special_tokens=False)["input_ids"]

    def _tokenize_contexts(self, contexts: list[str]) -> dict[str, list[int]]:
        """
        Each distinct non-empty context's own token ids.

        Few-shot contexts repeat once per choice, so each is tokenized only once.
        """
        distinct = sorted({context for context in contexts if context})

        return dict(zip(distinct, self._tokenize(distinct), strict=True))

    def _prefix_token(self) -> int:
        """
        The token an empty context becomes: BOS if the tokenizer has one, else EOS, as lm-evaluation-harness's `prefix_token_id`.

        It isn't padding: the continuation's first token needs one real token to be predicted from.

        Raises:
            `ValueError`: if the tokenizer has neither.
        """
        prefix = self.tokenizer.bos_token_id
        if prefix is None:
            prefix = self.tokenizer.eos_token_id
        if prefix is None:
            raise ValueError(
                "an empty context needs a BOS or EOS token, and the tokenizer has neither"
            )

        return prefix

    def _check_continuations(self, requests: Sequence[Request], encoded: list[_Encoded]) -> None:
        """
        Checks that every continuation can be scored, as lm-evaluation-harness asserts in `_loglikelihood_tokens`.

        Raises:
            `ValueError`: if a continuation has no tokens (it would score 0, above every real choice), or is longer than `max_length` on its own.
        """
        for request, (_, continuation) in zip(requests, encoded, strict=True):
            if not continuation:
                raise ValueError(f"continuation has no tokens: {request}")
            if len(continuation) > self.max_length:
                raise ValueError(f"continuation longer than {self.max_length} tokens: {request}")

    def _batches(self, order: list[int], input_lengths: list[int]) -> Iterator[list[int]]:
        """Splits `order` (longest first) into runs whose padded size fits `batch_tokens`, given each request's model-input length."""
        batch: list[int] = []
        for i in order:
            # `batch[0]` is the batch's longest request, so every row pads to its length.
            if batch and (len(batch) + 1) * input_lengths[batch[0]] > self.batch_tokens:
                yield batch
                batch = []
            batch.append(i)

        if batch:
            yield batch

    def _score(self, encoded: list[_Encoded]) -> torch.Tensor:
        """Summed continuation log-probabilities of one batch."""
        batch = _pad_batch(encoded, self.max_length)
        device = self.device

        # TODO: try FlashAttention 3 from the `kernels` package and packed sequences (cu_seqlens)
        # instead of padding. Measure the speedup first, and record the kernel in the provenance:
        # a different kernel changes bfloat16 scores slightly. Packing also changes which
        # positions `logits_to_keep` must keep.
        logits = self.model(
            input_ids=batch.input_ids.to(device, non_blocking=True),
            attention_mask=batch.attention_mask.to(device, non_blocking=True),
            position_ids=batch.position_ids.to(device, non_blocking=True),
            use_cache=False,
            # Full-vocabulary logits for the few-shot context would dominate memory.
            logits_to_keep=batch.keep,
        ).logits
        if logits.shape[1] != batch.keep:
            raise ValueError(f"{self.name} ignored logits_to_keep: got {logits.shape[1]} positions")

        # With weights on several devices, the logits come out on the last one.
        targets = batch.targets.to(logits.device, non_blocking=True)
        scored = batch.scored.to(logits.device, non_blocking=True)
        token_logprob = _target_logprobs(logits, targets)

        # `where`, not a product: logits at padding positions may not be finite.
        return token_logprob.where(scored, 0.0).sum(-1).to(device, non_blocking=True)


@dataclass(frozen=True, slots=True)
class _PaddedBatch:
    """
    One batch of encoded requests as model inputs, left-padded so every continuation ends at the last position.

    Args:
        input_ids (`torch.Tensor`):
            int64 `[rows, width]`, left padding, then each window without its last token.
        attention_mask (`torch.Tensor`):
            int64 `[rows, width]`, 0 at padding.
        position_ids (`torch.Tensor`):
            int64 `[rows, width]`, counting from 0 at each row's first real token.
        targets (`torch.Tensor`):
            int64 `[rows, keep]`, the token each of the last `keep` positions predicts.
        scored (`torch.Tensor`):
            bool `[rows, keep]`, True where that target is a continuation token.
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    targets: torch.Tensor
    scored: torch.Tensor

    @property
    def keep(self) -> int:
        """How many final positions need logits: the batch's longest continuation."""
        return self.targets.shape[1]


def _pad_batch(encoded: list[_Encoded], max_length: int) -> _PaddedBatch:
    """
    Left-pads one batch of (context, continuation) token ids into model inputs, on the CPU.

    Inputs longer than `max_length` lose tokens from the left, keeping the whole continuation, as lm-evaluation-harness's `(context_enc + continuation_enc)[-(self.max_length + 1) :]` in `_loglikelihood_tokens`.
    """
    # Each window is one token longer than the model's input (see `_input_length`).
    windows = [(context + continuation)[-(max_length + 1) :] for context, continuation in encoded]
    width = max(_input_length(pair, max_length) for pair in encoded)
    keep = max(len(continuation) for _, continuation in encoded)
    # The pad id never matters: padding is masked out of attention and never scored. 0 is used
    # instead of `tokenizer.pad_token_id` because that can be None, or added after the
    # embedding matrix was sized.
    pad = 0

    input_ids, attention_mask, position_ids, targets, scored = [], [], [], [], []
    for window, (_, continuation) in zip(windows, encoded, strict=True):
        length = len(window) - 1
        padding = width - length
        input_ids.append([pad] * padding + window[:-1])
        attention_mask.append([0] * padding + [1] * length)
        position_ids.append([0] * padding + list(range(length)))

        # Right-aligned like the kept logits: the last `len(continuation)` of the `keep`
        # positions predict the continuation's tokens.
        target_padding = keep - len(continuation)
        targets.append([pad] * target_padding + window[-len(continuation) :])
        scored.append([False] * target_padding + [True] * len(continuation))

    return _PaddedBatch(
        input_ids=torch.tensor(input_ids),
        attention_mask=torch.tensor(attention_mask),
        position_ids=torch.tensor(position_ids),
        targets=torch.tensor(targets),
        scored=torch.tensor(scored),
    )


def _target_logprobs(
    logits: torch.Tensor, targets: torch.Tensor, chunk_positions: int = 4096
) -> torch.Tensor:
    """
    The float32 log-softmax of `logits` `[rows, positions, vocab]` at `targets` `[rows, positions]`.

    OLMES runs `log_softmax` in the model's dtype (`oe_eval/models/eleuther_huggingface.py`). float32 here is our choice, so bfloat16 rounding of the normalizer doesn't move scores. `logits` can hold `batch_tokens * vocab` values, so a float32 copy of all of it would double peak memory: rows are converted a few at a time.

    Args:
        logits (`torch.Tensor`):
            The model's logits, in any dtype.
        targets (`torch.Tensor`):
            int64, the token whose log-probability each position gives.
        chunk_positions (`int`, *optional*, defaults to 4096):
            The most positions (rows times positions per row) converted to float32 at once, about 1.6 GB for a 100k vocabulary. At least one row is always converted.
    """
    rows_per_chunk = max(1, chunk_positions // logits.shape[1])
    token_logprob = torch.empty(targets.shape, dtype=torch.float32, device=logits.device)

    for start in range(0, len(logits), rows_per_chunk):
        rows = slice(start, start + rows_per_chunk)
        chunk = logits[rows].float()
        target_logits = chunk.gather(-1, targets[rows, :, None]).squeeze(-1)
        token_logprob[rows] = target_logits - chunk.logsumexp(-1)

    return token_logprob


def _window_length(encoded: _Encoded) -> int:
    """The token count of one request's context and continuation together, before truncation."""
    context, continuation = encoded

    return len(context) + len(continuation)


def _input_length(encoded: _Encoded, max_length: int) -> int:
    """The model-input length of one request after left truncation."""
    # A window holds at most `max_length + 1` tokens: the model's input, then one last token that
    # is only ever a target.
    return min(_window_length(encoded), max_length + 1) - 1


def _move_whitespace(request: Request) -> tuple[str, str]:
    """
    The request's (context, continuation), with whitespace at the end of the context moved to the start of the continuation.

    This is the first step of lm-evaluation-harness's `_encode_pair`, which OLMES uses, so `"Answer:"` + `" Paris"` and `"Answer: "` + `"Paris"` score the same tokens. Unlike lm-evaluation-harness, which checks for an empty context before moving whitespace, a whitespace-only context becomes empty and gets the prefix token.
    """
    context = request.context.rstrip()

    return context, request.context[len(context) :] + request.continuation


@contextmanager
def _eval_mode(model: torch.nn.Module) -> Generator[None]:
    """Puts `model` in eval mode, then gives each module back the mode it had, so a training loop can score between steps."""
    modes = [(module, module.training) for module in model.modules()]
    model.eval()

    try:
        yield
    finally:
        for module, training in modes:
            module.training = training
