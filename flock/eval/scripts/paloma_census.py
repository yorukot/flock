"""Counts the docs, domains, bytes, tokens and windows of every Paloma source in one split, and prints the domain list `flock/eval/tasks/paloma.py` freezes.

Run from the repository root, after accepting the dataset's terms on the Hugging Face Hub:

    uv run python flock/eval/scripts/paloma_census.py test paloma-census-test.json

It reads the raw files at the pinned revision, not `datasets`, so the counts don't depend on a `datasets` version, and it loads only a tokenizer, never model weights.
"""

import gzip
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from flock.eval.tasks import PerplexityTask
from flock.eval.tasks.paloma import PALOMA

# Every Paloma task scores the same revision with the same window, so the first stands for all,
# and the census counts what the tasks score.
TASK = PALOMA.tasks[0]
assert isinstance(TASK, PerplexityTask)
REPO = TASK.dataset.repo
REVISION = TASK.dataset.revision
WINDOW = TASK.window
TOKENIZER = ("allenai/OLMo-2-0425-1B", "a1847dff35000b4271fa70afc5db10fd29fedbdf")


def count_source(tokenizer: PreTrainedTokenizerBase, files: list[str]) -> dict[str, Any]:
    """
    Counts one source's split from its `.jsonl.gz` files, overall and per domain.

    Raises:
        `ValueError`: if some rows have a `subdomain` and others don't, since `DOMAINS` can't list the rows without one.
    """
    totals: Counter[str] = Counter()
    domain_docs: Counter[str] = Counter()
    domain_tokens: Counter[str] = Counter()
    doc_ids = []
    empty_files = []
    for file in files:
        path = hf_hub_download(REPO, file, repo_type="dataset", revision=REVISION)
        with gzip.open(path, "rt") as f:
            rows = [json.loads(line) for line in f]
        if not rows:
            empty_files.append(file)
            continue

        # No special tokens, as `ScoringModel` tokenizes.
        encoded = tokenizer([row["text"] for row in rows], add_special_tokens=False)
        lengths = [len(token_ids) for token_ids in encoded["input_ids"]]
        for row, length in zip(rows, lengths, strict=True):
            # A source with one domain has no `subdomain` field. Its domain is "", and a null
            # `subdomain` counts as none, as `PerplexityTask.load` reads it.
            domain = row.get("subdomain") or ""
            doc_ids.append(row["id"])
            domain_docs[domain] += 1
            domain_tokens[domain] += length
            totals["docs"] += 1
            totals["bytes"] += len(row["text"].encode())
            totals["tokens"] += length
            totals["empty_texts"] += length == 0
            totals[f"over_{WINDOW}"] += length > WINDOW
            # Rounded up: the last window holds what is left.
            totals["windows"] += max(1, -(-length // WINDOW))

    if "" in domain_docs and len(domain_docs) > 1:
        raise ValueError(f"{files[0]}: some rows have a subdomain and some don't")

    return {
        **totals,
        "domains": len(domain_docs),
        "repeated_ids": len(doc_ids) - len(set(doc_ids)),
        "empty_files": empty_files,
        "domain_docs": dict(sorted(domain_docs.items(), key=_rank)),
        "domain_tokens": dict(sorted(domain_tokens.items(), key=_rank)),
    }


def _rank(item: tuple[str, int]) -> tuple[int, str]:
    """Sorts domains by Paloma's rank prefix (`1_www.nytimes.com` before `10_...`), then by name."""
    prefix = item[0].split("_")[0]
    return (int(prefix) if prefix.isdigit() else 0, item[0])


def main(split: str, out: Path) -> None:
    """Writes the census of `split` to `out` as JSON and prints each source's domains."""
    repo, revision = TOKENIZER
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
    assert tokenizer is not None  # The stubs allow None; loading raises instead.
    files = HfApi().list_repo_files(REPO, repo_type="dataset", revision=REVISION)

    # Files are `<source>/<split>/<name>.jsonl.gz`.
    sources: dict[str, list[str]] = {}
    for file in sorted(files):
        parts = file.split("/")
        if len(parts) == 3 and parts[1] == split:
            sources.setdefault(parts[0], []).append(file)

    census = {}
    for source, source_files in sources.items():
        census[source] = count_source(tokenizer, source_files)
        domains = [domain for domain in census[source]["domain_docs"] if domain]
        print(f"{source!r}: {tuple(domains)!r},")

    header = {"revision": REVISION, "split": split, "tokenizer": "@".join(TOKENIZER)}
    out.write_text(json.dumps({**header, "sources": census}, indent=1) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]))
