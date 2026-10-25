"""Paloma (Magnusson et al. 2023, arXiv 2312.10523): bits per byte over the domains of many sources."""

from flock.eval.tasks.base import Dataset
from flock.eval.tasks.perplexity import PerplexityTask
from flock.eval.tasks.suite import Suite

# Each source's domains in the `test` split at the pinned revision, as
# `flock/eval/scripts/paloma_census.py` lists them. A source with one domain lists none.
DOMAINS: dict[str, tuple[str, ...]] = {
    "wikitext_103": (),
}


def source_suite(source: str, domains: tuple[str, ...]) -> Suite:
    """
    Returns a Paloma source as a suite of one task per domain, scored as the unweighted mean of their `bits_per_byte`.

    Paloma macro-averages a source's domains (sec. 4), averaging perplexity. The mean here is over bits per byte, the log of a per-byte perplexity, which has no precedent: it keeps one domain with a very high perplexity from dominating its source, and it is the scale every other suite averages on. A domain's task is named `paloma_<source>__<domain>`, the key OLMES gives the domain (`PerplexityMetric` in `oe_eval/metrics/perplexity.py`), and a source with one domain has the task `paloma_<source>`.
    """
    # The `test` split, as OLMES's `paloma_<source>::paloma` tasks score it
    # (`oe_eval/configs/tasks.py`).
    dataset = Dataset(
        repo="allenai/paloma",
        revision="65cd6fc59dba021b21db414fa5e8d7765ffbe5e6",
        subset=source,
        split="test",
    )
    if not domains:
        return Suite(f"paloma:{source}", (PerplexityTask(f"paloma_{source}", dataset),))

    tasks = tuple(
        PerplexityTask(f"paloma_{source}__{domain}", dataset, domain=domain) for domain in domains
    )
    return Suite(f"paloma:{source}", tasks)


# One name runs every source. There is no score over all sources: Paloma defines none.
PALOMA = Suite(
    "paloma",
    tuple(source_suite(source, domains) for source, domains in DOMAINS.items()),
    scored=False,
)
