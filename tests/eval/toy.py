"""Toy dataset rows and toy tasks in ARC's and Paloma's row formats, read by every eval test instead of a download."""

from flock.eval.tasks import Dataset, Fewshot, MultipleChoiceTask, PerplexityTask
from flock.eval.tasks.arc import arc_doc

# Paloma's row format, as the split "text" of a source with one domain and the split "domains"
# of a source with two. Longer than one window of `TEXT_TASK`, with multi-byte UTF-8.
TEXT_ROWS = [
    {"id": "t-1", "text": "The quick brown fox jumps over the lazy dog."},
    {"id": "t-2", "text": "臺灣最高的山是哪一座？答案：玉山 🏔"},
    {"id": "t-3", "text": "繁體中文的評估。 The lazy dog sleeps."},
]
DOMAIN_ROWS = [
    {**TEXT_ROWS[0], "subdomain": "news"},
    {**TEXT_ROWS[1], "subdomain": "exam"},
    {**TEXT_ROWS[2], "subdomain": "news"},
]

ROWS = {
    "train": [
        {
            "id": "shot-1",
            "question": "Which gas do plants take in?",
            "choices": {"text": ["oxygen", "carbon dioxide"], "label": ["A", "B"]},
            "answerKey": "B",
        },
        {
            "id": "shot-2",
            "question": "臺灣最高的山是哪一座？",
            "choices": {"text": ["玉山", "雪山", "合歡山"], "label": ["1", "2", "3"]},
            "answerKey": "1",
        },
    ],
    "test": [
        {
            "id": "q-1",
            "question": "What do bees make?",
            "choices": {"text": ["honey", "milk", "silk"], "label": ["A", "B", "C"]},
            "answerKey": "A",
        },
        {
            "id": "q-2",
            "question": "哪一個是水果？",
            "choices": {"text": ["石頭", "蘋果", "椅子", "雲"], "label": ["1", "2", "3", "4"]},
            "answerKey": "2",
        },
    ],
    "text": TEXT_ROWS,
    "domains": DOMAIN_ROWS,
}

TASK = MultipleChoiceTask(
    name="toy",
    dataset=Dataset("cerulean/toy", "0" * 40, None, "test"),
    to_doc=arc_doc,
    primary_metric="acc_per_char",
    fewshot=Fewshot("train", ("shot-2", "shot-1")),
)

TEXT_TASK = PerplexityTask(
    name="toy_text", dataset=Dataset("cerulean/toy", "0" * 40, None, "text"), window=8
)
