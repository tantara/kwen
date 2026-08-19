"""Stats helpers — no tokenizer download required."""

from korean_sft.stats import summarize


def test_summarize_counts_and_average():
    stats = summarize([10, 20, 30])
    assert stats["num_samples"] == 3
    assert stats["total_tokens"] == 60
    assert stats["avg_tokens_per_sample"] == 20.0
    assert stats["min_tokens"] == 10
    assert stats["max_tokens"] == 30
    assert stats["median_tokens"] == 20.0


def test_summarize_empty():
    stats = summarize([])
    assert stats["num_samples"] == 0
    assert stats["total_tokens"] == 0
    assert stats["avg_tokens_per_sample"] == 0.0
