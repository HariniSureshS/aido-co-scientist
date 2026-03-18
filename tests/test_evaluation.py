"""Tests for the Elo-based ranking system."""

from __future__ import annotations

from co_scientist.evaluation.ranking import EloRanker


def test_elo_ranker_basic():
    """Register models, record matchups, and verify rankings are sensible."""
    ranker = EloRanker(k_factor=32.0, margin_scale=0.1)

    ranker.register_model("model_a", "logistic")
    ranker.register_model("model_b", "xgboost")
    ranker.register_model("model_c", "random_forest")

    # model_a beats model_b by a large margin
    ranker.record_matchup("model_a", "model_b", score_a=0.95, score_b=0.70)
    # model_a beats model_c by a smaller margin
    ranker.record_matchup("model_a", "model_c", score_a=0.95, score_b=0.90)
    # model_c beats model_b
    ranker.record_matchup("model_c", "model_b", score_a=0.90, score_b=0.70)

    rankings = ranker.get_rankings()
    assert len(rankings) == 3

    # model_a should be ranked first (highest Elo)
    assert rankings[0].name == "model_a"
    # model_b should be last
    assert rankings[-1].name == "model_b"

    # Elo of model_a should be above starting value of 1500
    assert rankings[0].elo > 1500.0

    # Match history should have 3 entries
    assert len(ranker.match_history) == 3

    # Approach rankings should work
    approach_rankings = ranker.get_approach_rankings()
    assert len(approach_rankings) == 3


def test_elo_ranker_empty():
    """An empty ranker returns empty rankings."""
    ranker = EloRanker()
    rankings = ranker.get_rankings()
    assert rankings == []
    assert ranker.format_table() == "No models ranked yet."
