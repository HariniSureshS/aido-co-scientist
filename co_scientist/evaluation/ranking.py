"""Elo-style tournament ranking of model approaches.

Instead of just comparing raw metric values, models are ranked via
pairwise matchups using a margin-based Elo system. This gives a more
nuanced ranking that accounts for score differences between models.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from co_scientist.evaluation.types import ModelResult


@dataclass
class Player:
    """A model in the tournament."""

    name: str  # e.g. "xgboost_default"
    approach: str  # e.g. "xgboost"
    elo: float = 1500.0
    matches: int = 0
    wins: int = 0


class EloRanker:
    """Elo-based ranking of model approaches via pairwise matchups."""

    def __init__(self, k_factor: float = 32.0, margin_scale: float = 0.1):
        self.players: dict[str, Player] = {}
        self.k_factor = k_factor
        self.margin_scale = margin_scale
        self.match_history: list[dict[str, Any]] = []

    def register_model(self, name: str, approach: str) -> None:
        """Register a model as a player in the tournament."""
        if name not in self.players:
            self.players[name] = Player(name=name, approach=approach)

    def record_matchup(
        self,
        model_a: str,
        model_b: str,
        score_a: float,
        score_b: float,
        lower_is_better: bool = False,
    ) -> None:
        """Record a pairwise matchup between two models.

        Uses margin-based outcome: the actual outcome is derived from
        the score difference via a sigmoid function, giving partial
        credit for close losses.
        """
        if model_a not in self.players or model_b not in self.players:
            return

        pa = self.players[model_a]
        pb = self.players[model_b]

        # Compute margin-based actual outcome
        diff = score_a - score_b
        if lower_is_better:
            diff = -diff  # flip so positive means A is better

        actual_a = 1.0 / (1.0 + math.exp(-diff / max(self.margin_scale, 1e-8)))

        # Expected outcome from current Elo ratings
        expected_a = 1.0 / (1.0 + math.pow(10, (pb.elo - pa.elo) / 400))

        # Update Elo ratings
        pa.elo += self.k_factor * (actual_a - expected_a)
        pb.elo += self.k_factor * ((1 - actual_a) - (1 - expected_a))

        pa.matches += 1
        pb.matches += 1
        if actual_a > 0.5:
            pa.wins += 1
        elif actual_a < 0.5:
            pb.wins += 1

        self.match_history.append({
            "model_a": model_a,
            "model_b": model_b,
            "score_a": score_a,
            "score_b": score_b,
            "actual_a": actual_a,
            "elo_a": pa.elo,
            "elo_b": pb.elo,
        })

    def update_from_results(
        self,
        results: list[ModelResult],
        lower_is_better: bool = False,
    ) -> None:
        """Run pairwise matchups for all models in the results list.

        Registers any new models and runs all-vs-all matchups.
        """
        for r in results:
            approach = r.model_name.split("_")[0] if "_" in r.model_name else r.model_name
            self.register_model(r.model_name, approach)

        # Pairwise matchups
        for i, ra in enumerate(results):
            for rb in results[i + 1:]:
                self.record_matchup(
                    ra.model_name, rb.model_name,
                    ra.primary_metric_value, rb.primary_metric_value,
                    lower_is_better=lower_is_better,
                )

    def get_rankings(self) -> list[Player]:
        """Get all players sorted by Elo rating (descending)."""
        return sorted(self.players.values(), key=lambda p: p.elo, reverse=True)

    def get_approach_rankings(self) -> list[dict[str, Any]]:
        """Get average Elo per model approach/type."""
        approach_elos: dict[str, list[float]] = {}
        for p in self.players.values():
            approach_elos.setdefault(p.approach, []).append(p.elo)

        rankings = []
        for approach, elos in approach_elos.items():
            avg_elo = sum(elos) / len(elos)
            rankings.append({
                "approach": approach,
                "avg_elo": avg_elo,
                "num_variants": len(elos),
            })

        return sorted(rankings, key=lambda x: x["avg_elo"], reverse=True)

    def format_table(self) -> str:
        """Format rankings as a readable table string."""
        rankings = self.get_rankings()
        if not rankings:
            return "No models ranked yet."

        lines = [f"{'Model':<30} {'Elo':>8} {'Matches':>8} {'Wins':>6}"]
        lines.append("-" * 56)
        for p in rankings:
            lines.append(f"{p.name:<30} {p.elo:>8.1f} {p.matches:>8} {p.wins:>6}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize rankings to a dict for reporting."""
        return {
            "rankings": [
                {
                    "name": p.name,
                    "approach": p.approach,
                    "elo": round(p.elo, 1),
                    "matches": p.matches,
                    "wins": p.wins,
                }
                for p in self.get_rankings()
            ],
            "approach_rankings": self.get_approach_rankings(),
            "total_matchups": len(self.match_history),
        }
