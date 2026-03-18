"""MCTS-inspired experiment tree search for the ReAct agent.

Replaces the linear ReAct loop with a tree of experiment branches.
The agent can explore multiple strategies from the same state and backtrack
to try alternatives when a line of experimentation stalls.

The tree search uses UCB1 for node selection and backpropagation of scores.
"""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console

console = Console()
logger = logging.getLogger(__name__)


@dataclass
class TreeNode:
    """A node in the experiment tree."""

    id: int
    state_snapshot: Any  # ReactState (shallow copy)
    scratchpad: list[Any]  # list[ScratchpadEntry]
    parent: TreeNode | None
    children: list[TreeNode] = field(default_factory=list)
    score: float = 0.0  # best metric at this node
    visits: int = 0
    depth: int = 0
    action_taken: str = ""  # description of branch action


class ExperimentTree:
    """Tree data structure for MCTS-inspired experiment search.

    Key design decisions:
    - TrainedModel objects are immutable after fit(), so we share references.
    - We only shallow-copy the trained_models and results lists.
    - profile, split, eval_config are read-only shared references.
    """

    def __init__(self, max_depth: int = 4, max_nodes: int = 20):
        self.root: TreeNode | None = None
        self.nodes: list[TreeNode] = []
        self.global_best_score: float = 0.0
        self.global_best_node: TreeNode | None = None
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self._lower_is_better = False
        self._next_id = 0

    def _new_id(self) -> int:
        nid = self._next_id
        self._next_id += 1
        return nid

    def init_root(self, state: Any, scratchpad: list[Any], score: float, lower_is_better: bool) -> TreeNode:
        """Create the root node from the initial state."""
        self._lower_is_better = lower_is_better
        node = TreeNode(
            id=self._new_id(),
            state_snapshot=self.snapshot_state(state),
            scratchpad=list(scratchpad),
            parent=None,
            score=score,
            visits=1,
            depth=0,
            action_taken="root",
        )
        self.root = node
        self.nodes.append(node)
        self.global_best_score = score
        self.global_best_node = node
        return node

    @staticmethod
    def snapshot_state(state: Any) -> Any:
        """Create a shallow copy of ReactState.

        Shared references: profile, split, eval_config (read-only).
        Copied: trained_models list, results list, best_result, best_trained.
        """
        from co_scientist.agents.react import ReactState

        return ReactState(
            profile=state.profile,  # shared ref
            split=state.split,  # shared ref
            eval_config=state.eval_config,  # shared ref
            seed=state.seed,
            trained_models=list(state.trained_models),  # shallow copy of list
            results=list(state.results),  # shallow copy of list
            best_result=state.best_result,  # shared ref (immutable after creation)
            best_trained=state.best_trained,  # shared ref
        )

    @staticmethod
    def restore_state(node: TreeNode) -> Any:
        """Restore a ReactState from a tree node's snapshot."""
        return ExperimentTree.snapshot_state(node.state_snapshot)

    def ucb1_score(self, node: TreeNode, exploration_weight: float = 1.41) -> float:
        """Compute UCB1 score for node selection."""
        if node.visits == 0:
            return float("inf")

        parent_visits = node.parent.visits if node.parent else 1
        exploitation = node.score
        if self._lower_is_better:
            # Invert so lower scores are better (higher UCB)
            exploitation = -exploitation

        exploration = exploration_weight * math.sqrt(math.log(parent_visits + 1) / (node.visits + 1))
        return exploitation + exploration

    def select_node_to_expand(self) -> TreeNode:
        """Walk the tree via UCB1 to select a leaf node for expansion."""
        if self.root is None:
            raise RuntimeError("Tree not initialized")

        node = self.root
        while node.children:
            # Pick child with best UCB1
            node = max(node.children, key=lambda n: self.ucb1_score(n))

        return node

    def add_child(
        self,
        parent: TreeNode,
        state: Any,
        scratchpad: list[Any],
        action: str,
        score: float,
    ) -> TreeNode:
        """Add a child node to the tree."""
        child = TreeNode(
            id=self._new_id(),
            state_snapshot=self.snapshot_state(state),
            scratchpad=list(scratchpad),
            parent=parent,
            score=score,
            visits=1,
            depth=parent.depth + 1,
            action_taken=action,
        )
        parent.children.append(child)
        self.nodes.append(child)

        # Update global best
        is_better = (
            (score < self.global_best_score) if self._lower_is_better
            else (score > self.global_best_score)
        )
        if is_better or self.global_best_node is None:
            self.global_best_score = score
            self.global_best_node = child

        return child

    def backpropagate(self, node: TreeNode, score: float) -> None:
        """Update visits and scores up the tree from a node."""
        current = node
        while current is not None:
            current.visits += 1
            # Update score to be the best among all descendants
            is_better = (
                (score < current.score) if self._lower_is_better
                else (score > current.score)
            )
            if is_better:
                current.score = score
            current = current.parent

    def best_path(self) -> list[TreeNode]:
        """Return the path from root to the global best node."""
        if self.global_best_node is None:
            return []

        path = []
        node = self.global_best_node
        while node is not None:
            path.append(node)
            node = node.parent
        path.reverse()
        return path

    def summary(self) -> dict[str, Any]:
        """Return a summary of the tree search for reporting."""
        best_path = self.best_path()
        return {
            "total_nodes": len(self.nodes),
            "max_depth_reached": max((n.depth for n in self.nodes), default=0),
            "global_best_score": self.global_best_score,
            "best_path_length": len(best_path),
            "best_path_actions": [n.action_taken for n in best_path],
            "branches_explored": sum(1 for n in self.nodes if n.children),
        }
