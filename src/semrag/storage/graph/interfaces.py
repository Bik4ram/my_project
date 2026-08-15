"""
Graph store interface.

Deliberately the same three operations `PostgresGraphQueries` implements
(`transitive_dependencies`, `transitive_dependents`, `shortest_path`) —
not a general Cypher-passthrough. This is what makes the two
implementations genuinely comparable (same questions, two engines) rather
than an apples-to-oranges "Postgres does X, Neo4j does Y, therefore
Neo4j is more powerful" comparison that would beg the question this part
is actually trying to answer: for the traversal patterns THIS system
needs, does a dedicated graph database earn its operational cost.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class GraphPathResult:
    node_id: str
    depth: int
    path: list[str]


class GraphStore(Protocol):
    def transitive_dependencies(
        self, project_id: str, start_node_id: str, max_depth: int = 10
    ) -> list[GraphPathResult]: ...

    def transitive_dependents(
        self, project_id: str, start_node_id: str, max_depth: int = 10
    ) -> list[GraphPathResult]: ...

    def shortest_path(
        self, project_id: str, source_id: str, target_id: str, max_depth: int = 15
    ) -> list[str] | None: ...
