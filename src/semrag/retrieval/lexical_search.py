"""
Lexical search over Part 6's `chunks` table, using PostgreSQL's built-in
full-text search (`ts_rank_cd` over the `text_search` generated column
added in this part's migration).

**Naming this honestly:** this is Postgres full-text search, not the
literal BM25 algorithm. Postgres's `ts_rank`/`ts_rank_cd` functions use a
different (older, simpler) term-weighting scheme than BM25's saturating
term-frequency formula — they're both "lexical/keyword relevance
ranking," in the same family, but not the same math. Part 1's
architecture doc used "BM25" as shorthand for "the lexical half of hybrid
retrieval," and that's the role this module fills; calling it BM25
outright would overstate what's actually running. A real BM25
implementation (or the `pg_search`/ParadeDB extension, which does
implement BM25 on Postgres) is a legitimate future upgrade if ranking
quality against real queries shows Postgres's native ranking falling
short — not assumed necessary here without that evidence, consistent
with every other storage/algorithm decision in this series.

Exact-identifier queries (an engineer searching for `i2c_master` or
`start_condition`) are exactly what lexical search is good at and vector
search is comparatively weak at (an embedding model has no special
affinity for exact identifier matches over near-synonyms) — this is why
hybrid retrieval fuses both rather than relying on vector search alone,
per Part 1's explicit architecture.
"""
from __future__ import annotations

import re

from sqlalchemy import text
from sqlalchemy.orm import Session

from semrag.retrieval.models import RetrievalResult
from semrag.storage.vector.models import VectorSearchFilter

_LEXEME_PATTERN = re.compile(r"'([^']+)'")


class PostgresLexicalSearch:
    def __init__(self, session: Session):
        self._session = session

    def _query_lexemes(self, query_text: str) -> list[str]:
        """Return the stemmed lexemes Postgres extracts from the query.

        plainto_tsquery ANDs every token ('a' & 'b' & 'c'), and real
        questions have enough distinct terms that no single chunk contains
        all of them — in the rtl_uvm benchmark only 3 of 104 queries
        produced ANY lexical matches with the previous AND semantics, so
        the lexical leg of the hybrid retriever was effectively dead. We
        keep Postgres's stemming (consistent with how chunks are indexed)
        and treat any-match as the match condition, then rank by how many
        distinct query lexemes a chunk actually contains (a cheap,
        IDF-free stand-in for "covers more of the query") before falling
        back to ts_rank_cd as a tie-breaker."""
        plain = self._session.execute(
            text("SELECT plainto_tsquery('english', :query_text)::text"),
            {"query_text": query_text},
        ).scalar()
        if not plain:
            return []
        return list(dict.fromkeys(_LEXEME_PATTERN.findall(plain)))

    def search(
        self,
        query_text: str,
        top_k: int,
        filters: VectorSearchFilter | None = None,
    ) -> list[RetrievalResult]:
        lexemes = self._query_lexemes(query_text)
        if not lexemes:
            return []

        sql = """
            WITH q AS (
                SELECT CAST(
                    array_to_string(
                        array(select quote_literal(unnest(CAST(:lexemes AS text[])))),
                        ' | '
                    ) AS tsquery
                ) AS orq
            )
            SELECT c.chunk_id, c.project_id, c.relative_path, c.chunk_type, c.text,
                   c.source_node_id, c.line_start, c.line_end,
                   c.language, c.category, c.parser_name, c.chunker_name, c.symbol_name,
                   sub.matched AS score
            FROM chunks c, q,
            LATERAL (
                SELECT count(DISTINCT t) AS matched,
                       ts_rank_cd(c.text_search, q.orq) AS rk
                FROM unnest(tsvector_to_array(c.text_search)) AS t
                WHERE t = ANY(:lexemes)
            ) sub
            WHERE c.text_search @@ q.orq
        """
        params: dict = {"lexemes": lexemes, "top_k": top_k}

        if filters:
            if filters.project_id:
                sql += " AND c.project_id = :project_id"
                params["project_id"] = filters.project_id
            if filters.chunk_type:
                sql += " AND c.chunk_type = :chunk_type"
                params["chunk_type"] = filters.chunk_type
            if filters.relative_path:
                sql += " AND c.relative_path = :relative_path"
                params["relative_path"] = filters.relative_path

        sql += " ORDER BY sub.matched DESC, sub.rk DESC LIMIT :top_k"

        rows = self._session.execute(text(sql), params).all()
        return [
            RetrievalResult(
                chunk_id=row.chunk_id,
                project_id=row.project_id,
                relative_path=row.relative_path,
                chunk_type=row.chunk_type,
                text=row.text,
                source_node_id=row.source_node_id,
                line_start=row.line_start,
                line_end=row.line_end,
                language=row.language or "",
                category=row.category or "",
                parser_name=row.parser_name,
                chunker_name=row.chunker_name or "",
                symbol_name=row.symbol_name,
                fused_score=row.score,
                signal_scores={"lexical": row.score},
            )
            for row in rows
        ]
