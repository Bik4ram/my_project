"""
Proves Part 10's RetrievalResult and Part 11's PromptBuilder actually fit
together - real retrieval output, fed into the real prompt assembly, with
real citations pointing at real files.
"""
from pathlib import Path

from semrag.ingestion.chunking.pipeline import chunk_project
from semrag.ingestion.embedding.cache import InMemoryEmbeddingCache
from semrag.ingestion.embedding.pipeline import embed_chunks
from semrag.ingestion.embedding.tfidf_provider import TfidfEmbeddingProvider
from semrag.ingestion.ir.builder import IRBuilder
from semrag.ingestion.parsers.pipeline import build_default_registry
from semrag.ingestion.scanner.manifest_store import InMemoryManifestStore
from semrag.ingestion.scanner.orchestrator import Scanner
from semrag.reasoning.prompting.pipeline import PromptBuilder
from semrag.retrieval.pipeline import HybridRetriever
from semrag.retrieval.reranking import LexicalOverlapReranker
from semrag.storage.postgres.ir_repository import IRGraphRepository
from semrag.storage.vector.models import VectorSearchFilter
from semrag.storage.vector.pgvector_models import EMBEDDING_DIMENSION
from semrag.storage.vector.pgvector_store import PgvectorStore


def _write(root: Path, relative: str, content: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_retrieval_to_prompt_end_to_end(tmp_path: Path, db_session):
    _write(
        tmp_path, "RTL/i2c_master.sv",
        "module i2c_master(input logic clk, output logic scl);\n"
        "  assert property (@(posedge clk) scl == 1'b0)\n"
        "    else $error(\"start condition assertion failed\");\n"
        "endmodule\n",
    )

    manifest_store = InMemoryManifestStore()
    scanner = Scanner(manifest_store=manifest_store)
    manifest, _ = scanner.scan("proj1", str(tmp_path))

    parser_registry = build_default_registry()
    parse_results = {
        rel: r for rel, rec in manifest.files.items()
        if (r := parser_registry.parse_file(tmp_path / rel, rec)) is not None
    }
    ir_graph = IRBuilder().build("proj1", manifest.files, parse_results)
    chunks = chunk_project(str(tmp_path), manifest.files, parse_results, ir_graph)

    provider = TfidfEmbeddingProvider(dimension=EMBEDDING_DIMENSION)
    provider.fit([c.text for c in chunks])
    vector_chunks = embed_chunks(chunks, provider, InMemoryEmbeddingCache())

    IRGraphRepository(db_session).save(ir_graph)
    db_session.flush()
    PgvectorStore(db_session).upsert_chunks(vector_chunks)
    db_session.flush()

    retriever = HybridRetriever(db_session, reranker=LexicalOverlapReranker())
    query = "why does the start condition assertion fail"
    query_embedding = provider.embed_texts([query])[0]
    results = retriever.retrieve(
        query, query_embedding, project_id="proj1", top_k=5,
        filters=VectorSearchFilter(project_id="proj1"),
    )
    assert len(results) > 0

    prompt = PromptBuilder().build(query, results)

    assert prompt.messages[0].role == "system"
    final_message = prompt.messages[-1].content
    assert "[Source: RTL/i2c_master.sv" in final_message
    assert "start condition assertion failed" in final_message
    assert query in final_message
    assert len(prompt.included_chunks) > 0
