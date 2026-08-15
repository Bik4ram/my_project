"""
Proves SearchCodebaseTool (wrapping Part 10's real HybridRetriever) and
AgentLoop actually work together against real indexed data - a scripted
LLM (no live model reachable) drives the loop, but the tool's search
result comes from the real pipeline, not a fake.
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
from semrag.reasoning.agents.agent_loop import AgentLoop
from semrag.reasoning.agents.tools import SearchCodebaseTool, ToolRegistry
from semrag.reasoning.llm.interfaces import CompletionResult, ToolCall
from semrag.retrieval.pipeline import HybridRetriever
from semrag.storage.postgres.ir_repository import IRGraphRepository
from semrag.storage.vector.pgvector_models import EMBEDDING_DIMENSION
from semrag.storage.vector.pgvector_store import PgvectorStore


class _ScriptedFakeLLM:
    def __init__(self, script):
        self._script = list(script)
        self.call_count = 0

    def complete(self, messages, tools=None):
        result = self._script[self.call_count]
        self.call_count += 1
        return result

    def stream(self, messages, tools=None):
        raise NotImplementedError


def _write(root: Path, relative: str, content: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_agent_search_tool_returns_real_grounded_results(tmp_path: Path, db_session):
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

    retriever = HybridRetriever(db_session, reranker=None, enable_graph_expansion=False)
    search_tool = SearchCodebaseTool(
        retriever, embed_query_fn=lambda q: provider.embed_texts([q])[0], project_id="proj1"
    )
    registry = ToolRegistry([search_tool])

    llm = _ScriptedFakeLLM([
        CompletionResult(
            content=None,
            tool_calls=[ToolCall(id="call_1", name="search_codebase",
                                  arguments={"query": "start condition assertion i2c master"})],
            finish_reason="tool_calls",
        ),
        CompletionResult(content="The assertion checks that scl is low after start condition.", finish_reason="stop"),
    ])
    loop = AgentLoop(llm, registry)

    result = loop.run([{"role": "user", "content": "why does the start condition assertion fail?"}])

    assert result.final_answer is not None
    tool_message = next(m for m in result.messages if m.get("role") == "tool")
    assert "[Source: RTL/i2c_master.sv" in tool_message["content"]
    assert "start condition assertion failed" in tool_message["content"]
