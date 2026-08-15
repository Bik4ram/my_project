"""
Agent tools.

Three real tools, each wrapping something already built in this series
rather than reimplementing it: search_codebase wraps Part 10's
HybridRetriever, read_file reads real file content directly (bounded,
since an agent handing back an entire large file defeats the point of
retrieval), list_files wraps Part 5's file listing. Each tool exposes an
OpenAI-compatible function-calling schema (to_openai_schema) so the same
definitions feed both LLMProvider.complete(tools=...) and this module's
own dispatch - one definition, not two that can drift apart.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session

from semrag.storage.postgres.models import FileRow


class Tool(Protocol):
    name: str
    description: str

    def parameters_schema(self) -> dict: ...

    def execute(self, **kwargs) -> str: ...

    def to_openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema(),
            },
        }


class SearchCodebaseTool:
    name = "search_codebase"
    description = "Search the indexed codebase for relevant code, docs, or logs using hybrid retrieval."

    def __init__(self, hybrid_retriever, embed_query_fn, project_id: str):
        self._retriever = hybrid_retriever
        self._embed_query_fn = embed_query_fn
        self._project_id = project_id

    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural-language search query."},
                "top_k": {"type": "integer", "description": "Max results to return.", "default": 5},
            },
            "required": ["query"],
        }

    def execute(self, query: str, top_k: int = 5) -> str:
        from semrag.storage.vector.models import VectorSearchFilter

        embedding = self._embed_query_fn(query)
        results = self._retriever.retrieve(
            query, embedding, project_id=self._project_id, top_k=top_k,
            filters=VectorSearchFilter(project_id=self._project_id),
        )
        if not results:
            return "No relevant results found in the codebase for this query."

        from semrag.reasoning.prompting.citation import format_chunk_with_citation

        return "\n\n---\n\n".join(format_chunk_with_citation(r) for r in results)

    def to_openai_schema(self) -> dict:
        return Tool.to_openai_schema(self)


class ReadFileTool:
    name = "read_file"
    description = "Read the content of a specific file in the project, optionally a line range."
    MAX_CHARS = 20_000

    def __init__(self, project_root: str):
        self._project_root = Path(project_root)

    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "relative_path": {"type": "string", "description": "Path relative to the project root."},
                "line_start": {"type": "integer", "description": "1-indexed start line (optional)."},
                "line_end": {"type": "integer", "description": "1-indexed end line, inclusive (optional)."},
            },
            "required": ["relative_path"],
        }

    def execute(self, relative_path: str, line_start: int | None = None, line_end: int | None = None) -> str:
        target = (self._project_root / relative_path).resolve()
        if not str(target).startswith(str(self._project_root.resolve())):
            return f"Error: {relative_path} resolves outside the project root - refusing to read it."
        if not target.exists():
            return f"Error: {relative_path} does not exist."

        text = target.read_text(errors="replace")
        if line_start is not None or line_end is not None:
            lines = text.splitlines()
            start = (line_start or 1) - 1
            end = line_end or len(lines)
            text = "\n".join(lines[start:end])

        if len(text) > self.MAX_CHARS:
            text = text[: self.MAX_CHARS] + f"\n\n[... truncated, {len(text) - self.MAX_CHARS} more characters ...]"
        return text

    def to_openai_schema(self) -> dict:
        return Tool.to_openai_schema(self)


class ListFilesTool:
    name = "list_files"
    description = "List files in the indexed project, optionally filtered by category."

    def __init__(self, session: Session, project_id: str):
        self._session = session
        self._project_id = project_id

    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "category": {"type": "string", "description": "Optional file category filter, e.g. 'rtl_verilog'."},
            },
        }

    def execute(self, category: str | None = None) -> str:
        stmt = select(FileRow.relative_path).where(FileRow.project_id == self._project_id)
        if category:
            stmt = stmt.where(FileRow.category == category)
        rows = self._session.execute(stmt.order_by(FileRow.relative_path)).all()
        if not rows:
            return "No files found."
        return "\n".join(r.relative_path for r in rows)

    def to_openai_schema(self) -> dict:
        return Tool.to_openai_schema(self)


class ToolRegistry:
    def __init__(self, tools: list):
        self._tools = {t.name: t for t in tools}

    def schemas(self) -> list[dict]:
        return [t.to_openai_schema() for t in self._tools.values()]

    def execute(self, name: str, arguments: dict) -> str:
        from semrag.core.observability import timer, count

        tool = self._tools.get(name)
        if tool is None:
            count("agent.tool.unknown", 1, tool=name)
            return f"Error: unknown tool '{name}'."
        with timer("agent.tool.duration_ms", tool=name):
            try:
                result = tool.execute(**arguments)
                count("agent.tool.ok", 1, tool=name)
                return result
            except TypeError as exc:
                count("agent.tool.bad_args", 1, tool=name)
                return f"Error: invalid arguments for tool '{name}': {exc}"
            except Exception as exc:  # noqa: BLE001 - a tool failure must not crash the agent loop
                count("agent.tool.error", 1, tool=name)
                return f"Error executing tool '{name}': {exc}"
