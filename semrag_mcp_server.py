import asyncio
import json
import os

import httpx
from mcp.server.mcpserver import MCPServer

SEMRAG_URL = os.getenv("SEMRAG_URL", "http://127.0.0.1:8000")
SEMRAG_API_KEY = os.getenv("SEMRAG_API_KEY", "dev-local-key")

mcp = MCPServer("semrag")


def _headers():
    return {
        "Authorization": f"Bearer {SEMRAG_API_KEY}",
        "Content-Type": "application/json",
    }


async def _request(method, path, **kwargs):
    url = f"{SEMRAG_URL}{path}"

    async with httpx.AsyncClient(timeout=600.0) as client:
        response = await client.request(
            method,
            url,
            headers=_headers(),
            **kwargs,
        )

    if response.status_code != 200:
        return (
            f"SemiRAG API error ({response.status_code}): "
            f"{response.text}"
        )

    try:
        return response.json()
    except Exception:
        return response.text


async def _request_index(method, path, **kwargs):
    """Like _request but also treats 202 Accepted as success.

    The index endpoint returns 202 when a new job is created and 200
    when an already-queued/running job is returned for the project.
    """
    url = f"{SEMRAG_URL}{path}"

    async with httpx.AsyncClient(timeout=600.0) as client:
        response = await client.request(
            method,
            url,
            headers=_headers(),
            **kwargs,
        )

    if response.status_code not in (200, 202):
        return (
            f"SemiRAG API error ({response.status_code}): "
            f"{response.text}"
        )

    try:
        return response.json()
    except Exception:
        return response.text


@mcp.tool()
async def semrag_ask(project_id: str, question: str) -> str:
    """
    Ask SemiRAG a question about an indexed semiconductor codebase.

    Uses retrieval, ranking, grounding, and LLM generation.
    """

    if not project_id.strip():
        return "Error: project_id cannot be empty."

    if not question.strip():
        return "Error: question cannot be empty."

    data = await _request(
        "POST",
        f"/v1/projects/{project_id}/answer",
        json={"query": question},
    )

    if isinstance(data, str):
        return data

    answer = data.get("answer", "")
    citations = data.get("citations", [])
    sources = data.get("sources", [])

    result = f"ANSWER:\n{answer}"

    if citations:
        result += "\n\nCITATIONS:\n"
        for citation in citations:
            result += f"- {citation}\n"

    if sources:
        result += "\nSOURCES:\n"
        for source in sources:
            result += f"- {source}\n"

    return result


@mcp.tool()
async def semrag_search(
    project_id: str,
    query: str,
    top_k: int = 10,
    enable_graph_expansion: bool = True,
    enable_reranking: bool = True,
    enable_intent_boost: bool = True,
) -> str:
    """
    Search the indexed semiconductor codebase.

    Returns compact, authoritative evidence including
    exact file paths and source line ranges.
    """

    if not project_id.strip():
        return "Error: project_id cannot be empty."

    if not query.strip():
        return "Error: query cannot be empty."

    if not 1 <= top_k <= 50:
        return "Error: top_k must be between 1 and 50."

    data = await _request(
        "POST",
        f"/v1/projects/{project_id}/query",
        json={
            "query": query,
            "top_k": top_k,
            "enable_graph_expansion": enable_graph_expansion,
            "enable_reranking": enable_reranking,
            "enable_intent_boost": enable_intent_boost,
        },
    )

    if isinstance(data, str):
        return data

    chunks = data.get("included_chunks", [])

    if not chunks:
        return f"No relevant results found for: {query}"

    results = [
        f"SEARCH RESULTS FOR: {query}",
        f"Results returned: {len(chunks)}",
        "",
    ]

    for i, chunk in enumerate(chunks, 1):
        relative_path = chunk.get("relative_path", "unknown")
        line_start = chunk.get("line_start")
        line_end = chunk.get("line_end")
        text = chunk.get("text", "")
        language = chunk.get("language")
        category = chunk.get("category")
        symbol_name = chunk.get("symbol_name")
        score = chunk.get("fused_score")

        results.append(f"RESULT {i}")
        results.append(f"FILE: {relative_path}")

        if line_start is not None and line_end is not None:
            results.append(f"LINES: {line_start}-{line_end}")

        if language:
            results.append(f"LANGUAGE: {language}")

        if category:
            results.append(f"CATEGORY: {category}")

        if symbol_name:
            results.append(f"SYMBOL: {symbol_name}")

        if score is not None:
            results.append(f"SCORE: {score:.4f}")

        results.append("CONTENT:")
        results.append(text.strip())
        results.append("")
        results.append("---")

    return "\n".join(results)
@mcp.tool()
async def semrag_list_files(project_id: str) -> str:
    """
    List files indexed in a SemiRAG project.
    """

    if not project_id.strip():
        return "Error: project_id cannot be empty."

    data = await _request(
        "GET",
        f"/v1/projects/{project_id}/files",
    )

    if isinstance(data, str):
        return data

    return json.dumps(data, indent=2)


@mcp.tool()
async def semrag_read_file(
    project_id: str,
    file_path: str,
    start_line: int | None = None,
    end_line: int | None = None,
) -> str:
    """
    Read indexed source-file content.

    Optional start_line and end_line can restrict the returned range.
    """

    if not project_id.strip():
        return "Error: project_id cannot be empty."

    if not file_path.strip():
        return "Error: file_path cannot be empty."

    params = {
        "file_path": file_path,
    }

    if start_line is not None:
        params["start_line"] = start_line

    if end_line is not None:
        params["end_line"] = end_line

    data = await _request(
        "GET",
        f"/v1/projects/{project_id}/files/content",
        params=params,
    )

    if isinstance(data, str):
        return data

    return json.dumps(data, indent=2)


@mcp.tool()
async def semrag_find_symbol(
    project_id: str,
    symbol: str,
    kind: str | None = None,
    limit: int = 20,
) -> str:
    """
    Find a symbol in the indexed semiconductor codebase.

    First uses the symbol index. If the symbol index has no result,
    falls back to semantic retrieval.
    """

    if not project_id.strip():
        return "Error: project_id cannot be empty."

    if not symbol.strip():
        return "Error: symbol cannot be empty."

    if not 1 <= limit <= 100:
        return "Error: limit must be between 1 and 100."

    params = {
        "symbol": symbol,
        "limit": limit,
    }

    if kind is not None:
        params["kind"] = kind

    data = await _request(
        "GET",
        f"/v1/projects/{project_id}/symbols",
        params=params,
    )

    if isinstance(data, str):
        return data

    if data.get("count", 0) > 0:
        return json.dumps(data, indent=2)

    search_data = await _request(
        "POST",
        f"/v1/projects/{project_id}/query",
        json={
            "query": symbol,
            "top_k": min(limit, 10),
            "enable_graph_expansion": True,
            "enable_reranking": True,
            "enable_intent_boost": True,
        },
    )

    if isinstance(search_data, str):
        return search_data

    fallback_result = {
        "project_id": project_id,
        "symbol": symbol,
        "source": "semantic_search_fallback",
        "symbol_index_results": 0,
        "retrieval": search_data,
    }

    return json.dumps(fallback_result, indent=2)

@mcp.tool()
async def semrag_project_info(project_id: str) -> str:
    """
    Get metadata and information about a SemiRAG project.
    """

    if not project_id.strip():
        return "Error: project_id cannot be empty."

    data = await _request(
        "GET",
        f"/v1/projects/{project_id}/info",
    )

    if isinstance(data, str):
        return data

    return json.dumps(data, indent=2)


@mcp.tool()
async def semrag_index(project_id: str) -> str:
    """
    Re-index a project and wait for the job to finish.

    Uses the existing POST /v1/projects/{project_id}/index API to enqueue
    the job, then polls GET /v1/projects/{project_id}/jobs/{job_id} until
    the job reaches a terminal state (completed, failed, or cancelled).

    Returns a concise structured result with the project_id, job_id, final
    status, progress, stage, duration, statistics, and any job error.
    """

    if not project_id.strip():
        return "Error: project_id cannot be empty."

    info = await _request("GET", f"/v1/projects/{project_id}/info")
    if isinstance(info, str):
        return info

    root_path = info.get("root_path")
    if not root_path:
        return f"Error: cannot determine project root for '{project_id}'."

    data = await _request_index(
        "POST",
        f"/v1/projects/{project_id}/index",
        json={"root_path": root_path},
    )
    if isinstance(data, str):
        return data

    job_id = data.get("job_id")
    if not job_id:
        return (
            f"Error: index API returned no job_id for project '{project_id}'."
        )

    status_path = f"/v1/projects/{project_id}/jobs/{job_id}"
    poll_interval = 2.0
    max_wait_seconds = 600.0
    waited = 0.0
    last_state = "unknown"

    while waited < max_wait_seconds:
        status = await _request("GET", status_path)
        if isinstance(status, str):
            return status

        last_state = status.get("state", "unknown")
        if last_state in ("completed", "failed", "cancelled"):
            return json.dumps(
                {
                    "project_id": project_id,
                    "job_id": job_id,
                    "status": last_state,
                    "progress": status.get("progress"),
                    "current_stage": status.get("current_stage"),
                    "duration": status.get("duration"),
                    "error": status.get("error"),
                    "statistics": status.get("statistics", {}),
                    "waited_seconds": round(waited, 1),
                },
                indent=2,
            )

        await asyncio.sleep(poll_interval)
        waited += poll_interval

    return (
        f"Error: indexing job {job_id} for project '{project_id}' did not "
        f"reach a terminal state within {int(max_wait_seconds)}s "
        f"(last state: {last_state})."
    )


@mcp.tool()
async def semrag_write_file(
    project_id: str,
    file_path: str,
    content: str,
) -> str:
    """
    Create or replace a text file inside the project's root.

    The path must be project-relative; absolute paths and '..' traversal
    are rejected by the API. Re-indexing is the caller's responsibility
    — the response signals ``requires_reindex=true`` after a successful
    write so the agent can decide when to re-index.
    """

    if not project_id.strip():
        return "Error: project_id cannot be empty."

    if not file_path.strip():
        return "Error: file_path cannot be empty."

    data = await _request(
        "POST",
        f"/v1/projects/{project_id}/files/content",
        json={"file_path": file_path, "content": content},
    )

    if isinstance(data, str):
        return data

    return json.dumps(data, indent=2)


@mcp.tool()
async def semrag_apply_patch(
    project_id: str,
    file_path: str,
    old_text: str,
    new_text: str,
) -> str:
    """
    Apply a single exact-text replacement to an existing project file.

    The target file must already exist inside the project root. old_text
    must match EXACTLY ONCE in the file: zero matches or more than one
    match fail with no change made (no partial or ambiguous edits).
    Re-indexing is the caller's responsibility — the response signals
    ``requires_reindex=true`` after a successful patch.
    """

    if not project_id.strip():
        return "Error: project_id cannot be empty."

    if not file_path.strip():
        return "Error: file_path cannot be empty."

    if not old_text.strip():
        return "Error: old_text cannot be empty."

    data = await _request(
        "POST",
        f"/v1/projects/{project_id}/files/patch",
        json={
            "file_path": file_path,
            "old_text": old_text,
            "new_text": new_text,
        },
    )

    if isinstance(data, str):
        return data

    return json.dumps(data, indent=2)


@mcp.tool()
async def semrag_run_validation(project_id: str, target: str = "rtl_lint") -> str:
    """
    Run a named validation target for a project (currently Verilator lint).

    The command is assembled server-side from the project's registered
    validation config — the model can only pick a target name, never
    supply shell text. Available targets are returned by this tool when an
    unknown target is requested.

    Returns a structured result:
      status: "passed" | "failed" | "blocked"
      exit_code, stdout, stderr, command, duration_seconds, blocked_reason

    Interpretation:
      passed   — the code validated cleanly
      failed   — the validator found a defect in the code
      blocked  — a dependency or toolchain piece is missing (e.g. the UVM
                 library); see blocked_reason. Blocked is NOT a pass.
    """

    if not project_id.strip():
        return "Error: project_id cannot be empty."

    if not target.strip():
        return "Error: target cannot be empty."

    data = await _request(
        "POST",
        f"/v1/projects/{project_id}/validate",
        json={"target": target},
    )

    if isinstance(data, str):
        return data

    return json.dumps(data, indent=2)


@mcp.tool()
async def semrag_code(
    project_id: str,
    request: str,
    max_repair_attempts: int = 3,
    validation_target: str | None = None,
) -> str:
    """
    Run the grounded coding agent workflow end-to-end for a repository request.

    The workflow is deterministic and bounded:
      - classifies the user's intent (information / code change / debug /
        review / ambiguous) and NEVER modifies on ambiguous or
        information-only requests
      - retrieves indexed evidence first, then reads exact source before
        planning any patch (read-before-write enforced)
      - applies only the planner's declared minimal changes
      - runs the project's registered validation target
      - on failure, diagnoses and repairs (bounded, max_repair_attempts)
      - verifies each changed file, re-indexes, confirms retrievability
      - returns an A1-grounded summary of what was actually done

    Use for concrete, unambiguous code-change/debug requests. For pure
    questions use semrag_ask.
    """

    if not project_id.strip():
        return "Error: project_id cannot be empty."

    if not request.strip():
        return "Error: request cannot be empty."

    if not 0 <= max_repair_attempts <= 10:
        return "Error: max_repair_attempts must be between 0 and 10."

    payload: dict = {
        "request": request,
        "max_repair_attempts": max_repair_attempts,
    }

    if validation_target is not None:
        payload["validation_target"] = validation_target

    data = await _request(
        "POST",
        f"/v1/projects/{project_id}/code",
        json=payload,
    )

    if isinstance(data, str):
        return data

    return json.dumps(data, indent=2)


if __name__ == "__main__":
    mcp.run()
