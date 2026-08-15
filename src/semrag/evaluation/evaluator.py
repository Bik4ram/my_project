"""End-to-end evaluation harness with LLM self-evaluation judge.

Evaluates the full pipeline (retrieval → prompt → LLM → answer) including:

- Ground truth metrics: citation quality, hallucination rate, context utilization
- LLM self-evaluation: correctness, groundedness, completeness, hallucination
  risk, citation quality scored 1-10 by the model itself
- Per-query and aggregate report generation

All metrics are best-effort — no guarantee the LLM faithfully self-evaluates.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from semrag.evaluation.models import GoldenQuery
from semrag.services.services import QueryAnswer, QueryService

# ---------------------------------------------------------------------------
# LLM self-evaluation judge
# ---------------------------------------------------------------------------

_JUDGE_PROMPT = """You are evaluating the quality of your previous answer.
Score each dimension from 1 (worst) to 10 (best).

Dimensions:
- correctness: Is the answer factually correct based on the context?
- groundedness: Are claims supported by the provided source citations?
- completeness: Does the answer fully address the question?
- hallucination_risk: How much risk of hallucination/unsupported claims? (1=very safe, 10=very risky)
- citation_quality: Are citations valid, specific, and well-used?

Respond with valid JSON ONLY, no other text:
{"correctness": <int>, "groundedness": <int>, "completeness": <int>, "hallucination_risk": <int>, "citation_quality": <int>}"""


@dataclass
class LLMJudgeScores:
    correctness: int = 0
    groundedness: int = 0
    completeness: int = 0
    hallucination_risk: int = 0
    citation_quality: int = 0

    @property
    def valid(self) -> bool:
        return all(1 <= v <= 10 for v in (
            self.correctness, self.groundedness, self.completeness,
            self.hallucination_risk, self.citation_quality,
        ))


def _parse_judge_scores(raw: str | None) -> LLMJudgeScores | None:
    if not raw:
        return None
    try:
        data = json.loads(raw.strip())
        return LLMJudgeScores(**{k: int(v) for k, v in data.items()})
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Per-query evaluation result
# ---------------------------------------------------------------------------


@dataclass
class EvalAnswerResult:
    query_id: str = ""
    query_text: str = ""
    final_answer: str | None = None

    # Ground truth metrics
    citation_recall: float = 0.0
    citation_precision: float = 0.0
    hallucination_rate: float = 0.0
    context_utilization: float = 0.0

    # Citation details
    cited_paths: list[str] = field(default_factory=list)
    fabricated_paths: list[str] = field(default_factory=list)
    valid_citations_count: int = 0
    fabricated_citations_count: int = 0

    # LLM judge scores (may be None if judge was skipped or failed)
    llm_scores: LLMJudgeScores | None = None

    # Pipeline info
    latency_ms: float = 0.0
    iterations_used: int = 0
    hit_iteration_limit: bool = False
    error: str | None = None


@dataclass
class EvalAnswerReport:
    query_count: int = 0

    # Aggregate ground truth metrics
    mean_citation_recall: float = 0.0
    mean_citation_precision: float = 0.0
    mean_hallucination_rate: float = 0.0
    mean_context_utilization: float = 0.0

    # Aggregate LLM judge scores
    mean_correctness: float = 0.0
    mean_groundedness: float = 0.0
    mean_completeness: float = 0.0
    mean_hallucination_risk: float = 0.0
    mean_citation_quality: float = 0.0

    # Latency
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0

    # Per-query results
    per_query_results: list[EvalAnswerResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# End-to-end evaluator
# ---------------------------------------------------------------------------


class EndToEndEvaluator:
    """Run a set of golden queries through the full pipeline (real LLM,
    real retriever) and evaluate the answers.

    Usage::

        evaluator = EndToEndEvaluator(query_service)
        report = evaluator.evaluate_all(benchmarks, run_llm_judge=True)
        print(report.mean_citation_precision)
    """

    def __init__(self, query_service: QueryService):
        self._query_service = query_service

    def evaluate_all(
        self,
        golden_queries: list[GoldenQuery],
        run_llm_judge: bool = True,
    ) -> EvalAnswerReport:
        results: list[EvalAnswerResult] = []
        for gq in golden_queries:
            result = self._evaluate_single(gq, run_llm_judge=run_llm_judge)
            results.append(result)

        return self._aggregate(results)

    # ------------------------------------------------------------------

    def _evaluate_single(
        self,
        gq: GoldenQuery,
        run_llm_judge: bool = True,
    ) -> EvalAnswerResult:
        result = EvalAnswerResult(query_id=gq.query_id, query_text=gq.query_text)

        try:
            answer: QueryAnswer = self._query_service.ask(gq.project_id, gq.query_text)
        except Exception as exc:  # noqa: BLE001
            result.error = f"{type(exc).__name__}: {exc}"
            return result

        result.final_answer = answer.final_answer
        result.iterations_used = answer.iterations_used
        result.hit_iteration_limit = answer.hit_iteration_limit

        # Compute ground truth metrics from the debug data if available
        if answer.debug is not None:
            result.latency_ms = answer.debug.llm_latency_ms

        # Citation analysis
        included = (answer.messages or [])[:]
        if answer.final_answer:
            citation_data = self._analyze_citations(answer.final_answer, included)
            result.cited_paths = citation_data["cited_paths"]
            result.fabricated_paths = citation_data["fabricated_paths"]
            result.valid_citations_count = citation_data["valid_count"]
            result.fabricated_citations_count = citation_data["fabricated_count"]

            total_cited = len(result.cited_paths)
            if total_cited > 0:
                result.citation_recall = result.valid_citations_count / total_cited
                result.citation_precision = result.valid_citations_count / total_cited
                result.hallucination_rate = result.fabricated_citations_count / total_cited

        # LLM self-evaluation judge
        if run_llm_judge and answer.final_answer:
            result.llm_scores = self._llm_judge(answer)

        return result

    # ------------------------------------------------------------------

    def _analyze_citations(self, answer_text: str, messages: list) -> dict:
        """Extract citation validation data from the answer and messages.
        Works best when messages contain RetrievalResult-like data in the
        user message content. Falls back to heuristic path extraction."""
        from semrag.evaluation.citation_validator import validate_citations

        # Reconstruct included chunks from the user message content
        included_chunks = self._extract_chunks_from_messages(messages)
        validation = validate_citations(answer_text, included_chunks)
        return {
            "cited_paths": list(validation.cited_paths),
            "fabricated_paths": list(validation.fabricated_citations),
            "valid_count": len(validation.valid_citations),
            "fabricated_count": len(validation.fabricated_citations),
        }

    @staticmethod
    def _extract_chunks_from_messages(messages: list) -> list:
        """Parse [Source: ...] headers from the user message to reconstruct
        the list of chunks the LLM was given. Used for citation validation
        when the full RetrievalResult list isn't available."""
        import re

        from semrag.retrieval.models import RetrievalResult

        chunks: list[RetrievalResult] = []
        seen: set[str] = set()
        pattern = re.compile(r"\[Source:\s*([^,\]]+?)(?:,\s*lines\s+(\d+)-(\d+))?\]")

        for msg in messages:
            if not isinstance(msg, dict):
                continue
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            for match in pattern.finditer(content):
                path = match.group(1).strip()
                if path in seen:
                    continue
                seen.add(path)
                lstart = int(match.group(2)) if match.group(2) else None
                lend = int(match.group(3)) if match.group(3) else None
                chunks.append(RetrievalResult(
                    chunk_id=path,
                    project_id="",
                    relative_path=path,
                    chunk_type="",
                    text="",
                    line_start=lstart,
                    line_end=lend,
                    fused_score=0.0,
                ))
        return chunks

    # ------------------------------------------------------------------

    def _llm_judge(self, answer: QueryAnswer) -> LLMJudgeScores | None:
        """Ask the LLM to self-evaluate its answer. Best-effort — returns
        None if the LLM provider is not available or the response cannot
        be parsed."""
        from semrag.reasoning.llm.interfaces import LLMProvider

        llm: LLMProvider = self._query_service._llm  # noqa: SLF001
        if not llm:
            return None

        if not answer.final_answer:
            return None

        try:
            messages = [
                {"role": "system", "content": _JUDGE_PROMPT},
                {"role": "user", "content": f"Question: {answer.debug.query if answer.debug else '(unknown)'}\n\nMy Answer: {answer.final_answer}"},
            ]
            completion = llm.complete(messages)
            return _parse_judge_scores(completion.content)
        except Exception:  # noqa: BLE001
            return None

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate(results: list[EvalAnswerResult]) -> EvalAnswerReport:
        n = len(results)
        if n == 0:
            return EvalAnswerReport()

        report = EvalAnswerReport(query_count=n, per_query_results=results)

        # Ground truth metrics
        report.mean_citation_recall = sum(r.citation_recall for r in results) / n
        report.mean_citation_precision = sum(r.citation_precision for r in results) / n
        report.mean_hallucination_rate = sum(r.hallucination_rate for r in results) / n
        report.mean_context_utilization = sum(r.context_utilization for r in results) / n

        # LLM judge scores (only queries with scores)
        scored = [r for r in results if r.llm_scores is not None and r.llm_scores.valid]
        if scored:
            report.mean_correctness = sum(s.llm_scores.correctness for s in scored) / len(scored)
            report.mean_groundedness = sum(s.llm_scores.groundedness for s in scored) / len(scored)
            report.mean_completeness = sum(s.llm_scores.completeness for s in scored) / len(scored)
            report.mean_hallucination_risk = sum(s.llm_scores.hallucination_risk for s in scored) / len(scored)
            report.mean_citation_quality = sum(s.llm_scores.citation_quality for s in scored) / len(scored)

        # Latency percentiles
        latencies = sorted(r.latency_ms for r in results if r.latency_ms > 0)
        if latencies:
            report.latency_p50_ms = latencies[len(latencies) // 2]
            p95_idx = min(int(len(latencies) * 0.95), len(latencies) - 1)
            report.latency_p95_ms = latencies[p95_idx]

        return report
