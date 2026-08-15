"""Tests for the structural-query detector and FILE INDEX prompt
augmentation. The detector lives in `semrag.retrieval.intent` so
test it here alongside the intent classifier tests."""
from __future__ import annotations

import pytest

from semrag.retrieval.intent import is_structural_query


@pytest.mark.parametrize(
    "query",
    [
        "Summarize the repository structure.",
        "summarize this repository",
        "Summarise the project",
        "Overview of the codebase",
        "What's in this repo?",
        "what is in this repo",
        "Describe the repository",
        "Describe the project",
        "Explain the architecture",
        "Explain the complete architecture",
        "Module hierarchy",
        "Directory structure",
        "high-level overview",
        "review this repository like a senior DV engineer",
        "describe this codebase",
        "summarize",
        "describe",
        "overview",
    ],
)
def test_is_structural_query_true(query):
    assert is_structural_query(query) is True


@pytest.mark.parametrize(
    "query",
    [
        "What does the scoreboard verify?",
        "How does the i2c_master module work?",
        "Trace a write transaction.",
        "Why does my test fail?",
        "How is the SDA line driven?",
        "",
    ],
)
def test_is_structural_query_false(query):
    assert is_structural_query(query) is False