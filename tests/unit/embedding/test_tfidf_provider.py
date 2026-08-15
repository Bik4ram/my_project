import math

import pytest

from semrag.ingestion.embedding.tfidf_provider import TfidfEmbeddingProvider

CORPUS = [
    "module i2c_master implements the I2C protocol master controller",
    "module i2c_slave implements the I2C protocol slave responder",
    "always_ff posedge clk negedge rst_n start_condition assertion check",
    "python regression runner executes UVM testbench simulations",
    "python test harness collects coverage reports from simulation logs",
    "markdown specification document describes register map addresses",
    "tcl build script compiles the design using synthesis tool",
    "assert property start condition scl clock timing verification",
]


def test_raises_before_fit():
    provider = TfidfEmbeddingProvider(dimension=32)
    with pytest.raises(RuntimeError):
        provider.embed_texts(["some text"])


def test_fit_on_empty_corpus_raises():
    provider = TfidfEmbeddingProvider(dimension=32)
    with pytest.raises(ValueError):
        provider.fit([])


def test_embeddings_have_configured_dimension():
    provider = TfidfEmbeddingProvider(dimension=32)
    provider.fit(CORPUS)
    embeddings = provider.embed_texts(["module i2c_master test"])
    assert len(embeddings) == 1
    assert len(embeddings[0]) == 32


def test_small_corpus_pads_to_target_dimension():
    # A tiny corpus can't support 768 real SVD components — must still
    # return vectors of exactly the configured dimension, zero-padded.
    provider = TfidfEmbeddingProvider(dimension=768)
    provider.fit(CORPUS)  # only 8 short documents
    embeddings = provider.embed_texts(["module i2c_master test"])
    assert len(embeddings[0]) == 768


def test_embeddings_are_unit_normalized():
    provider = TfidfEmbeddingProvider(dimension=16)
    provider.fit(CORPUS)
    embeddings = provider.embed_texts(["module i2c_master implements protocol"])
    norm = math.sqrt(sum(x * x for x in embeddings[0]))
    assert abs(norm - 1.0) < 1e-6


def test_similar_texts_are_closer_than_dissimilar_texts():
    provider = TfidfEmbeddingProvider(dimension=16)
    provider.fit(CORPUS)

    anchor, similar, dissimilar = provider.embed_texts([
        "module i2c_master implements the I2C protocol",
        "module i2c_slave implements the I2C protocol",  # shares most vocabulary
        "python test harness collects coverage from logs",  # shares little vocabulary
    ])

    def cosine(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        return dot  # already unit-normalized, so dot product == cosine similarity

    sim_to_similar = cosine(anchor, similar)
    sim_to_dissimilar = cosine(anchor, dissimilar)
    assert sim_to_similar > sim_to_dissimilar


def test_deterministic_given_same_fit_and_input():
    provider_a = TfidfEmbeddingProvider(dimension=16, random_state=42)
    provider_a.fit(CORPUS)
    provider_b = TfidfEmbeddingProvider(dimension=16, random_state=42)
    provider_b.fit(CORPUS)

    result_a = provider_a.embed_texts(["module i2c_master test"])
    result_b = provider_b.embed_texts(["module i2c_master test"])
    assert result_a == result_b


def test_model_name_reflects_configured_dimension():
    provider = TfidfEmbeddingProvider(dimension=256)
    assert "256" in provider.model_name
