"""Tests for the Ollama embeddings gateway and hybrid document search."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from inference.embeddings_gateway import EmbeddingsGateway

try:
    import numpy  # noqa: F401

    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

numpy_required = pytest.mark.skipif(
    not _HAS_NUMPY, reason="numpy not installed — run `pip install -e .[ml]`"
)


@pytest.fixture
def gateway():
    return EmbeddingsGateway(base_url="http://localhost:11434", model="nomic-embed-text")


def _mock_embedding_response(vector):
    resp = MagicMock()
    resp.raise_for_status = MagicMock(return_value=None)
    resp.json = MagicMock(return_value={"embedding": vector})
    return resp


class TestEmbeddingsGateway:
    async def test_embed_single_success(self, gateway):
        client = AsyncMock()
        client.post = AsyncMock(return_value=_mock_embedding_response([0.1, 0.2, 0.3]))
        gateway._http_client = client

        result = await gateway.embed(["hello world"])

        assert result["success"] is True
        assert result["vectors"] == [[0.1, 0.2, 0.3]]
        assert result["model"] == "nomic-embed-text"

    async def test_embed_batch_issues_one_call_per_text(self, gateway):
        client = AsyncMock()
        client.post = AsyncMock(return_value=_mock_embedding_response([0.5, 0.5]))
        gateway._http_client = client

        result = await gateway.embed(["a", "b", "c"])

        assert result["success"] is True
        assert len(result["vectors"]) == 3
        assert client.post.call_count == 3

    async def test_embed_empty_list_short_circuits(self, gateway):
        result = await gateway.embed([])
        assert result == {"success": True, "vectors": [], "model": "nomic-embed-text"}

    async def test_embed_failure_degrades_gracefully(self, gateway):
        client = AsyncMock()
        client.post = AsyncMock(side_effect=RuntimeError("connection refused"))
        gateway._http_client = client

        result = await gateway.embed(["hello"])

        assert result["success"] is False
        assert "connection refused" in result["error"]
        assert result["vectors"] == []

    def test_env_var_defaults(self, monkeypatch):
        monkeypatch.setenv("CIVIC_AI_EMBEDDING_MODEL", "mxbai-embed-large")
        monkeypatch.setenv("CIVIC_AI_EMBEDDING_BASE_URL", "http://ollama.local:11434")
        gw = EmbeddingsGateway()
        assert gw.model == "mxbai-embed-large"
        assert gw.base_url == "http://ollama.local:11434"


@numpy_required
class TestHybridSearch:
    def _kb(self, tmp_path):
        kb = tmp_path / "kb"
        kb.mkdir()
        (kb / "policy.md").write_text(
            "## Data Retention\n"
            "County data must be retained for seven years per state law.\n\n"
            "## Public Records\n"
            "Florida sunshine law applies to all county communications."
        )
        return kb

    def _fake_gateway(self, vector_fn):
        mock = AsyncMock()

        async def _embed(texts):
            return {
                "success": True,
                "vectors": [vector_fn(t) for t in texts],
                "model": "test",
            }

        mock.embed = _embed
        return mock

    async def test_hybrid_search_blends_semantic_and_keyword(self, tmp_path):
        from agents.document_analysis_agent import DocumentAnalysisAgent

        # Put "retention" content near [1,0], "records" near [0,1].
        def vec(text):
            return [1.0, 0.0] if "retention" in text.lower() else [0.0, 1.0]

        agent = DocumentAnalysisAgent(
            docs_dir=self._kb(tmp_path),
            embeddings_gateway=self._fake_gateway(vec),
        )

        results = await agent.search("retention policy", max_results=2)

        assert results["mode"] == "hybrid"
        assert results["results"], "expected at least one match"
        top = results["results"][0]
        assert top["section"].lower().startswith("data retention")
        assert "semantic_score" in top and "keyword_score" in top
        assert 0.0 <= top["semantic_score"] <= 1.0
        assert 0.0 <= top["keyword_score"] <= 1.0

    async def test_falls_back_to_keyword_when_embeddings_fail(self, tmp_path):
        from agents.document_analysis_agent import DocumentAnalysisAgent

        failing = AsyncMock()
        failing.embed = AsyncMock(
            return_value={"success": False, "error": "offline", "vectors": []}
        )

        agent = DocumentAnalysisAgent(
            docs_dir=self._kb(tmp_path), embeddings_gateway=failing
        )

        results = await agent.search("retention")
        assert results["mode"] == "keyword"
        assert results["total_matches"] >= 1

    async def test_works_without_gateway(self, tmp_path):
        from agents.document_analysis_agent import DocumentAnalysisAgent

        agent = DocumentAnalysisAgent(docs_dir=self._kb(tmp_path))
        results = await agent.search("sunshine")
        assert results["mode"] == "keyword"
        assert results["total_matches"] >= 1


@numpy_required
class TestVectorCache:
    async def test_cache_reused_when_content_unchanged(self, tmp_path):
        from agents.document_analysis_agent import DocumentAnalysisAgent

        kb = tmp_path / "kb"
        kb.mkdir()
        (kb / "a.md").write_text("## Section\ncontent here")

        calls = {"n": 0}

        async def _embed(texts):
            calls["n"] += len(texts)
            return {"success": True, "vectors": [[0.5, 0.5]] * len(texts), "model": "x"}

        gw = AsyncMock()
        gw.embed = _embed

        agent1 = DocumentAnalysisAgent(docs_dir=kb, embeddings_gateway=gw)
        await agent1._ensure_vectors()
        first_count = calls["n"]
        assert first_count >= 1
        assert (kb / ".index.npz").exists()

        # New agent, same content — should hit cache, no re-embed for chunks
        # (a single query-time embed still happens on search(), but not on _ensure_vectors).
        agent2 = DocumentAnalysisAgent(docs_dir=kb, embeddings_gateway=gw)
        await agent2._ensure_vectors()
        assert calls["n"] == first_count, "expected cache hit, saw fresh embed"

    async def test_cache_invalidated_when_content_changes(self, tmp_path):
        from agents.document_analysis_agent import DocumentAnalysisAgent

        kb = tmp_path / "kb"
        kb.mkdir()
        doc = kb / "a.md"
        doc.write_text("## Section\noriginal content")

        calls = {"n": 0}

        async def _embed(texts):
            calls["n"] += len(texts)
            return {"success": True, "vectors": [[0.5, 0.5]] * len(texts), "model": "x"}

        gw = AsyncMock()
        gw.embed = _embed

        agent1 = DocumentAnalysisAgent(docs_dir=kb, embeddings_gateway=gw)
        await agent1._ensure_vectors()
        baseline = calls["n"]

        # Edit the source — hash changes, cache must be bypassed for changed chunk
        doc.write_text("## Section\nupdated content")
        agent2 = DocumentAnalysisAgent(docs_dir=kb, embeddings_gateway=gw)
        await agent2._ensure_vectors()
        assert calls["n"] > baseline


@pytest.mark.integration
class TestRealOllama:
    async def test_real_nomic_embed(self):
        """Requires Ollama running + `ollama pull nomic-embed-text`."""
        gw = EmbeddingsGateway()
        result = await gw.embed(["county data retention policy"])
        await gw.close()

        if not result["success"]:
            pytest.skip(f"Ollama not available: {result.get('error')}")
        assert len(result["vectors"]) == 1
        assert len(result["vectors"][0]) >= 384  # nomic is 768d, mxbai 1024d, min guard 384
