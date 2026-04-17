"""Embeddings Gateway — Air-gapped semantic vectors via Ollama.

Uses Ollama's /api/embeddings endpoint. Default model: nomic-embed-text (768d).
Graceful degradation: failures return success=False; callers fall back to keyword search.

Env vars:
    CIVIC_AI_EMBEDDING_BASE_URL   default: http://localhost:11434
    CIVIC_AI_EMBEDDING_MODEL      default: nomic-embed-text
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from loguru import logger


class EmbeddingsGateway:
    """Ollama-backed embeddings gateway. County-safe, air-gapped."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 60.0,
    ):
        self.base_url = base_url or os.getenv(
            "CIVIC_AI_EMBEDDING_BASE_URL", "http://localhost:11434"
        )
        self.model = model or os.getenv("CIVIC_AI_EMBEDDING_MODEL", "nomic-embed-text")
        self.timeout = timeout
        self._http_client: Any = None

    async def _get_client(self) -> Any:
        if self._http_client is None:
            import httpx

            self._http_client = httpx.AsyncClient(timeout=self.timeout)
        return self._http_client

    async def embed(self, texts: List[str]) -> Dict[str, Any]:
        """Embed a list of texts. Preserves order.

        Returns:
            {"success": bool, "vectors": List[List[float]], "model": str, "error"?: str}
        """
        if not texts:
            return {"success": True, "vectors": [], "model": self.model}

        try:
            client = await self._get_client()
            vectors: List[List[float]] = []
            for text in texts:
                response = await client.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model, "prompt": text},
                )
                response.raise_for_status()
                data = response.json()
                vectors.append(data["embedding"])

            return {"success": True, "vectors": vectors, "model": self.model}

        except Exception as e:
            logger.warning(f"Embeddings failed (model={self.model}): {e}")
            return {
                "success": False,
                "error": str(e),
                "model": self.model,
                "vectors": [],
            }

    async def close(self) -> None:
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
