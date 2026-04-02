"""
Local LLM Gateway — Air-gapped inference via Ollama.
No data leaves the county network. All inference runs on local hardware.
"""

import asyncio
from typing import Any, AsyncIterator, Dict, List, Optional

from loguru import logger

from inference.model_config import COUNTY_MODELS, ModelConfig, get_air_gapped_models


class LocalLLMGateway:
    """
    Ollama-backed LLM gateway for air-gapped county inference.

    Usage:
        gateway = LocalLLMGateway()
        if await gateway.health_check():
            response = await gateway.chat("Summarize this policy...", model="phi4")
    """

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self._http_client: Any = None

    async def _get_client(self) -> Any:
        """Lazy-init HTTP client"""
        if self._http_client is None:
            try:
                import httpx
                self._http_client = httpx.AsyncClient(timeout=120.0)
            except ImportError:
                raise RuntimeError(
                    "httpx is required for LocalLLMGateway. Install with: pip install httpx"
                )
        return self._http_client

    async def health_check(self) -> Dict[str, Any]:
        """Check if Ollama is running and list available models"""
        try:
            client = await self._get_client()
            response = await client.get(f"{self.base_url}/api/tags")
            data = response.json()

            available_models = [m["name"] for m in data.get("models", [])]
            county_models = get_air_gapped_models()
            county_model_ids = [m.model_id for m in county_models]

            return {
                "status": "healthy",
                "ollama_url": self.base_url,
                "total_models": len(available_models),
                "available_models": available_models,
                "county_recommended": [
                    m.model_id for m in county_models if m.model_id in available_models
                ],
                "missing_recommended": [
                    m.model_id for m in county_models if m.model_id not in available_models
                ],
            }
        except Exception as e:
            return {
                "status": "unavailable",
                "error": str(e),
                "ollama_url": self.base_url,
                "help": "Install Ollama from https://ollama.com and run: ollama pull phi4",
            }

    async def chat(
        self,
        prompt: str,
        model: str = "phi4",
        system_message: Optional[str] = None,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Send a chat completion request to Ollama.

        Args:
            prompt: User message
            model: Ollama model name (default: phi4)
            system_message: Optional system prompt
            temperature: Sampling temperature (0.0 - 1.0)

        Returns:
            Response with content and metadata
        """
        client = await self._get_client()

        messages: List[Dict[str, str]] = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        try:
            response = await client.post(
                f"{self.api_url}/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": temperature},
                },
            )
            response.raise_for_status()
            data = response.json()

            return {
                "success": True,
                "content": data.get("message", {}).get("content", ""),
                "model": model,
                "air_gapped": True,
                "eval_count": data.get("eval_count"),
                "eval_duration_ms": data.get("eval_duration", 0) / 1_000_000,
            }

        except Exception as e:
            logger.error(f"Ollama chat failed (model={model}): {e}")
            return {
                "success": False,
                "error": str(e),
                "model": model,
                "help": f"Ensure model is pulled: ollama pull {model}",
            }

    async def generate(
        self,
        prompt: str,
        model: str = "phi4",
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Send a generate (completion) request to Ollama.

        Args:
            prompt: Full prompt text
            model: Ollama model name
            temperature: Sampling temperature

        Returns:
            Generated text with metadata
        """
        client = await self._get_client()

        try:
            response = await client.post(
                f"{self.api_url}/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": temperature},
                },
            )
            response.raise_for_status()
            data = response.json()

            return {
                "success": True,
                "content": data.get("response", ""),
                "model": model,
                "air_gapped": True,
            }

        except Exception as e:
            logger.error(f"Ollama generate failed (model={model}): {e}")
            return {"success": False, "error": str(e), "model": model}

    async def list_models(self) -> List[str]:
        """List models currently available in Ollama"""
        health = await self.health_check()
        return health.get("available_models", [])

    async def close(self):
        """Close the HTTP client"""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
