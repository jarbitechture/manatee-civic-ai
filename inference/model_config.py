"""
County-safe model configuration.
Only lists models appropriate for government use — no personal API keys or cloud references.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ModelConfig:
    """Configuration for an available LLM model"""

    name: str
    provider: str  # "ollama", "azure_openai", "openai"
    model_id: str
    context_window: int
    description: str
    requires_api_key: bool
    base_url: Optional[str] = None
    air_gapped: bool = False


# Models suitable for county government use
COUNTY_MODELS: Dict[str, ModelConfig] = {
    # --- Local models (air-gapped, no data leaves the network) ---
    "phi4": ModelConfig(
        name="Phi-4",
        provider="ollama",
        model_id="phi4",
        context_window=16384,
        description="Microsoft Phi-4. Fast general-purpose model. Good for summarization and Q&A.",
        requires_api_key=False,
        base_url="http://localhost:11434/v1",
        air_gapped=True,
    ),
    "llama3.1:8b": ModelConfig(
        name="Llama 3.1 8B",
        provider="ollama",
        model_id="llama3.1:8b",
        context_window=131072,
        description="Meta Llama 3.1. Strong reasoning, 128K context. Good for document analysis.",
        requires_api_key=False,
        base_url="http://localhost:11434/v1",
        air_gapped=True,
    ),
    "deepseek-r1:8b": ModelConfig(
        name="DeepSeek R1 8B",
        provider="ollama",
        model_id="deepseek-r1:8b",
        context_window=32768,
        description="DeepSeek R1. Strong chain-of-thought reasoning. Good for policy analysis.",
        requires_api_key=False,
        base_url="http://localhost:11434/v1",
        air_gapped=True,
    ),
    "qwen2.5:7b": ModelConfig(
        name="Qwen 2.5 7B",
        provider="ollama",
        model_id="qwen2.5:7b",
        context_window=32768,
        description="Alibaba Qwen 2.5. Balanced speed and quality. Good for general tasks.",
        requires_api_key=False,
        base_url="http://localhost:11434/v1",
        air_gapped=True,
    ),
    # --- Azure OpenAI (data stays in county Azure tenant) ---
    "azure-gpt4o": ModelConfig(
        name="GPT-4o (Azure)",
        provider="azure_openai",
        model_id="gpt-4o",
        context_window=128000,
        description="OpenAI GPT-4o via Azure. Requires Azure OpenAI resource in county tenant.",
        requires_api_key=True,
    ),
    "azure-gpt4o-mini": ModelConfig(
        name="GPT-4o Mini (Azure)",
        provider="azure_openai",
        model_id="gpt-4o-mini",
        context_window=128000,
        description="Smaller, faster, cheaper GPT-4o. Good for high-volume tasks.",
        requires_api_key=True,
    ),
}


def get_model_config(model_name: str) -> Optional[ModelConfig]:
    """Get configuration for a specific model"""
    return COUNTY_MODELS.get(model_name)


def get_air_gapped_models() -> List[ModelConfig]:
    """Get all models that run locally with no external network access"""
    return [m for m in COUNTY_MODELS.values() if m.air_gapped]


def get_cloud_models() -> List[ModelConfig]:
    """Get all models that require cloud API access"""
    return [m for m in COUNTY_MODELS.values() if m.requires_api_key]
