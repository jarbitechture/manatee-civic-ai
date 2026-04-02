"""
Model and Prompt Registry System
Provides version control, rollback capability, and audit trail for AI models and prompts
Critical for governance and compliance in production AI systems
"""

import json
import hashlib
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from enum import Enum


class DeploymentStatus(Enum):
    """Status of a model/prompt deployment"""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    ROLLED_BACK = "rolled_back"


@dataclass
class ModelVersion:
    """Represents a versioned model"""

    model_id: str
    version: str
    provider: str  # openai, azure, anthropic, etc.
    model_name: str
    deployment_name: Optional[str]
    status: DeploymentStatus
    deployed_at: str
    deployed_by: str
    metadata: Dict[str, Any]
    performance_metrics: Dict[str, float]
    checksum: str


@dataclass
class PromptVersion:
    """Represents a versioned prompt with comprehensive metadata for PromptOps"""

    # Core identification
    prompt_id: str
    version: str
    prompt_text: str
    system_prompt: Optional[str]
    status: DeploymentStatus
    created_at: str
    created_by: str
    purpose: str
    tags: List[str]
    checksum: str

    # Risk and compliance
    risk_tier: str = "green"  # green, yellow, red
    data_types_used: List[str] = None  # ["public", "internal", "pii"]
    requires_human_review: bool = False

    # Technical metadata
    supported_tools: List[str] = None  # ["m365-copilot", "gpt", "claude"]
    task_type: str = ""  # "extraction", "generation", "analysis", "formatting"

    # Quality scores (0-10 scale) - 9-dimension rubric (Feb 9, 2026)
    accuracy_score: Optional[float] = None
    safety_score: Optional[float] = None
    business_value_score: Optional[float] = None
    time_saved_score: Optional[float] = None
    relevance_score: Optional[float] = None
    format_fidelity_score: Optional[float] = None
    readability_score: Optional[float] = None
    policy_alignment_score: Optional[float] = None
    consistency_score: Optional[float] = None
    overall_quality_score: Optional[float] = None  # Weighted average

    # Legacy field (deprecated - use business_value_score instead)
    reusability_score: Optional[float] = None

    # Testing metadata
    test_results: Dict[str, Any] = None
    test_cases: List[Dict[str, Any]] = None  # Detailed test case results
    edge_cases_tested: List[str] = None
    known_limitations: List[str] = None

    # Usage tracking
    times_used: int = 0
    success_rate: Optional[float] = None  # Percentage (0-100)
    avg_time_saved_minutes: Optional[float] = None

    # Governance (Phase 2 additions - Feb 9, 2026 meeting)
    owner: str = ""  # Who owns/maintains this prompt
    last_reviewed: str = ""
    next_review: str = ""  # ISO date string
    reviewer: str = ""  # Who reviewed this prompt (distinct from creator)
    team_process: str = ""  # Which team/process uses this (IT, HR, Finance, etc.)
    business_unit: str = ""  # Department/division (InfoTech, Public Works, etc.)

    # Platform and schema (Phase 2)
    platform: str = "Both"  # "Copilot", "GPT", or "Both"
    output_schema_type: str = ""  # "action_item", "triage", "meeting_summary", etc.
    persona: str = ""  # Persona (Customer Service Agent, Technical Expert, etc.)

    def __post_init__(self):
        """Initialize default values for mutable fields"""
        if self.data_types_used is None:
            self.data_types_used = []
        if self.supported_tools is None:
            self.supported_tools = []
        if self.test_results is None:
            self.test_results = {}
        if self.test_cases is None:
            self.test_cases = []
        if self.edge_cases_tested is None:
            self.edge_cases_tested = []
        if self.known_limitations is None:
            self.known_limitations = []


class ModelPromptRegistry:
    """
    Registry for managing model and prompt versions
    Enables rollback, audit trail, and compliance tracking
    """

    def __init__(self, registry_dir: str = "governance/registry"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)

        self.models_file = self.registry_dir / "models.json"
        self.prompts_file = self.registry_dir / "prompts.json"
        self.audit_log_file = self.registry_dir / "audit_log.json"

        self.models: Dict[str, List[ModelVersion]] = {}
        self.prompts: Dict[str, List[PromptVersion]] = {}
        self.audit_log: List[Dict] = []

        self._load_registry()

    def _load_registry(self):
        """Load registry from disk with backward compatibility"""
        if self.models_file.exists():
            with open(self.models_file, "r") as f:
                data = json.load(f)
                for model_id, versions in data.items():
                    self.models[model_id] = [
                        ModelVersion(**{**v, "status": DeploymentStatus(v["status"])})
                        for v in versions
                    ]

        if self.prompts_file.exists():
            with open(self.prompts_file, "r") as f:
                data = json.load(f)
                for prompt_id, versions in data.items():
                    loaded_versions = []
                    for v in versions:
                        # Ensure status is correct type
                        v["status"] = DeploymentStatus(v["status"])

                        # Add default values for new fields if missing (backward compatibility)
                        defaults = {
                            "risk_tier": "green",
                            "data_types_used": [],
                            "requires_human_review": False,
                            "supported_tools": [],
                            "task_type": "",
                            "accuracy_score": None,
                            "safety_score": None,
                            "business_value_score": None,
                            "time_saved_score": None,
                            "relevance_score": None,
                            "format_fidelity_score": None,
                            "readability_score": None,
                            "policy_alignment_score": None,
                            "consistency_score": None,
                            "reusability_score": None,  # Legacy
                            "overall_quality_score": None,
                            "test_cases": [],
                            "edge_cases_tested": [],
                            "known_limitations": [],
                            "times_used": 0,
                            "success_rate": None,
                            "avg_time_saved_minutes": None,
                            "owner": v.get("created_by", ""),
                            "last_reviewed": "",
                            "next_review": "",
                            "reviewer": "",
                            "team_process": "",
                            "business_unit": "",
                            "platform": "Both",
                            "output_schema_type": "",
                            "persona": "",
                        }

                        # Merge defaults with existing data (existing data takes precedence)
                        for key, default_value in defaults.items():
                            if key not in v:
                                v[key] = default_value

                        loaded_versions.append(PromptVersion(**v))

                    self.prompts[prompt_id] = loaded_versions

        if self.audit_log_file.exists():
            with open(self.audit_log_file, "r") as f:
                self.audit_log = json.load(f)

    def _save_registry(self):
        """Save registry to disk"""
        # Save models
        models_data = {}
        for model_id, versions in self.models.items():
            models_data[model_id] = [{**asdict(v), "status": v.status.value} for v in versions]
        with open(self.models_file, "w") as f:
            json.dump(models_data, f, indent=2)

        # Save prompts
        prompts_data = {}
        for prompt_id, versions in self.prompts.items():
            prompts_data[prompt_id] = [{**asdict(v), "status": v.status.value} for v in versions]
        with open(self.prompts_file, "w") as f:
            json.dump(prompts_data, f, indent=2)

        # Save audit log
        with open(self.audit_log_file, "w") as f:
            json.dump(self.audit_log, f, indent=2)

    def _calculate_checksum(self, data: str) -> str:
        """Calculate checksum for versioning"""
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def _log_action(self, action: str, entity_type: str, entity_id: str, details: Dict):
        """Log an action to audit trail"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "details": details,
        }
        self.audit_log.append(log_entry)
        self._save_registry()

    def register_model(
        self,
        model_id: str,
        provider: str,
        model_name: str,
        deployed_by: str,
        deployment_name: Optional[str] = None,
        status: DeploymentStatus = DeploymentStatus.DEVELOPMENT,
        metadata: Optional[Dict] = None,
        performance_metrics: Optional[Dict] = None,
    ) -> ModelVersion:
        """Register a new model version"""

        # Generate version number
        if model_id in self.models:
            version = f"v{len(self.models[model_id]) + 1}"
        else:
            version = "v1"
            self.models[model_id] = []

        # Create checksum from model config
        config_str = f"{provider}:{model_name}:{deployment_name}"
        checksum = self._calculate_checksum(config_str)

        model_version = ModelVersion(
            model_id=model_id,
            version=version,
            provider=provider,
            model_name=model_name,
            deployment_name=deployment_name,
            status=status,
            deployed_at=datetime.now().isoformat(),
            deployed_by=deployed_by,
            metadata=metadata or {},
            performance_metrics=performance_metrics or {},
            checksum=checksum,
        )

        self.models[model_id].append(model_version)
        self._save_registry()

        self._log_action(
            action="MODEL_REGISTERED",
            entity_type="model",
            entity_id=model_id,
            details={"version": version, "provider": provider, "model_name": model_name},
        )

        return model_version

    def register_prompt(
        self,
        prompt_id: str,
        prompt_text: str,
        created_by: str,
        purpose: str,
        system_prompt: Optional[str] = None,
        status: DeploymentStatus = DeploymentStatus.DEVELOPMENT,
        tags: Optional[List[str]] = None,
        test_results: Optional[Dict] = None,
        # New PromptOps fields
        risk_tier: str = "green",
        data_types_used: Optional[List[str]] = None,
        requires_human_review: bool = False,
        supported_tools: Optional[List[str]] = None,
        task_type: str = "",
        owner: str = "",
    ) -> PromptVersion:
        """Register a new prompt version"""

        # Generate version number
        if prompt_id in self.prompts:
            version = f"v{len(self.prompts[prompt_id]) + 1}"
        else:
            version = "v1"
            self.prompts[prompt_id] = []

        # Create checksum from prompt text
        checksum = self._calculate_checksum(prompt_text)

        prompt_version = PromptVersion(
            prompt_id=prompt_id,
            version=version,
            prompt_text=prompt_text,
            system_prompt=system_prompt,
            status=status,
            created_at=datetime.now().isoformat(),
            created_by=created_by,
            purpose=purpose,
            tags=tags or [],
            checksum=checksum,
            # New PromptOps fields
            risk_tier=risk_tier,
            data_types_used=data_types_used or [],
            requires_human_review=requires_human_review,
            supported_tools=supported_tools or [],
            task_type=task_type,
            owner=owner or created_by,
            test_results=test_results or {},
            test_cases=[],
            edge_cases_tested=[],
            known_limitations=[],
            times_used=0,
            success_rate=None,
            avg_time_saved_minutes=None,
            accuracy_score=None,
            safety_score=None,
            reusability_score=None,
            time_saved_score=None,
            overall_quality_score=None,
            last_reviewed="",
            next_review="",
        )

        self.prompts[prompt_id].append(prompt_version)
        self._save_registry()

        self._log_action(
            action="PROMPT_REGISTERED",
            entity_type="prompt",
            entity_id=prompt_id,
            details={"version": version, "purpose": purpose},
        )

        return prompt_version

    def get_model_version(
        self, model_id: str, version: Optional[str] = None
    ) -> Optional[ModelVersion]:
        """Get a specific model version (or latest if version not specified)"""
        if model_id not in self.models:
            return None

        if version:
            for model in self.models[model_id]:
                if model.version == version:
                    return model
            return None
        else:
            # Return latest version
            return self.models[model_id][-1]

    def get_prompt_version(
        self, prompt_id: str, version: Optional[str] = None
    ) -> Optional[PromptVersion]:
        """Get a specific prompt version (or latest if version not specified)"""
        if prompt_id not in self.prompts:
            return None

        if version:
            for prompt in self.prompts[prompt_id]:
                if prompt.version == version:
                    return prompt
            return None
        else:
            # Return latest version
            return self.prompts[prompt_id][-1]

    def promote_model(self, model_id: str, version: str, new_status: DeploymentStatus):
        """Promote a model to a new status (e.g., testing -> production)"""
        model = self.get_model_version(model_id, version)
        if not model:
            raise ValueError(f"Model {model_id}:{version} not found")

        model.status = new_status
        self._save_registry()

        self._log_action(
            action="MODEL_PROMOTED",
            entity_type="model",
            entity_id=model_id,
            details={"version": version, "new_status": new_status.value},
        )

    def promote_prompt(self, prompt_id: str, version: str, new_status: DeploymentStatus):
        """Promote a prompt to a new status"""
        prompt = self.get_prompt_version(prompt_id, version)
        if not prompt:
            raise ValueError(f"Prompt {prompt_id}:{version} not found")

        prompt.status = new_status
        self._save_registry()

        self._log_action(
            action="PROMPT_PROMOTED",
            entity_type="prompt",
            entity_id=prompt_id,
            details={"version": version, "new_status": new_status.value},
        )

    def rollback_model(self, model_id: str, target_version: str) -> ModelVersion:
        """Rollback to a previous model version"""
        model = self.get_model_version(model_id, target_version)
        if not model:
            raise ValueError(f"Model {model_id}:{target_version} not found")

        # Mark current production version as rolled back
        for m in self.models[model_id]:
            if m.status == DeploymentStatus.PRODUCTION:
                m.status = DeploymentStatus.ROLLED_BACK

        # Promote target version to production
        model.status = DeploymentStatus.PRODUCTION
        self._save_registry()

        self._log_action(
            action="MODEL_ROLLBACK",
            entity_type="model",
            entity_id=model_id,
            details={"target_version": target_version},
        )

        return model

    def rollback_prompt(self, prompt_id: str, target_version: str) -> PromptVersion:
        """Rollback to a previous prompt version"""
        prompt = self.get_prompt_version(prompt_id, target_version)
        if not prompt:
            raise ValueError(f"Prompt {prompt_id}:{target_version} not found")

        # Mark current production version as rolled back
        for p in self.prompts[prompt_id]:
            if p.status == DeploymentStatus.PRODUCTION:
                p.status = DeploymentStatus.ROLLED_BACK

        # Promote target version to production
        prompt.status = DeploymentStatus.PRODUCTION
        self._save_registry()

        self._log_action(
            action="PROMPT_ROLLBACK",
            entity_type="prompt",
            entity_id=prompt_id,
            details={"target_version": target_version},
        )

        return prompt

    def list_models(self, status: Optional[DeploymentStatus] = None) -> List[ModelVersion]:
        """List all models, optionally filtered by status"""
        all_models = []
        for versions in self.models.values():
            all_models.extend(versions)

        if status:
            all_models = [m for m in all_models if m.status == status]

        return all_models

    def list_prompts(self, status: Optional[DeploymentStatus] = None) -> List[PromptVersion]:
        """List all prompts, optionally filtered by status"""
        all_prompts = []
        for versions in self.prompts.values():
            all_prompts.extend(versions)

        if status:
            all_prompts = [p for p in all_prompts if p.status == status]

        return all_prompts

    def get_audit_log(self, entity_type: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """Get audit log, optionally filtered by entity type"""
        log = self.audit_log[-limit:]  # Get last N entries

        if entity_type:
            log = [entry for entry in log if entry.get("entity_type") == entity_type]

        return log


# Example usage and testing
if __name__ == "__main__":
    print("=" * 80)
    print("MODEL/PROMPT REGISTRY TEST")
    print("=" * 80)

    registry = ModelPromptRegistry()

    # Register a model
    print("\n1. Registering Azure GPT-4 model...")
    model = registry.register_model(
        model_id="azure-gpt-4-main",
        provider="azure_openai",
        model_name="gpt-4",
        deployment_name="gpt-4-deployment",
        deployed_by="admin@mymanatee.org",
        status=DeploymentStatus.DEVELOPMENT,
        metadata={"region": "eastus", "resource_group": "civic-ai-rg"},
    )
    print(f"   ✅ Registered: {model.model_id} {model.version}")

    # Register a prompt
    print("\n2. Registering chatbot prompt...")
    prompt = registry.register_prompt(
        prompt_id="civic-ai-greeting",
        prompt_text="You are a helpful AI assistant for Manatee County. Greet the user professionally.",
        system_prompt="Always maintain professional tone. Protect PII.",
        created_by="admin@mymanatee.org",
        purpose="Initial greeting for citizens contacting the county",
        status=DeploymentStatus.DEVELOPMENT,
        tags=["greeting", "public-facing", "manatee-county"],
    )
    print(f"   ✅ Registered: {prompt.prompt_id} {prompt.version}")

    # Promote to production
    print("\n3. Promoting model to production...")
    registry.promote_model("azure-gpt-4-main", "v1", DeploymentStatus.PRODUCTION)
    print(f"   ✅ Promoted to PRODUCTION")

    # Register new version
    print("\n4. Registering updated prompt version...")
    prompt_v2 = registry.register_prompt(
        prompt_id="civic-ai-greeting",
        prompt_text="You are a helpful AI assistant for Manatee County. Provide warm, professional greetings.",
        system_prompt="Always maintain professional tone. Protect PII. Be warm and welcoming.",
        created_by="admin@mymanatee.org",
        purpose="Enhanced greeting with warmer tone",
        status=DeploymentStatus.TESTING,
        tags=["greeting", "public-facing", "manatee-county", "enhanced"],
    )
    print(f"   ✅ Registered: {prompt_v2.prompt_id} {prompt_v2.version}")

    # List production models
    print("\n5. Production models:")
    prod_models = registry.list_models(status=DeploymentStatus.PRODUCTION)
    for m in prod_models:
        print(f"   - {m.model_id} {m.version} ({m.provider}:{m.model_name})")

    # Get audit log
    print("\n6. Recent audit log:")
    for entry in registry.get_audit_log(limit=5):
        print(f"   [{entry['timestamp']}] {entry['action']}: {entry['entity_id']}")

    print("\n✅ Model/Prompt Registry Test Complete")
