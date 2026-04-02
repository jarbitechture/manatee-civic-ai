# Manatee Civic AI — Runbook

## What This Is

A civic AI platform for Manatee County with 4 agents, a governance layer, and local LLM inference. Everything runs on-premise — no citizen data leaves the county network unless you choose a cloud model.

---

## 1. Prerequisites

| Requirement | Version | Why |
|-------------|---------|-----|
| Python | 3.11+ | async/await, typing features |
| pip | latest | dependency installation |
| Ollama | latest | local LLM inference (optional but recommended) |
| Git | any | version control |

Optional for ML features (golden record analyzer, embeddings):
- `sentence-transformers`, `scikit-learn`, `scipy` — install with `pip install ".[ml]"`

---

## 2. Installation

```bash
git clone <repo-url> manatee-civic-ai
cd manatee-civic-ai
pip install -e .
```

For ML features:
```bash
pip install -e ".[ml]"
```

## Verify It Works

After installing, run this smoke test:

```python
python3 -c "
from agents import CivicAIPolicyAgent, CitizenServiceAgent, WebIntelligenceAgent, DocumentAnalysisAgent
from governance import PIIRedactor, SafetyGates, AuditLogger, ModelPromptRegistry
from inference import LocalLLMGateway, COUNTY_MODELS

# Agents
agent = CivicAIPolicyAgent()
print(f'CivicAI KB sections: {len(agent.kb_sections)}')          # Expected: 13

cs = CitizenServiceAgent()
print(f'CitizenService tools: {len(cs.tools)}')                   # Expected: 6

da = DocumentAnalysisAgent()
print(f'DocAnalysis chunks: {len(da.document_index)}')             # Expected: 28

wi = WebIntelligenceAgent()
print(f'WebIntel peers: {len(wi.PEER_COUNTIES)}')                 # Expected: 5

# Governance
redactor = PIIRedactor()
redacted, _ = redactor.redact_text('SSN is 123-45-6789 and email is test@example.com')
print(f'Redacted: {redacted}')                                    # Expected: SSN is XXX-XX-XXXX and email is XXXX@example.com

# Inference
print(f'County models: {len(COUNTY_MODELS)}')                     # Expected: 6

print('All checks passed')
"
```

All checks should pass with just `pip install -e .` — no API keys or Ollama required.

---

## 3. Local LLM Setup (Air-Gapped)

Install Ollama: https://ollama.com

```bash
# Pull a recommended model
ollama pull phi4

# Verify it's running
curl http://localhost:11434/api/tags
```

Test from Python:
```python
from inference import LocalLLMGateway
import asyncio

async def test():
    gw = LocalLLMGateway()
    health = await gw.health_check()
    print(health["status"])  # "healthy"

    response = await gw.chat("What is Manatee County's population?", model="phi4")
    print(response["content"])

asyncio.run(test())
```

### Recommended Models

| Model | Size | Best For | Pull Command |
|-------|------|----------|-------------|
| phi4 | 2.7 GB | General Q&A, summarization | `ollama pull phi4` |
| llama3.1:8b | 4.7 GB | Document analysis, long context (128K) | `ollama pull llama3.1:8b` |
| deepseek-r1:8b | 4.9 GB | Policy analysis, reasoning | `ollama pull deepseek-r1:8b` |
| qwen2.5:7b | 4.4 GB | Balanced speed/quality | `ollama pull qwen2.5:7b` |

All models run locally. No API key needed. No data leaves the machine.

### Azure OpenAI (Alternative)

If the county has an Azure OpenAI resource:

```python
from inference import LocalLLMGateway

gw = LocalLLMGateway(base_url="https://{resource}.openai.azure.com/openai/deployments/{model}")
# Set COOKBOOK_LLM_API_KEY in environment
```

---

## 4. Agents

### 4.1 Civic AI Policy Agent

**What it does:** Government AI implementation guidance — frameworks, case studies, NIST RMF, policy checklists.

**Knowledge base:** Reads from `knowledge_base/COUNTY_AI_POLICY_RESEARCH.md` and `knowledge_base/GOVERNMENT_AI_DOCUMENTS_KB.md` (13 indexed government AI documents, 38K+ words).

```python
from agents import CivicAIPolicyAgent
import asyncio

agent = CivicAIPolicyAgent()

async def demo():
    # Get implementation framework
    framework = await agent.get_implementation_framework(
        county_size="medium", current_maturity="planning"
    )

    # Search the knowledge base
    results = await agent.search_knowledge_base("NIST risk management")

    # Get Miami-Dade case study
    case = await agent.analyze_case_study("Miami-Dade")

    # Get policy checklist
    checklist = await agent.get_policy_checklist(focus_area="governance")

asyncio.run(demo())
```

**Key methods:**
| Method | Returns |
|--------|---------|
| `get_implementation_framework()` | 5-phase rollout plan |
| `analyze_case_study(county)` | Timeline, achievements, lessons (8 counties) |
| `get_nist_framework_guidance()` | NIST AI RMF application for counties |
| `get_policy_checklist(focus)` | Actionable checklist (governance/security/training/deployment) |
| `search_knowledge_base(query)` | Ranked results across 13 gov AI documents |
| `get_2026_priorities()` | Current year implementation priorities |

### 4.2 Citizen Service Agent

**What it does:** Public-facing chatbot for Manatee County services — department routing, 311 requests, FAQ matching, human escalation.

**Requires:** An LLM client for natural language responses (works without one using template responses).

```python
from agents import CitizenServiceAgent
from agents.base_agent import AgentContext

agent = CitizenServiceAgent()
context = AgentContext(user_id="citizen-1", session_id="s-1", request="How do I pay my water bill?")

result = await agent.run(context)
print(result.output)
```

**Key methods:**
| Method | Returns |
|--------|---------|
| `answer_inquiry(text)` | LLM-generated response with department info |
| `find_department(service)` | Matching department, phone, hours |
| `create_311_request(desc, user_id)` | Service request ID and tracking info |
| `check_faq(question)` | Matched FAQ answer if found |
| `escalate_to_human(reason, user_id)` | Phone/email/in-person options + ticket ID |

### 4.3 Web Intelligence Agent

**What it does:** Monitors Florida legislation, peer county AI programs, and federal framework updates.

**Requires:** An HTTP client (e.g., `httpx.AsyncClient`) for live scanning. Works without one using knowledge base data only.

```python
from agents.web_intelligence_agent import WebIntelligenceAgent

agent = WebIntelligenceAgent()

# Scan for Florida AI legislation
results = await agent.scan_legislation(jurisdiction="florida")

# Monitor what peer counties are doing
peers = await agent.monitor_peer_counties()

# Generate a briefing for leadership
briefing = await agent.generate_briefing()
```

**Tracked peer counties:** Miami-Dade, Hillsborough, Orange (FL), San Diego (CA), Fairfax (VA)

**Tracked sources:** Florida Legislature, NIST AI RMF, GovAI Coalition, NACo

### 4.4 Document Analysis Agent

**What it does:** RAG over county documents — ingests policies and ordinances, provides searchable Q&A with source citations.

**Auto-ingests:** All `.md` files in `knowledge_base/` on startup.

```python
from agents.document_analysis_agent import DocumentAnalysisAgent

agent = DocumentAnalysisAgent()

# Search across all documents
results = await agent.search("PII requirements for AI systems")

# Ask a question with citations
answer = await agent.ask("What does NIST recommend for AI risk assessment?")

# Ingest additional documents
await agent.ingest_document("/path/to/county_ordinance.md")

# List everything indexed
docs = await agent.list_documents()
```

---

## 5. Governance Layer

### 5.1 PII Redaction (`governance/pii_redaction.py`)

Scans text for personally identifiable information before it reaches an LLM or gets stored.

**Detects:** SSNs, email addresses, phone numbers, credit card numbers, names, addresses.

### 5.2 Safety Gates (`governance/safety_gates.py`)

Pre-flight checks before any AI operation — validates that content passes safety thresholds.

### 5.3 Audit Logger (`governance/audit_logger.py`)

Logs all AI operations with timestamps, user IDs, and event types. Query by user, event type, or time range.

### 5.4 Model Registry (`governance/model_registry.py`)

Tracks which models and prompts are deployed, their versions, who deployed them, and their promotion status (development → testing → production).

---

## 6. Tools

### Golden Record Analyzer (`tools/golden_record_analyzer.py`)

Compares document versions using sentence-level semantic similarity. Uses `all-MiniLM-L6-v2` embeddings, cosine similarity matrices, and Bayesian beta-binomial scoring.

**Use case:** Compare two drafts of a county policy to see what changed, what was added, and which version scores higher on substance and style.

**Requires:** `pip install ".[ml]"` (installs sentence-transformers, scikit-learn, scipy, numpy, pandas, nltk, PyPDF2)

**Data directory:** Both tools read documents from a configurable directory. Set `CIVIC_AI_DATA_DIR` or they default to `data/` relative to the project root.

```bash
# Use default (./data/)
python tools/golden_record_analyzer.py

# Point to a custom directory
CIVIC_AI_DATA_DIR=/path/to/county/documents python tools/golden_record_analyzer.py

# Before/after comparator
CIVIC_AI_DATA_DIR=/path/to/county/documents python tools/before_after_comparator.py
```

Place your document files (PDF or text) in the data directory with `extracted_text/` subdirectory for source texts.

### Before/After Comparator (`tools/before_after_comparator.py`)

Side-by-side document comparison tool. Supports `--dir` and `--output` CLI arguments, or uses `CIVIC_AI_DATA_DIR` env var.

---

## 7. Directory Layout

```
manatee-civic-ai/
├── agents/                  4 agents + base class
├── governance/              PII, safety gates, audit, model registry
├── inference/               Local LLM gateway + model config
├── knowledge_base/          13 gov AI documents (38K+ words)
├── tools/                   Golden record analyzer, comparator
├── tests/
└── pyproject.toml
```

---

## 8. Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OLLAMA_HOST` | No | Ollama URL (defaults to `http://localhost:11434`) |
| `COOKBOOK_LLM_API_KEY` | No | API key for Azure OpenAI or OpenAI direct |
| `COOKBOOK_LLM_BASE_URL` | No | Custom LLM endpoint URL |
| `CIVIC_AI_DATA_DIR` | No | Document directory for tools (defaults to `./data/`) |

For air-gapped operation, no environment variables are needed. Just install Ollama and pull a model.

---

## 9. Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: loguru` | `pip install -e .` |
| `ModuleNotFoundError: sentence_transformers` | `pip install -e ".[ml]"` (also installs numpy, pandas, nltk, PyPDF2) |
| Ollama health check returns "unavailable" | Start Ollama: `ollama serve` |
| Ollama chat returns "model not found" | Pull the model: `ollama pull phi4` |
| Civic AI agent returns no KB results | Verify `knowledge_base/*.md` files exist |
| Document agent returns empty index | Check `knowledge_base/` has `.md` files |
| Audit logger not persisting | Logs are in-memory by default — extend for database storage |

---

## 10. What's Production-Ready vs Scaffold

| Component | Status | To Production |
|-----------|--------|--------------|
| Civic AI Policy Agent | Ready | Serving real data from 13 gov docs |
| Citizen Service Agent | Ready | Manatee-specific departments, FAQs, 311 |
| Governance modules | Ready | PII, safety, audit, registry all functional |
| Golden record analyzer | Ready | Proven on 3 policy documents |
| Local LLM gateway | Ready | Ollama wrapper, tested |
| Model config | Ready | 6 county-safe models defined |
| Web Intelligence Agent | Scaffold | Needs HTTP client + scheduled scanning |
| Document Analysis Agent | Scaffold | Keyword search works; add vector embeddings for semantic search |
