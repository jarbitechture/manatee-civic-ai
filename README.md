# Manatee Civic AI

Civic AI platform for Manatee County — 4 agents, a governance layer, and local LLM inference. Everything runs on-premise. No citizen data leaves the county network unless you choose a cloud model.

## What's Inside

### Agents

| Agent | Status | Description |
|-------|--------|-------------|
| **Civic AI Policy Agent** | Ready | Government AI implementation frameworks, NIST RMF guidance, case studies from 8 peer counties, policy checklists. Backed by 13 indexed government AI documents (38K+ words). |
| **Citizen Service Agent** | Ready | Public-facing chatbot for Manatee County — department routing, 311 service requests, FAQ matching, human escalation. Includes phone numbers, hours, and addresses for 5 departments. |
| **Web Intelligence Agent** | Scaffold | Monitors Florida AI legislation, peer county programs (Miami-Dade, Hillsborough, Orange, San Diego, Fairfax), and federal framework updates. Generates leadership briefings. |
| **Document Analysis Agent** | Scaffold | RAG over county documents — auto-ingests knowledge base on startup, provides searchable Q&A with source citations. |

### Governance

| Module | What It Does |
|--------|-------------|
| **PII Redaction** | Scans text for SSNs, emails, phone numbers, credit cards before it reaches an LLM or gets stored |
| **Safety Gates** | Pre-flight checks before any AI operation — validates content passes safety thresholds |
| **Audit Logger** | Logs all AI operations with timestamps, user IDs, event types. Query by user, event type, or time range |
| **Model Registry** | Tracks deployed models and prompts, versions, promotion status (development → testing → production) |

### Inference

| Component | Description |
|-----------|-------------|
| **Local LLM Gateway** | Ollama wrapper for air-gapped inference. No API key needed, no data leaves the machine. |
| **Model Config** | 6 county-safe models — 4 local (phi4, llama3.1:8b, deepseek-r1:8b, qwen2.5:7b) and 2 Azure OpenAI (gpt-4o, gpt-4o-mini) |

### Tools

| Tool | Description |
|------|-------------|
| **Golden Record Analyzer** | Compares document versions using sentence-level semantic similarity (all-MiniLM-L6-v2, Bayesian scoring). Proven on 3 county policy documents. |
| **Before/After Comparator** | Side-by-side document comparison with substance and style scoring |

## Quick Start

```bash
git clone <repo-url> manatee-civic-ai
cd manatee-civic-ai
pip install -e .
```

For ML features (golden record analyzer, embeddings):
```bash
pip install -e ".[ml]"
```

## Verify It Works

```python
python3 -c "
from agents import CivicAIPolicyAgent, CitizenServiceAgent, WebIntelligenceAgent, DocumentAnalysisAgent
from governance import PIIRedactor, SafetyGates, AuditLogger, ModelPromptRegistry
from inference import LocalLLMGateway, COUNTY_MODELS

agent = CivicAIPolicyAgent()
print(f'KB sections: {len(agent.kb_sections)}')       # 13

cs = CitizenServiceAgent()
print(f'Tools: {len(cs.tools)}')                       # 6

da = DocumentAnalysisAgent()
print(f'Doc chunks: {len(da.document_index)}')          # 28

redactor = PIIRedactor()
redacted, _ = redactor.redact_text('SSN is 123-45-6789')
print(f'Redacted: {redacted}')                          # SSN is XXX-XX-XXXX

print(f'Models: {len(COUNTY_MODELS)}')                  # 6
print('All checks passed')
"
```

## Local LLM (Air-Gapped)

```bash
# Install Ollama: https://ollama.com
ollama pull phi4

# Test from Python
python3 -c "
import asyncio
from inference import LocalLLMGateway

async def test():
    gw = LocalLLMGateway()
    health = await gw.health_check()
    print(health['status'])
    response = await gw.chat('What is Manatee County?', model='phi4')
    print(response['content'])

asyncio.run(test())
"
```

## Directory Layout

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

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OLLAMA_HOST` | No | Ollama URL (default: `http://localhost:11434`) |
| `COOKBOOK_LLM_API_KEY` | No | API key for Azure OpenAI or OpenAI |
| `COOKBOOK_LLM_BASE_URL` | No | Custom LLM endpoint URL |
| `CIVIC_AI_DATA_DIR` | No | Document directory for tools (default: `./data/`) |

For air-gapped operation, no environment variables are needed. Just install Ollama and pull a model.

## Requirements

- Python 3.11+
- pip
- Ollama (optional, for local LLM)

## Detailed Documentation

See [RUNBOOK.md](RUNBOOK.md) for full agent API documentation, code examples, troubleshooting, and production readiness status.
