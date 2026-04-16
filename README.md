# Manatee Civic AI

Civic AI platform for Manatee County — 4 agents, a governance layer, and local LLM inference. Everything runs on-premise. No citizen data leaves the county network unless you choose a cloud model.

The [Prompt Cookbook](https://github.com/jarbitechture/prompt-cookbook-gov) is the training tool. Civic AI is the governance layer. They work independently, but when the cookbook routes through Civic AI, training becomes supervised and auditable.

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
| **Circuit Breaker** | Prevents request pileup when the LLM is down. Opens after N failures, probes to test recovery, closes on success. Returns 503 instead of hanging. |
| **Model Registry** | Tracks deployed models and prompts, versions, promotion status (development → testing → production) |

### Governed LLM Proxy (`api_server.py`)

OpenAI-compatible API that sits between any county app and the LLM. Every request goes through PII redaction, safety gates, circuit breaker protection, and audit logging before reaching the model. Any tool that speaks the OpenAI chat completions format can use this as a drop-in replacement — including agent builder platforms like [Dify](https://github.com/langgenius/dify).

```
County App  →  Governed Proxy (/v1/chat/completions)  →  Ollama / Azure OpenAI
                  ├── PII redaction
                  ├── Safety gates (injection, jailbreak)
                  ├── Circuit breaker (503 when LLM is down)
                  └── Audit logging
```

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

### Start the Governed LLM Proxy

```bash
# With Ollama (air-gapped)
CIVIC_AI_API_KEY=your-key uvicorn api_server:app --port 8100

# With Azure OpenAI
CIVIC_AI_API_KEY=your-key \
CIVIC_AI_LLM_PROVIDER=azure_openai \
CIVIC_AI_LLM_API_KEY=your-azure-key \
CIVIC_AI_LLM_BASE_URL=https://{resource}.openai.azure.com/openai/deployments/{model}/v1 \
uvicorn api_server:app --port 8100
```

Any OpenAI-compatible app can now point at `http://localhost:8100/v1` and all requests go through governance.

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
├── governance/              PII, safety gates, audit, circuit breaker, model registry
├── inference/               Local LLM gateway + model config
├── knowledge_base/          13 gov AI documents (38K+ words)
├── tools/                   Golden record analyzer, comparator
├── tests/                   54 tests (unit + integration)
└── pyproject.toml
```

## Environment Variables

### Governed LLM Proxy

| Variable | Required | Description |
|----------|----------|-------------|
| `CIVIC_AI_API_KEY` | No | API key clients must send (Bearer token). If unset, proxy runs in open-access mode. |
| `CIVIC_AI_LLM_PROVIDER` | No | `ollama`, `azure_openai`, or `openai` (default: `ollama`) |
| `CIVIC_AI_LLM_API_KEY` | No | API key for the upstream LLM provider |
| `CIVIC_AI_LLM_BASE_URL` | No | LLM endpoint (default: `http://localhost:11434/v1`) |
| `CIVIC_AI_LLM_DEFAULT_MODEL` | No | Default model name (default: `phi4`) |
| `CIVIC_AI_AUDIT_DIR` | No | Audit log directory (default: `logs/audit`) |
| `CIVIC_AI_RATE_LIMIT` | No | Max requests per window (default: `60`) |
| `CIVIC_AI_RATE_WINDOW` | No | Rate limit window in seconds (default: `900`) |
| `CIVIC_AI_CB_FAILURES` | No | Circuit breaker failure threshold (default: `5`) |
| `CIVIC_AI_CB_TIMEOUT` | No | Circuit breaker recovery timeout in seconds (default: `30`) |

### Agents and Tools

| Variable | Required | Description |
|----------|----------|-------------|
| `OLLAMA_HOST` | No | Ollama URL (default: `http://localhost:11434`) |
| `CIVIC_AI_DATA_DIR` | No | Document directory for tools (default: `./data/`) |

For air-gapped operation, no environment variables are needed. Just install Ollama and pull a model.

## County Use Cases

### Prompt Cookbook Integration

The [Prompt Cookbook](https://github.com/jarbitechture/prompt-cookbook-gov) is a separate training app that teaches county staff to write prompts. It has "Try It" and "Chat" features that send prompts to an LLM. By pointing the cookbook's `COOKBOOK_LLM_BASE_URL` at this governed proxy, every staff prompt goes through PII redaction and audit logging — without changing the cookbook's code.

```
# In the Prompt Cookbook's environment:
COOKBOOK_LLM_API_KEY=your-civic-ai-key
COOKBOOK_LLM_BASE_URL=http://civic-ai-server:8100/v1
```

### Other County Applications

| Use Case | How It Works |
|----------|-------------|
| **311 Service Desk** | Citizen Service Agent + governed proxy. Staff paste citizen inquiries, AI drafts responses. PII is redacted before the LLM sees it. Every interaction is audit-logged for public records. |
| **Policy Drafting** | Staff write policy drafts with AI help. Golden Record Analyzer compares versions. Safety gates block prompts that try to override AI behavior. |
| **Board Meeting Prep** | Document Analysis Agent searches county ordinances and policies. Policy Agent provides framework guidance. All queries logged for transparency. |
| **IT Helpdesk Triage** | Staff describe issues in plain language, AI categorizes and routes. Governed proxy ensures no PII (employee IDs, passwords) reaches the LLM. |
| **Public Records Requests** | Document Analysis Agent searches indexed documents. Audit trail shows exactly what was searched, by whom, and when — ready for FOIA compliance. |
| **Training & Onboarding** | New hires use the Prompt Cookbook to learn AI prompting. Every practice prompt goes through governance. Managers can review audit logs to track training progress. |
| **Budget Analysis** | Staff paste spreadsheet data into prompts for summarization. PII redaction catches any embedded SSNs or account numbers before they reach the model. |
| **Legislative Monitoring** | Web Intelligence Agent tracks Florida AI bills and peer county programs. Generates briefings for county leadership. |

## Requirements

- Python 3.11+
- pip
- Ollama (optional, for local LLM)

## Deployment

This repo contains the governance code, agents, and proxy — all generic and safe for reuse. Deployment configuration (Docker/Podman compose, CI/CD pipelines, IIS reverse proxy setup) lives in a separate private repo. See [RUNBOOK.md](RUNBOOK.md) for the two-repo architecture and platform integration details.

## Detailed Documentation

See [RUNBOOK.md](RUNBOOK.md) for full agent API documentation, code examples, troubleshooting, and production readiness status.
