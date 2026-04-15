"""
Governed LLM Proxy — OpenAI-compatible API with PII redaction, safety gates, and audit logging.

Any app that speaks the OpenAI chat completions format can point here instead of directly
at OpenAI/Ollama/Azure. Every request goes through governance before reaching the LLM.

Usage:
    CIVIC_AI_API_KEY=your-key CIVIC_AI_LLM_PROVIDER=ollama uvicorn api_server:app --port 8100
"""

import hashlib
import os
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from governance.audit_logger import AuditLogger, AuditEventType, AuditSeverity
from governance.pii_redaction import PIIRedactor
from governance.safety_gates import SafetyGates, GateStatus


# ── Configuration ──────────────────────────────────────────────────

CIVIC_AI_API_KEY = os.environ.get("CIVIC_AI_API_KEY", "")
LLM_PROVIDER = os.environ.get("CIVIC_AI_LLM_PROVIDER", "ollama")  # ollama | azure_openai | openai
LLM_API_KEY = os.environ.get("CIVIC_AI_LLM_API_KEY", "")
LLM_BASE_URL = os.environ.get("CIVIC_AI_LLM_BASE_URL", "http://localhost:11434/v1")
LLM_DEFAULT_MODEL = os.environ.get("CIVIC_AI_LLM_DEFAULT_MODEL", "phi4")
AUDIT_LOG_DIR = os.environ.get("CIVIC_AI_AUDIT_DIR", "logs/audit")
RATE_LIMIT_MAX = int(os.environ.get("CIVIC_AI_RATE_LIMIT", "60"))
RATE_LIMIT_WINDOW = int(os.environ.get("CIVIC_AI_RATE_WINDOW", "900"))  # 15 minutes


# ── Rate Limiter ───────────────────────────────────────────────────

class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window = window_seconds
        self.buckets: dict[str, list[float]] = defaultdict(list)

    def check(self, key: str) -> bool:
        now = time.time()
        cutoff = now - self.window
        self.buckets[key] = [t for t in self.buckets[key] if t > cutoff]
        if len(self.buckets[key]) >= self.max_requests:
            return False
        self.buckets[key].append(now)
        return True


# ── Globals ────────────────────────────────────────────────────────

rate_limiter = RateLimiter(RATE_LIMIT_MAX, RATE_LIMIT_WINDOW)
redactor = PIIRedactor()
gates = SafetyGates(strict_mode=False)  # non-strict: warnings pass, only failures block
audit = AuditLogger(log_dir=AUDIT_LOG_DIR)
http_client: httpx.AsyncClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(timeout=120.0)
    yield
    await http_client.aclose()


app = FastAPI(
    title="Civic AI Governed LLM Proxy",
    version="0.1.0",
    lifespan=lifespan,
)


# ── Request/Response Models ────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int | None = None

class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: Usage


# ── Auth Middleware ─────────────────────────────────────────────────

def verify_api_key(request: Request) -> str:
    """Verify the API key and return the client identifier."""
    if not CIVIC_AI_API_KEY:
        # No key configured — open access (dev mode)
        return request.client.host if request.client else "unknown"

    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    token = auth[7:]
    if token != CIVIC_AI_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return request.client.host if request.client else "unknown"


# ── Governance Pipeline ────────────────────────────────────────────

def run_governance(messages: list[ChatMessage], client_ip: str) -> tuple[list[ChatMessage], bool]:
    """
    Run all governance checks on the messages.
    Returns (governed_messages, blocked).
    """
    governed = []
    any_pii = False

    for msg in messages:
        # PII redaction on user messages
        if msg.role == "user":
            redacted_text, pii_matches = redactor.redact_text(msg.content)
            if pii_matches:
                any_pii = True
                audit.log_event(
                    event_type=AuditEventType.PII_REDACTION,
                    user_id=client_ip,
                    action=f"Redacted {len(pii_matches)} PII instances",
                    resource="chat-input",
                    pii_detected=True,
                    metadata={"types": [m.pii_type.value for m in pii_matches]},
                )
            governed.append(ChatMessage(role=msg.role, content=redacted_text))
        else:
            governed.append(msg)

    # Safety gates on the last user message
    last_user = next((m for m in reversed(governed) if m.role == "user"), None)
    if last_user:
        passed, results = gates.run_all_gates(prompt_text=last_user.content)
        failed_gates = [r for r in results if r.status == GateStatus.FAILED]

        if failed_gates:
            for gate in failed_gates:
                audit.log_event(
                    event_type=AuditEventType.SAFETY_GATE_TRIGGERED,
                    user_id=client_ip,
                    action=f"Gate '{gate.gate_name}' failed: {', '.join(gate.violations)}",
                    resource="chat-input",
                    result="failure",
                    severity=AuditSeverity.WARNING,
                )
            return governed, True  # blocked

    return governed, False


# ── LLM Forwarding ─────────────────────────────────────────────────

async def forward_to_llm(request: ChatRequest, governed_messages: list[ChatMessage]) -> dict:
    """Forward the governed request to the actual LLM."""
    payload = {
        "model": request.model or LLM_DEFAULT_MODEL,
        "messages": [{"role": m.role, "content": m.content} for m in governed_messages],
        "temperature": request.temperature,
        "stream": False,
    }
    if request.max_tokens:
        payload["max_tokens"] = request.max_tokens

    headers = {"Content-Type": "application/json"}
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"

    response = await http_client.post(
        f"{LLM_BASE_URL}/chat/completions",
        json=payload,
        headers=headers,
    )
    response.raise_for_status()
    return response.json()


async def forward_to_llm_stream(request: ChatRequest, governed_messages: list[ChatMessage]) -> AsyncIterator[bytes]:
    """Forward the governed request to the LLM with SSE streaming."""
    payload = {
        "model": request.model or LLM_DEFAULT_MODEL,
        "messages": [{"role": m.role, "content": m.content} for m in governed_messages],
        "temperature": request.temperature,
        "stream": True,
    }
    if request.max_tokens:
        payload["max_tokens"] = request.max_tokens

    headers = {"Content-Type": "application/json"}
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"

    async with http_client.stream(
        "POST",
        f"{LLM_BASE_URL}/chat/completions",
        json=payload,
        headers=headers,
    ) as response:
        response.raise_for_status()
        async for chunk in response.aiter_bytes():
            yield chunk


# ── Endpoints ──────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "provider": LLM_PROVIDER, "governance": "active"}


@app.get("/v1/models")
async def list_models():
    """Proxy model list from the LLM provider."""
    try:
        headers = {}
        if LLM_API_KEY:
            headers["Authorization"] = f"Bearer {LLM_API_KEY}"
        response = await http_client.get(f"{LLM_BASE_URL}/models", headers=headers)
        return response.json()
    except Exception as e:
        return {"object": "list", "data": [{"id": LLM_DEFAULT_MODEL, "object": "model"}]}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest, raw_request: Request):
    """OpenAI-compatible chat completions with governance."""
    client_ip = verify_api_key(raw_request)

    # Rate limiting
    if not rate_limiter.check(client_ip):
        audit.log_event(
            event_type=AuditEventType.ACCESS_DENIED,
            user_id=client_ip,
            action="Rate limit exceeded",
            resource="chat-completions",
            result="failure",
            severity=AuditSeverity.WARNING,
        )
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    # Governance pipeline
    governed_messages, blocked = run_governance(request.messages, client_ip)

    if blocked:
        raise HTTPException(
            status_code=422,
            detail="Request blocked by safety gates. The prompt may contain injection attempts or other policy violations.",
        )

    # Audit the request
    audit.log_event(
        event_type=AuditEventType.PROMPT_EXECUTION,
        user_id=client_ip,
        action="Chat completion",
        resource=request.model or LLM_DEFAULT_MODEL,
        ip_address=client_ip,
        metadata={
            "message_count": len(request.messages),
            "stream": request.stream,
        },
    )

    # Forward to LLM
    try:
        if request.stream:
            return StreamingResponse(
                forward_to_llm_stream(request, governed_messages),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )

        llm_response = await forward_to_llm(request, governed_messages)
        return llm_response

    except httpx.HTTPStatusError as e:
        audit.log_event(
            event_type=AuditEventType.ERROR,
            user_id=client_ip,
            action=f"LLM error: {e.response.status_code}",
            resource=request.model or LLM_DEFAULT_MODEL,
            result="failure",
            severity=AuditSeverity.ERROR,
        )
        raise HTTPException(status_code=502, detail=f"LLM provider error: {e.response.status_code}")
    except Exception as e:
        audit.log_event(
            event_type=AuditEventType.ERROR,
            user_id=client_ip,
            action=f"Unexpected error: {str(e)}",
            resource=request.model or LLM_DEFAULT_MODEL,
            result="failure",
            severity=AuditSeverity.ERROR,
        )
        raise HTTPException(status_code=500, detail="Internal server error")
