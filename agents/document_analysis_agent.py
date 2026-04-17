"""
Document Analysis Agent — RAG over county policies and ordinances
Ingests county documents and provides conversational search with source citations.

Search is hybrid: semantic (Ollama embeddings) + keyword, blended 70/30. If the
embeddings gateway is unavailable or numpy is missing, search falls back to
keyword-only. Vectors are cached per content hash in `<docs_dir>/.index.npz`.
"""

import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from agents.base_agent import AgentContext, AgentResult, BaseAgent

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore[assignment]
    _HAS_NUMPY = False

SEMANTIC_WEIGHT = 0.7
KEYWORD_WEIGHT = 0.3
CACHE_FILENAME = ".index.npz"


class DocumentAnalysisAgent(BaseAgent):
    """
    RAG agent for county document analysis.

    Capabilities:
    - Ingest PDF/text documents into a searchable index
    - Keyword and semantic search across all ingested documents
    - Answer questions with source citations
    - Summarize documents or sections
    """

    def __init__(
        self,
        model_client: Any = None,
        docs_dir: Optional[Path] = None,
        embeddings_gateway: Any = None,
    ):
        self.docs_dir = docs_dir or Path(__file__).parent.parent / "knowledge_base"
        self.document_index: List[Dict[str, Any]] = []
        self.embeddings_gateway = embeddings_gateway
        self._vectors: Any = None  # (n, d) normalized matrix, or None
        self._vectors_loaded = False
        super().__init__(
            name="Document Analysis Agent",
            description="RAG over county policies, ordinances, and governance documents",
            agent_type="analysis",
            capabilities=[
                "Ingest county documents",
                "Search across document corpus",
                "Answer questions with citations",
                "Summarize documents",
            ],
            model_client=model_client,
        )
        self._auto_ingest()

    def _register_tools(self):
        self.register_tool("ingest_document", self.ingest_document, "Ingest a document into the index")
        self.register_tool("search", self.search, "Search across ingested documents")
        self.register_tool("ask", self.ask, "Ask a question and get a cited answer")
        self.register_tool("summarize", self.summarize, "Summarize a document")
        self.register_tool("list_documents", self.list_documents, "List all ingested documents")

    def _auto_ingest(self):
        """Automatically ingest markdown files from the knowledge base directory"""
        if not self.docs_dir.exists():
            logger.warning(f"Documents directory not found: {self.docs_dir}")
            return

        for md_file in self.docs_dir.glob("*.md"):
            try:
                content = md_file.read_text()
                self._index_document(md_file.name, content, str(md_file))
            except Exception as e:
                logger.warning(f"Failed to ingest {md_file.name}: {e}")

        logger.info(f"Auto-ingested {len(self.document_index)} document chunks")

    def _index_document(self, name: str, content: str, source_path: str):
        """Split a document into chunks and add to the index"""
        # Split on markdown headers
        sections = re.split(r"\n(?=## )", content)

        for i, section in enumerate(sections):
            if not section.strip():
                continue

            # Extract title from first line
            lines = section.strip().split("\n")
            title = lines[0].lstrip("#").strip() if lines else f"{name} section {i}"

            content = section.strip()
            chunk = {
                "id": hashlib.md5(f"{name}:{i}".encode()).hexdigest()[:12],
                "document": name,
                "section_title": title,
                "content": content,
                "content_hash": hashlib.sha256(content.encode()).hexdigest(),
                "source_path": source_path,
                "word_count": len(section.split()),
                "chunk_index": i,
            }
            self.document_index.append(chunk)
        # New or re-ingested content → cached vectors may be stale
        self._vectors_loaded = False
        self._vectors = None

    async def execute(self, context: AgentContext) -> AgentResult:
        request = context.request.lower()

        try:
            if "summarize" in request or "summary" in request:
                doc_name = context.metadata.get("document")
                result = await self.summarize(document=doc_name)
                return AgentResult(success=True, output=result, metadata={"operation": "summarize"})

            elif "list" in request or "documents" in request or "index" in request:
                result = await self.list_documents()
                return AgentResult(success=True, output=result, metadata={"operation": "list"})

            else:
                result = await self.ask(context.request)
                return AgentResult(success=True, output=result, metadata={"operation": "ask"})

        except Exception as e:
            logger.error(f"Document Analysis Agent failed: {e}")
            return AgentResult(success=False, output=None, error=str(e))

    async def ingest_document(self, file_path: str) -> Dict[str, Any]:
        """Ingest a document from a file path"""
        path = Path(file_path)
        if not path.exists():
            return {"success": False, "error": f"File not found: {file_path}"}

        content = path.read_text()
        chunks_before = len(self.document_index)
        self._index_document(path.name, content, str(path))
        chunks_added = len(self.document_index) - chunks_before

        return {
            "success": True,
            "document": path.name,
            "chunks_added": chunks_added,
            "total_chunks": len(self.document_index),
        }

    async def _ensure_vectors(self) -> None:
        """Populate self._vectors. Uses on-disk cache keyed by chunk content hash."""
        if self._vectors_loaded:
            return
        if self.embeddings_gateway is None or not _HAS_NUMPY or not self.document_index:
            self._vectors_loaded = True
            return

        current_ids = [c["id"] for c in self.document_index]
        current_hashes = [c["content_hash"] for c in self.document_index]

        cache_path = self.docs_dir / CACHE_FILENAME
        cached: Dict[str, Any] = {}
        if cache_path.exists():
            try:
                data = np.load(cache_path, allow_pickle=False)
                for cid, chash, vec in zip(data["ids"], data["hashes"], data["vectors"]):
                    cached[f"{cid}:{chash}"] = vec
            except Exception as e:
                logger.warning(f"Vector cache load failed ({cache_path.name}): {e}")

        vectors: List[Optional[Any]] = [None] * len(self.document_index)
        to_embed_idx: List[int] = []
        for i, (cid, chash) in enumerate(zip(current_ids, current_hashes)):
            key = f"{cid}:{chash}"
            if key in cached:
                vectors[i] = np.asarray(cached[key], dtype=np.float32)
            else:
                to_embed_idx.append(i)

        if to_embed_idx:
            texts = [self.document_index[i]["content"] for i in to_embed_idx]
            result = await self.embeddings_gateway.embed(texts)
            if not result.get("success"):
                logger.warning(
                    f"Embeddings unavailable — keyword-only search. "
                    f"{result.get('error', 'unknown error')}"
                )
                self._vectors_loaded = True
                return
            fresh = result.get("vectors") or []
            if len(fresh) != len(to_embed_idx):
                logger.warning("Embedding response length mismatch; skipping semantic path.")
                self._vectors_loaded = True
                return
            for j, i in enumerate(to_embed_idx):
                vectors[i] = np.asarray(fresh[j], dtype=np.float32)

        try:
            matrix = np.stack(vectors)
        except ValueError as e:
            logger.warning(f"Vector stack failed: {e}")
            self._vectors_loaded = True
            return

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._vectors = matrix / norms
        self._vectors_loaded = True

        try:
            np.savez(
                cache_path,
                ids=np.array(current_ids),
                hashes=np.array(current_hashes),
                vectors=matrix,
            )
        except Exception as e:
            logger.warning(f"Vector cache save failed: {e}")

    def _keyword_scores(self, keywords: List[str]) -> List[float]:
        scores: List[float] = []
        for chunk in self.document_index:
            content_lower = chunk["content"].lower()
            title_lower = chunk["section_title"].lower()
            score = 0.0
            for kw in keywords:
                score += content_lower.count(kw)
                if kw in title_lower:
                    score += 3.0
            scores.append(score)
        max_score = max(scores) if scores else 0.0
        if max_score > 0:
            scores = [s / max_score for s in scores]
        return scores

    async def _semantic_scores(self, query: str) -> Optional[List[float]]:
        if self._vectors is None or self.embeddings_gateway is None or not _HAS_NUMPY:
            return None
        result = await self.embeddings_gateway.embed([query])
        if not result.get("success") or not result.get("vectors"):
            return None
        q = np.asarray(result["vectors"][0], dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return None
        q = q / q_norm
        sims = self._vectors @ q  # cosine since both normalized
        return [max(0.0, float(s)) for s in sims]

    async def search(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Hybrid search: semantic (70%) + keyword (30%). Falls back to keyword-only."""
        await self._ensure_vectors()

        keywords = [k for k in query.lower().split() if len(k) > 2]
        keyword_scores = self._keyword_scores(keywords)
        semantic_scores = await self._semantic_scores(query)

        if semantic_scores is not None:
            blended = [
                SEMANTIC_WEIGHT * s + KEYWORD_WEIGHT * k
                for s, k in zip(semantic_scores, keyword_scores)
            ]
            mode = "hybrid"
        else:
            blended = keyword_scores
            mode = "keyword"

        scored_results: List[Dict[str, Any]] = []
        for i, chunk in enumerate(self.document_index):
            if blended[i] <= 0:
                continue
            paragraphs = chunk["content"].split("\n\n")
            best_para = max(
                paragraphs,
                key=lambda p: sum(p.lower().count(kw) for kw in keywords),
                default=chunk["content"][:500],
            )
            entry: Dict[str, Any] = {
                "document": chunk["document"],
                "section": chunk["section_title"],
                "score": blended[i],
                "snippet": best_para[:500],
                "source": chunk["source_path"],
            }
            if mode == "hybrid":
                entry["semantic_score"] = semantic_scores[i]  # type: ignore[index]
                entry["keyword_score"] = keyword_scores[i]
            scored_results.append(entry)

        scored_results.sort(key=lambda x: x["score"], reverse=True)

        return {
            "query": query,
            "results": scored_results[:max_results],
            "total_matches": len(scored_results),
            "documents_searched": len(set(c["document"] for c in self.document_index)),
            "mode": mode,
        }

    async def ask(self, question: str) -> Dict[str, Any]:
        """Answer a question with source citations"""
        search_results = await self.search(question, max_results=3)

        if not search_results["results"]:
            return {
                "question": question,
                "answer": "No relevant information found in the document index.",
                "sources": [],
            }

        # Build context from top results
        context_parts = []
        sources = []
        for r in search_results["results"]:
            context_parts.append(f"[{r['document']} — {r['section']}]\n{r['snippet']}")
            sources.append({"document": r["document"], "section": r["section"]})

        context_text = "\n\n---\n\n".join(context_parts)

        if self.model_client:
            answer = await self.call_llm(
                prompt=f"Based on the following documents, answer this question: {question}\n\n{context_text}",
                system_message="You are a county government policy analyst. Answer based only on the provided documents. Cite which document each fact comes from.",
            )
        else:
            answer = f"Top result from {sources[0]['document']}, section '{sources[0]['section']}':\n\n{search_results['results'][0]['snippet']}"

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
        }

    async def summarize(self, document: Optional[str] = None) -> Dict[str, Any]:
        """Summarize a document or all documents"""
        if document:
            chunks = [c for c in self.document_index if c["document"] == document]
        else:
            chunks = self.document_index

        if not chunks:
            return {"error": f"Document '{document}' not found in index"}

        doc_names = list(set(c["document"] for c in chunks))
        total_words = sum(c["word_count"] for c in chunks)
        sections = [c["section_title"] for c in chunks]

        return {
            "documents": doc_names,
            "total_sections": len(chunks),
            "total_words": total_words,
            "sections": sections[:20],
        }

    async def list_documents(self) -> Dict[str, Any]:
        """List all ingested documents"""
        docs: Dict[str, Dict[str, Any]] = {}
        for chunk in self.document_index:
            name = chunk["document"]
            if name not in docs:
                docs[name] = {"document": name, "sections": 0, "words": 0}
            docs[name]["sections"] += 1
            docs[name]["words"] += chunk["word_count"]

        return {
            "total_documents": len(docs),
            "total_chunks": len(self.document_index),
            "documents": list(docs.values()),
        }
