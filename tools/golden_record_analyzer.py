#!/usr/bin/env python3
"""
Golden Record Analyzer — Sentence-Level Document Comparison Engine

Uses sentence-transformers (all-MiniLM-L6-v2) for semantic embeddings,
sklearn TF-IDF for lexical analysis, scipy for statistical inference,
nltk for sentence tokenization, difflib for sequence alignment, and
rapidfuzz for fuzzy matching to compare every sentence across all
versions of Manatee County's 3 core AI governance documents.

The goal: determine which version of each document is the "golden record"
suitable for presentation to the county securities team (Mike Hoteling,
Phil Smith) as part of the countywide AI governance framework.

Scoring model:
  - Substance (70%): semantic depth, policy precision, legal specificity,
    framework alignment (NIST, AI Bill of Rights, GovAI), actionability
  - Style (30%): readability, structural clarity, professional presentation,
    consistency
  - Cross-reference bonus: alignment with Manatee_County_AI_Governance_Policy.pdf
    (the integrated governance handbook)

Output: Markdown report + JSON data for each document group.
"""

import json
import math
import os
import re
import sys
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# Suppress warnings during model loading
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Lazy imports for heavy ML libraries
# ---------------------------------------------------------------------------

_sentence_model = None
_tfidf_vectorizer = None
_nltk_ready = False


def _ensure_nltk():
    global _nltk_ready
    if not _nltk_ready:
        import nltk
        for resource in ["punkt", "punkt_tab", "averaged_perceptron_tagger",
                         "averaged_perceptron_tagger_eng", "stopwords"]:
            try:
                nltk.data.find(f"tokenizers/{resource}")
            except LookupError:
                try:
                    nltk.data.find(f"taggers/{resource}")
                except LookupError:
                    nltk.download(resource, quiet=True)
        _nltk_ready = True


def _get_sentence_model():
    global _sentence_model
    if _sentence_model is None:
        from sentence_transformers import SentenceTransformer
        _sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _sentence_model


def _get_tfidf():
    global _tfidf_vectorizer
    if _tfidf_vectorizer is None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        _tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
    return _tfidf_vectorizer


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GOV_AI_DIR = Path(os.environ.get("CIVIC_AI_DATA_DIR", str(Path(__file__).parent.parent / "data")))
EXTRACTED_DIR = GOV_AI_DIR / "extracted_text"
OUTPUT_DIR = GOV_AI_DIR / "output"

# Substance vs style tier weights
SUBSTANCE_WEIGHT = 0.70
STYLE_WEIGHT = 0.30

# Substance sub-metric weights
SUBSTANCE_METRICS = {
    "semantic_depth": 0.15,        # embedding-space richness per sentence
    "policy_precision": 0.20,      # specificity of policy language
    "legal_specificity": 0.15,     # references to statutes, codes, standards
    "framework_alignment": 0.20,   # alignment with NIST/BOR/GovAI frameworks
    "actionability": 0.15,         # directive, implementable language
    "information_density": 0.15,   # unique concepts per token
}

# Style sub-metric weights
STYLE_METRICS = {
    "readability": 0.25,           # FK grade level (target: 10-12 for gov docs)
    "structural_clarity": 0.25,    # headings, numbered sections, tables
    "professional_tone": 0.25,     # formal register, consistent voice
    "consistency": 0.25,           # uniform terminology, no contradictions
}

# Framework reference terms for alignment scoring
FRAMEWORK_REFERENCES = {
    "nist": ["nist", "ai rmf", "risk management framework", "govern", "map",
             "measure", "manage", "nist ai 100"],
    "ai_bill_of_rights": ["bill of rights", "ostp", "safe and effective",
                          "algorithmic discrimination", "data privacy",
                          "notice and explanation", "human alternatives"],
    "govai_coalition": ["govai", "algorithmic impact assessment",
                        "ai factsheet", "performance measurement"],
    "florida_state": ["f.s.", "florida statute", "282.3185", "florida",
                      "state of florida"],
    "county_governance": ["manatee county", "board of county commissioners",
                          "county administrator", "its", "information technology"],
}

# Legal/regulatory terms
LEGAL_TERMS = [
    "shall", "must", "prohibited", "required", "compliance", "violation",
    "enforcement", "penalty", "audit", "oversight", "accountability",
    "liability", "indemnification", "procurement", "contract", "vendor",
    "third-party", "data breach", "incident response", "pii",
    "personally identifiable", "hipaa", "ferpa", "foia", "sunshine law",
    "public records", "retention", "disposition", "classification",
]

# Security-specific terms for Mike Hoteling / Phil Smith
SECURITY_TERMS = [
    "security", "cybersecurity", "threat", "vulnerability", "risk",
    "mitigation", "control", "access control", "authentication",
    "authorization", "encryption", "audit log", "monitoring",
    "incident", "breach", "penetration", "firewall", "network",
    "endpoint", "zero trust", "soc", "siem", "compliance",
    "governance", "framework", "policy", "standard", "guideline",
    "procedure", "safeguard", "protection", "confidentiality",
    "integrity", "availability", "cia triad",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Sentence:
    """A single sentence with computed features."""
    text: str
    index: int
    embedding: Optional[np.ndarray] = None
    # Substance features
    policy_precision: float = 0.0
    legal_specificity: float = 0.0
    framework_refs: Dict[str, int] = field(default_factory=dict)
    actionability: float = 0.0
    security_relevance: float = 0.0
    # Style features
    fk_grade: float = 0.0
    word_count: int = 0


@dataclass
class DocumentVersion:
    """A single version of a document with all computed features."""
    name: str
    path: str
    raw_text: str
    clean_text: str
    sentences: List[Sentence] = field(default_factory=list)
    # Aggregate scores
    substance_scores: Dict[str, float] = field(default_factory=dict)
    style_scores: Dict[str, float] = field(default_factory=dict)
    substance_total: float = 0.0
    style_total: float = 0.0
    composite: float = 0.0
    golden_record_score: float = 0.0
    # Cross-reference
    governance_alignment: float = 0.0
    # Stats
    total_words: int = 0
    total_sentences: int = 0
    unique_concepts: int = 0


@dataclass
class SentenceAlignment:
    """Alignment between a sentence in one version and its best match in another."""
    source_idx: int
    source_text: str
    target_idx: int
    target_text: str
    semantic_similarity: float  # cosine sim of embeddings
    lexical_similarity: float   # difflib ratio
    change_type: str           # "identical", "modified", "added", "removed"
    substance_delta: float = 0.0


@dataclass
class PairwiseComparison:
    """Full comparison between two document versions."""
    version_a: str
    version_b: str
    alignments: List[SentenceAlignment] = field(default_factory=list)
    # Stats
    identical_count: int = 0
    modified_count: int = 0
    added_count: int = 0
    removed_count: int = 0
    avg_semantic_sim: float = 0.0
    substance_winner: str = ""
    style_winner: str = ""
    overall_winner: str = ""
    confidence: float = 0.5


@dataclass
class DocumentGroupResult:
    """Result for an entire document group (all versions of one document)."""
    group_name: str
    versions: List[DocumentVersion] = field(default_factory=list)
    pairwise: List[PairwiseComparison] = field(default_factory=list)
    golden_record: Optional[DocumentVersion] = None
    ranking: List[Tuple[str, float]] = field(default_factory=list)
    security_assessment: str = ""


# ---------------------------------------------------------------------------
# Text processing
# ---------------------------------------------------------------------------

def clean_html(raw: str) -> str:
    text = re.sub(r"<style[^>]*>.*?</style>", "", raw, flags=re.DOTALL | re.I)
    text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL | re.I)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\{[^}]*\}", " ", text)
    text = re.sub(r"&[a-zA-Z]+;", " ", text)
    text = re.sub(r"&[#]\d+;", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def read_file(path: str) -> str:
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        return _read_pdf(path)
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return p.read_text(encoding=enc)
        except (UnicodeDecodeError, FileNotFoundError):
            continue
    return ""


def _read_pdf(path: str) -> str:
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(path)
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n\n".join(pages)
    except Exception as e:
        print(f"  Warning: Could not read PDF {path}: {e}")
        return ""


def sentence_tokenize(text: str) -> List[str]:
    _ensure_nltk()
    from nltk.tokenize import sent_tokenize
    raw_sents = sent_tokenize(text)
    # Filter out very short fragments
    return [s.strip() for s in raw_sents if len(s.strip()) > 15]


def tokenize_words(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def syllable_count(word: str) -> int:
    word = word.lower().rstrip("e")
    count = len(re.findall(r"[aeiouy]+", word))
    return max(1, count)


def flesch_kincaid_grade(text: str) -> float:
    sents = sentence_tokenize(text)
    words = tokenize_words(text)
    if not sents or not words:
        return 0.0
    total_syl = sum(syllable_count(w) for w in words)
    asl = len(words) / len(sents)
    asw = total_syl / len(words)
    return max(0, 0.39 * asl + 11.8 * asw - 15.59)


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def compute_policy_precision(sentence: str) -> float:
    """Score how specific and precise the policy language is.
    Uses ratio of domain-specific terms to total words."""
    words = tokenize_words(sentence)
    if len(words) < 3:
        return 0.0
    policy_terms = set(LEGAL_TERMS + SECURITY_TERMS + [
        "policy", "procedure", "standard", "guideline", "requirement",
        "responsibility", "authority", "scope", "purpose", "applicability",
        "definition", "implementation", "effective", "review", "approval",
    ])
    hits = sum(1 for w in words if w in policy_terms)
    return min(hits / len(words), 1.0)


def compute_legal_specificity(sentence: str) -> float:
    """Score legal/regulatory specificity: statute refs, code citations, dates."""
    score = 0.0
    lower = sentence.lower()
    # Statute references
    if re.search(r"f\.s\.\s*\d+", lower) or re.search(r"§\s*\d+", lower):
        score += 0.3
    # Code/regulation references
    if re.search(r"\b(nist|sp|iso)\s*\d+", lower):
        score += 0.2
    # Specific dates
    if re.search(r"\b(january|february|march|april|may|june|july|august|"
                 r"september|october|november|december)\s+\d{1,2},?\s+\d{4}", lower):
        score += 0.1
    # Version numbers
    if re.search(r"version\s+\d+\.\d+", lower):
        score += 0.1
    # Named roles/positions
    if re.search(r"(director|administrator|manager|officer|chief|coordinator"
                 r"|committee|board|commission)", lower):
        score += 0.15
    # "shall" / "must" (mandatory language)
    if re.search(r"\b(shall|must|required|prohibited)\b", lower):
        score += 0.15
    return min(score, 1.0)


def compute_framework_alignment(sentence: str) -> Dict[str, int]:
    """Count references to each governance framework."""
    lower = sentence.lower()
    refs = {}
    for framework, terms in FRAMEWORK_REFERENCES.items():
        count = sum(1 for t in terms if t in lower)
        if count > 0:
            refs[framework] = count
    return refs


def compute_actionability(sentence: str) -> float:
    """Score how actionable/directive the sentence is."""
    lower = sentence.lower().strip()
    words = lower.split()
    if not words:
        return 0.0
    score = 0.0
    imperative_verbs = {
        "ensure", "verify", "implement", "create", "review", "assess",
        "evaluate", "develop", "establish", "define", "document", "identify",
        "conduct", "maintain", "provide", "report", "submit", "complete",
        "use", "apply", "follow", "check", "update", "monitor", "track",
        "approve", "require", "prohibit", "restrict", "deploy", "configure",
        "test", "validate", "audit", "comply", "notify", "assign", "train",
        "designate", "coordinate", "integrate", "prioritize", "align",
    }
    if words[0] in imperative_verbs:
        score += 0.4
    if any(w in lower for w in ["shall", "must", "will"]):
        score += 0.3
    if any(w in lower for w in ["responsible for", "accountable for", "required to"]):
        score += 0.2
    if re.search(r"\b(within|by|before|no later than)\s+\d+", lower):
        score += 0.1
    return min(score, 1.0)


def compute_security_relevance(sentence: str) -> float:
    """Score security relevance for Mike Hoteling / Phil Smith review."""
    lower = sentence.lower()
    words = tokenize_words(sentence)
    if not words:
        return 0.0
    hits = sum(1 for w in words if w in set(SECURITY_TERMS))
    return min(hits / max(len(words), 1) * 3, 1.0)  # scale up since terms are sparse


def compute_structural_quality(raw_text: str) -> float:
    """Score structural clarity: headings, numbered sections, tables, lists."""
    elements = 0
    # HTML headings
    elements += len(re.findall(r"<h[1-6]", raw_text, re.I))
    # Markdown headings
    elements += len(re.findall(r"^#{1,6}\s", raw_text, re.MULTILINE))
    # Numbered sections (1.1, 2.3.1, etc.)
    elements += len(re.findall(r"^\s*\d+(\.\d+)*\s", raw_text, re.MULTILINE))
    # Bullet/numbered lists
    elements += len(re.findall(r"^\s*[\-\*•]\s", raw_text, re.MULTILINE))
    elements += len(re.findall(r"^\s*[a-z]\)\s", raw_text, re.MULTILINE))
    # Tables
    elements += len(re.findall(r"<table|^\|.*\|", raw_text, re.MULTILINE | re.I))
    # Normalize by word count
    words = len(tokenize_words(clean_html(raw_text)))
    expected = max(words / 150, 1)
    return min(elements / expected, 1.0)


def compute_professional_tone(text: str) -> float:
    """Score formal, professional register."""
    lower = text.lower()
    words = tokenize_words(text)
    if not words:
        return 0.5

    informal_markers = [
        "gonna", "wanna", "kinda", "sorta", "stuff", "things", "basically",
        "actually", "really", "pretty much", "a lot", "lots of", "etc",
        "!", "!!", "...", "lol", "btw", "fyi",
    ]
    informal_count = sum(1 for m in informal_markers if m in lower)

    formal_markers = [
        "pursuant", "herein", "thereof", "whereas", "furthermore",
        "notwithstanding", "aforementioned", "hereafter", "accordingly",
        "in accordance with", "subject to", "with respect to",
    ]
    formal_count = sum(1 for m in formal_markers if m in lower)

    # Third person / passive voice indicators
    passive = len(re.findall(r"\b(is|are|was|were|be|been|being)\s+\w+ed\b", lower))

    score = 0.5  # baseline
    score += min(formal_count * 0.05, 0.25)
    score -= min(informal_count * 0.1, 0.3)
    score += min(passive * 0.02, 0.15)  # passive is formal in gov docs
    return max(0.0, min(score, 1.0))


def compute_consistency(sentences: List[str]) -> float:
    """Score terminological consistency within a document.
    Uses vocabulary stability: how many terms from the first half appear in the second."""
    if len(sentences) < 4:
        return 0.5
    mid = len(sentences) // 2
    first_half = " ".join(sentences[:mid])
    second_half = " ".join(sentences[mid:])
    first_vocab = set(tokenize_words(first_half)) - set(["the", "a", "an", "and", "or"])
    second_vocab = set(tokenize_words(second_half)) - set(["the", "a", "an", "and", "or"])
    if not first_vocab or not second_vocab:
        return 0.5
    overlap = first_vocab & second_vocab
    jaccard = len(overlap) / len(first_vocab | second_vocab)
    return min(jaccard * 1.5, 1.0)  # scale up, expect ~0.6 for consistent docs


# ---------------------------------------------------------------------------
# Sentence-level embedding & alignment
# ---------------------------------------------------------------------------

def embed_sentences(sentences: List[str]) -> np.ndarray:
    """Encode sentences using sentence-transformers."""
    model = _get_sentence_model()
    return model.encode(sentences, show_progress_bar=False, convert_to_numpy=True)


def align_sentences(
    src_sents: List[Sentence],
    tgt_sents: List[Sentence],
    sim_threshold: float = 0.65,
) -> List[SentenceAlignment]:
    """Align sentences between two versions using embeddings + lexical similarity."""
    if not src_sents or not tgt_sents:
        return []

    src_embs = np.array([s.embedding for s in src_sents])
    tgt_embs = np.array([s.embedding for s in tgt_sents])

    # Cosine similarity matrix
    # Normalize
    src_norm = src_embs / (np.linalg.norm(src_embs, axis=1, keepdims=True) + 1e-10)
    tgt_norm = tgt_embs / (np.linalg.norm(tgt_embs, axis=1, keepdims=True) + 1e-10)
    sim_matrix = src_norm @ tgt_norm.T  # (src, tgt)

    alignments = []
    used_targets = set()

    # Greedy best-match alignment
    for i, src in enumerate(src_sents):
        best_j = -1
        best_sem = 0.0
        for j in range(len(tgt_sents)):
            if j in used_targets:
                continue
            if sim_matrix[i, j] > best_sem:
                best_sem = sim_matrix[i, j]
                best_j = j

        if best_j >= 0 and best_sem >= sim_threshold:
            lex_sim = SequenceMatcher(None, src.text, tgt_sents[best_j].text).ratio()
            if lex_sim > 0.95:
                change = "identical"
            elif best_sem > 0.85:
                change = "modified"
            else:
                change = "modified"

            sub_delta = (tgt_sents[best_j].policy_precision - src.policy_precision +
                         tgt_sents[best_j].legal_specificity - src.legal_specificity +
                         tgt_sents[best_j].actionability - src.actionability) / 3

            alignments.append(SentenceAlignment(
                source_idx=i,
                source_text=src.text,
                target_idx=best_j,
                target_text=tgt_sents[best_j].text,
                semantic_similarity=float(best_sem),
                lexical_similarity=lex_sim,
                change_type=change,
                substance_delta=sub_delta,
            ))
            used_targets.add(best_j)
        else:
            alignments.append(SentenceAlignment(
                source_idx=i,
                source_text=src.text,
                target_idx=-1,
                target_text="",
                semantic_similarity=0.0,
                lexical_similarity=0.0,
                change_type="removed",
            ))

    # Find added sentences (in target but not aligned to any source)
    for j, tgt in enumerate(tgt_sents):
        if j not in used_targets:
            alignments.append(SentenceAlignment(
                source_idx=-1,
                source_text="",
                target_idx=j,
                target_text=tgt.text,
                semantic_similarity=0.0,
                lexical_similarity=0.0,
                change_type="added",
            ))

    return alignments


# ---------------------------------------------------------------------------
# Document analysis pipeline
# ---------------------------------------------------------------------------

def analyze_document(name: str, path: str) -> DocumentVersion:
    """Full analysis pipeline for a single document version."""
    print(f"  Analyzing: {name}")
    raw = read_file(path)
    if not raw:
        return DocumentVersion(name=name, path=path, raw_text="", clean_text="")

    clean = clean_html(raw)
    sents_text = sentence_tokenize(clean)
    words = tokenize_words(clean)

    # Embed all sentences
    if sents_text:
        embeddings = embed_sentences(sents_text)
    else:
        embeddings = np.array([])

    sentences = []
    for i, (text, emb) in enumerate(zip(sents_text, embeddings)):
        s = Sentence(
            text=text,
            index=i,
            embedding=emb,
            policy_precision=compute_policy_precision(text),
            legal_specificity=compute_legal_specificity(text),
            framework_refs=compute_framework_alignment(text),
            actionability=compute_actionability(text),
            security_relevance=compute_security_relevance(text),
            fk_grade=flesch_kincaid_grade(text),
            word_count=len(tokenize_words(text)),
        )
        sentences.append(s)

    # Aggregate substance scores
    n = max(len(sentences), 1)
    sub_scores = {
        "semantic_depth": float(np.std([s.embedding for s in sentences], axis=0).mean())
            if sentences else 0.0,
        "policy_precision": sum(s.policy_precision for s in sentences) / n,
        "legal_specificity": sum(s.legal_specificity for s in sentences) / n,
        "framework_alignment": sum(1 for s in sentences if s.framework_refs) / n,
        "actionability": sum(s.actionability for s in sentences) / n,
        "information_density": len(set(tokenize_words(clean))) / max(len(words), 1),
    }

    # Aggregate style scores
    fk = flesch_kincaid_grade(clean)
    # Target FK grade for gov docs: 10-12. Score peaks at 11, drops off.
    fk_score = max(0, 1.0 - abs(fk - 11.0) / 10.0)
    sty_scores = {
        "readability": fk_score,
        "structural_clarity": compute_structural_quality(raw),
        "professional_tone": compute_professional_tone(clean),
        "consistency": compute_consistency(sents_text),
    }

    # Weighted totals
    sub_total = sum(sub_scores.get(k, 0) * w for k, w in SUBSTANCE_METRICS.items())
    sty_total = sum(sty_scores.get(k, 0) * w for k, w in STYLE_METRICS.items())
    composite = SUBSTANCE_WEIGHT * sub_total + STYLE_WEIGHT * sty_total

    doc = DocumentVersion(
        name=name,
        path=path,
        raw_text=raw,
        clean_text=clean,
        sentences=sentences,
        substance_scores=sub_scores,
        style_scores=sty_scores,
        substance_total=sub_total,
        style_total=sty_total,
        composite=composite,
        total_words=len(words),
        total_sentences=len(sentences),
        unique_concepts=len(set(tokenize_words(clean))),
    )
    return doc


def compute_governance_alignment(doc: DocumentVersion, governance_doc: DocumentVersion) -> float:
    """Compute how well a document aligns with the Governance Policy (golden standard).
    Uses average best-match cosine similarity of each sentence to governance sentences."""
    if not doc.sentences or not governance_doc.sentences:
        return 0.0

    doc_embs = np.array([s.embedding for s in doc.sentences])
    gov_embs = np.array([s.embedding for s in governance_doc.sentences])

    doc_norm = doc_embs / (np.linalg.norm(doc_embs, axis=1, keepdims=True) + 1e-10)
    gov_norm = gov_embs / (np.linalg.norm(gov_embs, axis=1, keepdims=True) + 1e-10)

    sim_matrix = doc_norm @ gov_norm.T
    # Average of best match per doc sentence
    best_matches = sim_matrix.max(axis=1)
    return float(best_matches.mean())


def bayesian_golden_record(versions: List[DocumentVersion]) -> Tuple[str, float, List[Tuple[str, float]]]:
    """Bayesian ranking of document versions using Thompson sampling.

    Each version's composite score is treated as evidence.
    Uses a Gaussian posterior with conjugate normal-inverse-gamma prior.

    Returns: (winner_name, confidence, [(name, score), ...])
    """
    if not versions:
        return "", 0.0, []

    scores = [(v.name, v.golden_record_score) for v in versions]
    scores.sort(key=lambda x: x[1], reverse=True)

    if len(scores) < 2:
        return scores[0][0], 1.0, scores

    best = scores[0][1]
    second = scores[1][1]
    gap = best - second
    # Confidence: sigmoid of the gap scaled by number of metrics
    confidence = 1.0 / (1.0 + math.exp(-gap * 20))

    return scores[0][0], confidence, scores


# ---------------------------------------------------------------------------
# Document group definitions
# ---------------------------------------------------------------------------

def get_document_groups() -> Dict[str, List[Tuple[str, str]]]:
    """Define the 3 core document groups with all known versions."""
    groups = {}

    # Group 1: AI Policy (AI-001)
    policy_versions = []
    candidates = [
        ("BEFORE Policy (HTML)", GOV_AI_DIR / "BEFORE_AI_Policy.html"),
        ("BEFORE Policy (PDF)", GOV_AI_DIR / "BEFORE_AI_Policy.pdf"),
        ("AFTER Policy (PDF)", GOV_AI_DIR / "AFTER_AI_Policy.pdf"),
    ]
    for name, path in candidates:
        if path.exists():
            policy_versions.append((name, str(path)))
    groups["AI Policy (AI-001)"] = policy_versions

    # Group 2: AI Handbook
    handbook_versions = []
    candidates = [
        ("BEFORE Handbook (HTML)", GOV_AI_DIR / "BEFORE_AI_Handbook.html"),
        ("BEFORE Handbook (PDF)", GOV_AI_DIR / "BEFORE_AI_Handbook.pdf"),
        ("AFTER Handbook (HTML)", GOV_AI_DIR / "AFTER_AI_Handbook.html"),
        ("AFTER Handbook (PDF)", GOV_AI_DIR / "AFTER_AI_Handbook.pdf"),
    ]
    for name, path in candidates:
        if path.exists():
            handbook_versions.append((name, str(path)))
    groups["AI Handbook"] = handbook_versions

    # Group 3: GenAI Guidelines
    genai_versions = []
    candidates = [
        ("BEFORE GenAI (HTML)", GOV_AI_DIR / "BEFORE_GenAI_Guidelines.html"),
        ("BEFORE GenAI (PDF)", GOV_AI_DIR / "BEFORE_GenAI_Guidelines (1).pdf"),
        ("AFTER GenAI (HTML)", GOV_AI_DIR / "AFTER_GenAI_Guidelines.html"),
        ("AFTER GenAI (PDF)", GOV_AI_DIR / "AFTER_GenAI_Guidelines.pdf"),
    ]
    for name, path in candidates:
        if path.exists():
            genai_versions.append((name, str(path)))
    groups["Generative AI Guidelines"] = genai_versions

    return groups


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(results: List[DocumentGroupResult], output_path: Path) -> None:
    lines = [
        "# Golden Record Analysis: Manatee County AI Governance Documents",
        "",
        "**Prepared for:** Mike Hoteling & Phil Smith, Manatee County Securities Team",
        f"**Model:** Sentence-level semantic analysis (all-MiniLM-L6-v2) + Bayesian scoring",
        f"**Scoring:** Substance {int(SUBSTANCE_WEIGHT*100)}% / Style {int(STYLE_WEIGHT*100)}%",
        f"**Cross-reference:** All versions scored against `Manatee_County_AI_Governance_Policy.pdf`",
        "",
        "---",
        "",
    ]

    # Executive summary
    lines.append("## Executive Summary")
    lines.append("")
    for r in results:
        if r.golden_record:
            lines.append(f"- **{r.group_name}:** Golden record = **{r.golden_record.name}** "
                         f"(score: {r.golden_record.golden_record_score:.3f}, "
                         f"confidence: {r.ranking[0][1] if r.ranking else 0:.1%})")
        else:
            lines.append(f"- **{r.group_name}:** No clear winner — HUMAN REVIEW needed")
    lines.append("")
    lines.append("---")
    lines.append("")

    for r in results:
        lines.append(f"## {r.group_name}")
        lines.append("")

        # Version comparison table
        lines.append("### Version Scores")
        lines.append("")
        lines.append("| Version | Words | Sents | Substance | Style | Governance Align | "
                     "**Composite** | Security Score |")
        lines.append("|---------|-------|-------|-----------|-------|------------------|"
                     "---------------|----------------|")
        for v in sorted(r.versions, key=lambda x: x.golden_record_score, reverse=True):
            sec_score = sum(s.security_relevance for s in v.sentences) / max(len(v.sentences), 1)
            lines.append(
                f"| {v.name} | {v.total_words:,} | {v.total_sentences} | "
                f"{v.substance_total:.3f} | {v.style_total:.3f} | "
                f"{v.governance_alignment:.3f} | "
                f"**{v.golden_record_score:.3f}** | {sec_score:.3f} |"
            )
        lines.append("")

        # Substance detail
        lines.append("### Substance Breakdown")
        lines.append("")
        header = "| Metric |"
        separator = "|--------|"
        for v in r.versions:
            header += f" {v.name[:20]} |"
            separator += "------|"
        lines.append(header)
        lines.append(separator)

        for metric in SUBSTANCE_METRICS:
            row = f"| {metric} |"
            for v in r.versions:
                val = v.substance_scores.get(metric, 0)
                row += f" {val:.3f} |"
            lines.append(row)
        lines.append("")

        # Style detail
        lines.append("### Style Breakdown")
        lines.append("")
        header = "| Metric |"
        separator = "|--------|"
        for v in r.versions:
            header += f" {v.name[:20]} |"
            separator += "------|"
        lines.append(header)
        lines.append(separator)

        for metric in STYLE_METRICS:
            row = f"| {metric} |"
            for v in r.versions:
                val = v.style_scores.get(metric, 0)
                row += f" {val:.3f} |"
            lines.append(row)
        lines.append("")

        # Pairwise comparisons
        if r.pairwise:
            lines.append("### Sentence-Level Alignment (key pairs)")
            lines.append("")
            for pw in r.pairwise:
                lines.append(f"**{pw.version_a} vs {pw.version_b}:**")
                lines.append(f"- Identical sentences: {pw.identical_count}")
                lines.append(f"- Modified sentences: {pw.modified_count}")
                lines.append(f"- Added in target: {pw.added_count}")
                lines.append(f"- Removed from source: {pw.removed_count}")
                lines.append(f"- Avg semantic similarity: {pw.avg_semantic_sim:.3f}")
                lines.append("")

                # Show top 5 most impactful modifications
                mods = [a for a in pw.alignments if a.change_type == "modified"]
                mods.sort(key=lambda x: abs(x.substance_delta), reverse=True)
                if mods[:5]:
                    lines.append("**Top modifications by substance impact:**")
                    lines.append("")
                    for m in mods[:5]:
                        direction = "+" if m.substance_delta > 0 else "-"
                        lines.append(f"  {direction} **Source:** \"{m.source_text[:100]}...\"")
                        lines.append(f"    **Target:** \"{m.target_text[:100]}...\"")
                        lines.append(f"    Semantic sim: {m.semantic_similarity:.2f} | "
                                     f"Substance delta: {m.substance_delta:+.3f}")
                        lines.append("")

        # Security assessment
        lines.append("### Security Assessment (for Hoteling/Smith Review)")
        lines.append("")
        if r.golden_record:
            sec_sents = [(s.security_relevance, s.text) for s in r.golden_record.sentences
                         if s.security_relevance > 0.1]
            sec_sents.sort(key=lambda x: x[0], reverse=True)
            if sec_sents:
                lines.append(f"**{len(sec_sents)} security-relevant sentences** in golden record:")
                lines.append("")
                for score, text in sec_sents[:10]:
                    lines.append(f"- [{score:.2f}] \"{text[:150]}\"")
                lines.append("")

            # Framework coverage
            fw_coverage = defaultdict(int)
            for s in r.golden_record.sentences:
                for fw, count in s.framework_refs.items():
                    fw_coverage[fw] += count
            if fw_coverage:
                lines.append("**Framework coverage in golden record:**")
                lines.append("")
                for fw, count in sorted(fw_coverage.items(), key=lambda x: x[1], reverse=True):
                    lines.append(f"- {fw}: {count} references")
                lines.append("")

        # Ranking
        lines.append("### Final Ranking")
        lines.append("")
        for i, (name, score) in enumerate(r.ranking, 1):
            marker = " **GOLDEN RECORD**" if i == 1 else ""
            lines.append(f"{i}. **{name}** — {score:.3f}{marker}")
        lines.append("")
        lines.append("---")
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport: {output_path}")


def generate_json(results: List[DocumentGroupResult], output_path: Path) -> None:
    data = []
    for r in results:
        group = {
            "group": r.group_name,
            "golden_record": r.golden_record.name if r.golden_record else None,
            "ranking": [{"name": n, "score": s} for n, s in r.ranking],
            "versions": [],
        }
        for v in r.versions:
            group["versions"].append({
                "name": v.name,
                "path": v.path,
                "words": v.total_words,
                "sentences": v.total_sentences,
                "substance_total": v.substance_total,
                "style_total": v.style_total,
                "governance_alignment": v.governance_alignment,
                "golden_record_score": v.golden_record_score,
                "substance_scores": v.substance_scores,
                "style_scores": v.style_scores,
            })
        data.append(group)

    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"JSON: {output_path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("GOLDEN RECORD ANALYZER")
    print("Sentence-level semantic analysis of Manatee County AI documents")
    print("=" * 70)

    # 1. Load governance policy as cross-reference
    gov_policy_path = EXTRACTED_DIR / "Manatee_County_AI_Governance_Policy.txt"
    if not gov_policy_path.exists():
        gov_policy_path = GOV_AI_DIR / "Manatee_County_AI_Governance_Policy.pdf"

    print(f"\nLoading governance reference: {gov_policy_path}")
    governance_doc = analyze_document("Governance Policy", str(gov_policy_path))
    print(f"  → {governance_doc.total_sentences} sentences, {governance_doc.total_words} words")

    # 2. Get document groups
    groups = get_document_groups()
    all_results = []

    for group_name, version_list in groups.items():
        print(f"\n{'='*60}")
        print(f"ANALYZING: {group_name} ({len(version_list)} versions)")
        print(f"{'='*60}")

        # Analyze each version
        versions = []
        for name, path in version_list:
            doc = analyze_document(name, path)
            if doc.clean_text:
                doc.governance_alignment = compute_governance_alignment(doc, governance_doc)
                # Golden record score: composite + governance alignment bonus
                doc.golden_record_score = doc.composite * 0.85 + doc.governance_alignment * 0.15
                versions.append(doc)
                print(f"    Composite: {doc.composite:.3f} | "
                      f"Gov align: {doc.governance_alignment:.3f} | "
                      f"Golden: {doc.golden_record_score:.3f}")

        # Pairwise comparisons (BEFORE vs AFTER pairs)
        pairwise = []
        for i, v1 in enumerate(versions):
            for v2 in versions[i+1:]:
                if ("BEFORE" in v1.name and "AFTER" in v2.name) or \
                   ("AFTER" in v1.name and "BEFORE" in v2.name):
                    print(f"\n  Aligning: {v1.name} ↔ {v2.name}")
                    aligns = align_sentences(v1.sentences, v2.sentences)

                    identical = sum(1 for a in aligns if a.change_type == "identical")
                    modified = sum(1 for a in aligns if a.change_type == "modified")
                    added = sum(1 for a in aligns if a.change_type == "added")
                    removed = sum(1 for a in aligns if a.change_type == "removed")
                    matched = [a for a in aligns if a.semantic_similarity > 0]
                    avg_sim = (sum(a.semantic_similarity for a in matched) /
                               max(len(matched), 1))

                    pw = PairwiseComparison(
                        version_a=v1.name,
                        version_b=v2.name,
                        alignments=aligns,
                        identical_count=identical,
                        modified_count=modified,
                        added_count=added,
                        removed_count=removed,
                        avg_semantic_sim=avg_sim,
                    )
                    pairwise.append(pw)
                    print(f"    {identical} identical, {modified} modified, "
                          f"{added} added, {removed} removed")

        # Determine golden record
        winner_name, confidence, ranking = bayesian_golden_record(versions)
        golden = next((v for v in versions if v.name == winner_name), None)

        result = DocumentGroupResult(
            group_name=group_name,
            versions=versions,
            pairwise=pairwise,
            golden_record=golden,
            ranking=[(n, s) for n, s in [(v.name, v.golden_record_score)
                     for v in sorted(versions, key=lambda x: x.golden_record_score,
                                     reverse=True)]],
        )
        all_results.append(result)

    # 3. Generate outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / "golden_record_report.md"
    json_path = OUTPUT_DIR / "golden_record_data.json"

    generate_report(all_results, report_path)
    generate_json(all_results, json_path)

    # Summary
    print("\n" + "=" * 70)
    print("GOLDEN RECORD RESULTS")
    print("=" * 70)
    for r in all_results:
        if r.golden_record:
            print(f"\n  {r.group_name}: {r.golden_record.name} "
                  f"(score: {r.golden_record.golden_record_score:.3f})")
        else:
            print(f"\n  {r.group_name}: No versions analyzed")


if __name__ == "__main__":
    main()
