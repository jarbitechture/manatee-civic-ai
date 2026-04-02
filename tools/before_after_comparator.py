#!/usr/bin/env python3
"""
BEFORE/AFTER Document Comparator with Substance-Weighted Bayesian Model

Compares paired BEFORE/AFTER documents using a two-tier scoring system:
  - Substance metrics (70% weight): semantic change density, information gain,
    specificity delta, actionability delta, signal density delta
  - Style metrics (30% weight): readability (Flesch-Kincaid), formatting quality,
    token efficiency

The Bayesian posterior P(AFTER is better | evidence) uses a beta-binomial
model where each metric contributes a weighted Bernoulli trial proportional
to its tier weight.

Usage:
    python before_after_comparator.py [--dir DIR] [--output PATH] [--threshold FLOAT]
"""

import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Substance vs style split — user wants "more substance than style, happy medium"
SUBSTANCE_WEIGHT = 0.70
STYLE_WEIGHT = 0.30

# Individual metric weights within each tier (must sum to 1.0 per tier)
SUBSTANCE_METRICS = {
    "semantic_change_density": 0.25,   # how much meaningful content changed
    "information_gain": 0.25,          # net new information in AFTER
    "specificity_delta": 0.20,         # concrete vs abstract language shift
    "actionability_delta": 0.15,       # imperative/directive sentence shift
    "signal_density_delta": 0.15,      # unique tokens / total tokens shift
}

STYLE_METRICS = {
    "readability_delta": 0.40,         # Flesch-Kincaid grade level improvement
    "formatting_quality": 0.30,        # structural markup, headings, lists
    "token_efficiency": 0.30,          # conciseness (fewer words, same info)
}

# Bayesian decision thresholds
RECOMMEND_AFTER = 0.70
RECOMMEND_BEFORE = 0.30

# Stop words for signal density / specificity calculations
STOP_WORDS = frozenset(
    "a an the and or but in on at to for of is it its this that these those "
    "was were be been being have has had do does did will would shall should "
    "may might can could am are not no nor so if then than too very just about "
    "above after again all also any because before between both but by each "
    "few from further get got had has he her here hers herself him himself his "
    "how i into me more most my myself no nor now of off on once only or other "
    "our ours ourselves out over own same she so some such than that the their "
    "theirs them themselves then there these they this those through to under "
    "until up us we were what when where which while who whom why with you your "
    "yours yourself yourselves".split()
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MetricResult:
    name: str
    before_value: float
    after_value: float
    delta: float
    after_wins: bool
    tier: str  # "substance" or "style"
    weight: float  # weight within tier
    contribution: float = 0.0  # weighted contribution to posterior


@dataclass
class ComparisonResult:
    pair_name: str
    before_path: str
    after_path: str
    metrics: List[MetricResult] = field(default_factory=list)
    posterior: float = 0.5
    recommendation: str = "HUMAN REVIEW"
    rationale: str = ""
    substance_score: float = 0.0
    style_score: float = 0.0
    composite_score: float = 0.0


# ---------------------------------------------------------------------------
# Text analysis functions
# ---------------------------------------------------------------------------

def clean_text(raw: str) -> str:
    """Strip HTML/CSS markup, collapse whitespace."""
    text = re.sub(r"<style[^>]*>.*?</style>", "", raw, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\{[^}]*\}", " ", text)  # CSS rules
    text = re.sub(r"&[a-zA-Z]+;", " ", text)  # HTML entities
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def meaningful_tokens(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 2]


def sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"[.!?]+", text) if len(s.strip()) > 10]


def syllable_count(word: str) -> int:
    word = word.lower().rstrip("e")
    count = len(re.findall(r"[aeiouy]+", word))
    return max(1, count)


def flesch_kincaid_grade(text: str) -> float:
    """Compute FK grade level. Lower = more readable."""
    sents = sentences(text)
    words = tokenize(text)
    if not sents or not words:
        return 0.0
    total_syllables = sum(syllable_count(w) for w in words)
    asl = len(words) / len(sents)
    asw = total_syllables / len(words)
    return 0.39 * asl + 11.8 * asw - 15.59


def specificity_ratio(tokens: List[str]) -> float:
    """Ratio of concrete/meaningful tokens to total."""
    if not tokens:
        return 0.0
    meaningful = meaningful_tokens(tokens)
    return len(meaningful) / len(tokens)


def actionability_score(text: str) -> float:
    """Fraction of sentences starting with imperative verbs."""
    sents = sentences(text)
    if not sents:
        return 0.0
    imperative_starters = {
        "ensure", "verify", "implement", "create", "review", "assess",
        "evaluate", "develop", "establish", "define", "document", "identify",
        "conduct", "maintain", "provide", "report", "submit", "complete",
        "use", "apply", "follow", "check", "update", "monitor", "track",
        "approve", "require", "prohibit", "restrict", "allow", "enable",
        "deploy", "configure", "test", "validate", "audit", "comply",
    }
    imperative = 0
    for s in sents:
        first_word = s.split()[0].lower().rstrip("s") if s.split() else ""
        if first_word in imperative_starters:
            imperative += 1
    return imperative / len(sents)


def signal_density(tokens: List[str]) -> float:
    """Unique meaningful tokens / total tokens."""
    if not tokens:
        return 0.0
    meaningful = meaningful_tokens(tokens)
    unique = set(meaningful)
    return len(unique) / len(tokens)


def semantic_change_density(before_tokens: List[str], after_tokens: List[str]) -> float:
    """Fraction of meaningful content that changed between versions.
    Uses set-based Jaccard distance on meaningful token bigrams."""
    def bigrams(toks):
        m = meaningful_tokens(toks)
        return set(zip(m, m[1:])) if len(m) > 1 else set(m)

    b_bi = bigrams(before_tokens)
    a_bi = bigrams(after_tokens)
    if not b_bi and not a_bi:
        return 0.0
    union = b_bi | a_bi
    intersection = b_bi & a_bi
    return 1.0 - (len(intersection) / len(union)) if union else 0.0


def information_gain(before_tokens: List[str], after_tokens: List[str]) -> float:
    """Net new meaningful tokens in AFTER that weren't in BEFORE,
    normalized by AFTER's total meaningful tokens.
    Positive = AFTER has more novel information."""
    b_meaningful = set(meaningful_tokens(before_tokens))
    a_meaningful = set(meaningful_tokens(after_tokens))
    a_total = meaningful_tokens(after_tokens)
    if not a_total:
        return 0.0
    new_in_after = a_meaningful - b_meaningful
    lost_from_before = b_meaningful - a_meaningful
    # Net gain: new tokens minus lost tokens, normalized
    net = len(new_in_after) - len(lost_from_before)
    return net / max(len(a_total), 1)


def formatting_quality(text: str) -> float:
    """Score structural markup: headings, lists, tables, sections.
    Normalized 0-1 by document length."""
    raw_lower = text.lower()
    score = 0.0
    # Count structural elements
    headings = len(re.findall(r"<h[1-6]|^#{1,6}\s", text, re.MULTILINE))
    lists = len(re.findall(r"<[uo]l|^[\-\*]\s|^\d+\.\s", text, re.MULTILINE))
    tables = len(re.findall(r"<table|^\|.*\|", text, re.MULTILINE))
    sections = len(re.findall(r"<section|<article|---+", text, re.MULTILINE))
    # Normalize by word count (expect ~1 structural element per 200 words)
    words = len(tokenize(clean_text(text)))
    expected = max(words / 200, 1)
    total_elements = headings + lists + tables + sections
    return min(total_elements / expected, 1.0)


# ---------------------------------------------------------------------------
# Bayesian comparison engine
# ---------------------------------------------------------------------------

def compute_metrics(before_raw: str, after_raw: str) -> List[MetricResult]:
    """Compute all substance + style metrics for a BEFORE/AFTER pair."""
    before_clean = clean_text(before_raw)
    after_clean = clean_text(after_raw)
    b_tokens = tokenize(before_clean)
    a_tokens = tokenize(after_clean)

    results = []

    # --- Substance metrics ---
    # 1. Semantic change density (higher = more changed, neutral indicator)
    scd = semantic_change_density(b_tokens, a_tokens)
    results.append(MetricResult(
        name="Semantic change density",
        before_value=0.0, after_value=scd, delta=scd,
        after_wins=scd > 0.1,  # meaningful change threshold
        tier="substance",
        weight=SUBSTANCE_METRICS["semantic_change_density"],
    ))

    # 2. Information gain (positive = AFTER has more novel content)
    ig = information_gain(b_tokens, a_tokens)
    results.append(MetricResult(
        name="Information gain",
        before_value=0.0, after_value=ig, delta=ig,
        after_wins=ig > 0.0,
        tier="substance",
        weight=SUBSTANCE_METRICS["information_gain"],
    ))

    # 3. Specificity delta
    b_spec = specificity_ratio(b_tokens)
    a_spec = specificity_ratio(a_tokens)
    results.append(MetricResult(
        name="Specificity",
        before_value=b_spec, after_value=a_spec, delta=a_spec - b_spec,
        after_wins=a_spec > b_spec,
        tier="substance",
        weight=SUBSTANCE_METRICS["specificity_delta"],
    ))

    # 4. Actionability delta
    b_act = actionability_score(before_clean)
    a_act = actionability_score(after_clean)
    results.append(MetricResult(
        name="Actionability",
        before_value=b_act, after_value=a_act, delta=a_act - b_act,
        after_wins=a_act > b_act,
        tier="substance",
        weight=SUBSTANCE_METRICS["actionability_delta"],
    ))

    # 5. Signal density delta
    b_sig = signal_density(b_tokens)
    a_sig = signal_density(a_tokens)
    results.append(MetricResult(
        name="Signal density",
        before_value=b_sig, after_value=a_sig, delta=a_sig - b_sig,
        after_wins=a_sig > b_sig,
        tier="substance",
        weight=SUBSTANCE_METRICS["signal_density_delta"],
    ))

    # --- Style metrics ---
    # 6. Readability (FK grade: lower is better)
    b_fk = flesch_kincaid_grade(before_clean)
    a_fk = flesch_kincaid_grade(after_clean)
    results.append(MetricResult(
        name="Readability (FK grade)",
        before_value=b_fk, after_value=a_fk, delta=b_fk - a_fk,  # positive = AFTER improved
        after_wins=a_fk < b_fk,
        tier="style",
        weight=STYLE_METRICS["readability_delta"],
    ))

    # 7. Formatting quality
    b_fmt = formatting_quality(before_raw)
    a_fmt = formatting_quality(after_raw)
    results.append(MetricResult(
        name="Formatting quality",
        before_value=b_fmt, after_value=a_fmt, delta=a_fmt - b_fmt,
        after_wins=a_fmt > b_fmt,
        tier="style",
        weight=STYLE_METRICS["formatting_quality"],
    ))

    # 8. Token efficiency (fewer words for same/more info = better)
    b_wc = len(b_tokens)
    a_wc = len(a_tokens)
    # Efficiency: 1.0 if AFTER is shorter, scales down if longer
    efficiency = b_wc / a_wc if a_wc > 0 else 1.0
    results.append(MetricResult(
        name="Token efficiency",
        before_value=b_wc, after_value=a_wc,
        delta=b_wc - a_wc,
        after_wins=a_wc <= b_wc,
        tier="style",
        weight=STYLE_METRICS["token_efficiency"],
    ))

    return results


def bayesian_posterior(metrics: List[MetricResult]) -> float:
    """Compute P(AFTER is better | metrics) using substance-weighted beta-binomial.

    Each metric contributes a weighted Bernoulli trial:
      - Substance metrics: weight * SUBSTANCE_WEIGHT
      - Style metrics: weight * STYLE_WEIGHT

    The posterior is Beta(alpha, beta) where:
      alpha = 1 + sum(weighted wins for AFTER)
      beta  = 1 + sum(weighted wins for BEFORE)
    """
    alpha = 1.0  # uniform prior
    beta = 1.0

    for m in metrics:
        tier_weight = SUBSTANCE_WEIGHT if m.tier == "substance" else STYLE_WEIGHT
        contribution = m.weight * tier_weight

        if m.after_wins:
            alpha += contribution
            m.contribution = contribution
        else:
            beta += contribution
            m.contribution = -contribution

    # Posterior mean of Beta distribution
    posterior = alpha / (alpha + beta)
    return posterior


def substance_subscore(metrics: List[MetricResult]) -> float:
    """Weighted average of substance metric wins (0-1)."""
    total_w = sum(m.weight for m in metrics if m.tier == "substance")
    if total_w == 0:
        return 0.5
    wins = sum(m.weight for m in metrics if m.tier == "substance" and m.after_wins)
    return wins / total_w


def style_subscore(metrics: List[MetricResult]) -> float:
    """Weighted average of style metric wins (0-1)."""
    total_w = sum(m.weight for m in metrics if m.tier == "style")
    if total_w == 0:
        return 0.5
    wins = sum(m.weight for m in metrics if m.tier == "style" and m.after_wins)
    return wins / total_w


# ---------------------------------------------------------------------------
# File discovery & pairing
# ---------------------------------------------------------------------------

def find_pairs(directory: str) -> List[Tuple[str, str, str]]:
    """Find BEFORE/AFTER file pairs. Returns [(pair_name, before_path, after_path)]."""
    dir_path = Path(directory)
    befores = {}
    afters = {}

    for f in dir_path.rglob("*"):
        if f.is_dir() or f.suffix.lower() not in {
            ".md", ".txt", ".html", ".htm", ".py", ".json", ".yaml", ".csv",
        }:
            continue
        name_lower = f.stem.lower()
        if "before" in name_lower:
            root = re.sub(r"before[_\-\s]*", "", name_lower, flags=re.IGNORECASE).strip("_()")
            befores.setdefault(root, []).append(str(f))
        elif "after" in name_lower:
            root = re.sub(r"after[_\-\s]*", "", name_lower, flags=re.IGNORECASE).strip("_()")
            afters.setdefault(root, []).append(str(f))

    pairs = []
    matched_afters = set()
    for root, before_files in befores.items():
        if root in afters:
            # Take one pair (prefer .html/.md over others)
            b = sorted(before_files, key=lambda x: (0 if x.endswith(".html") else 1))[0]
            a = sorted(afters[root], key=lambda x: (0 if x.endswith(".html") else 1))[0]
            pair_name = root.replace("_", " ").title() or f.stem
            pairs.append((pair_name, b, a))
            matched_afters.add(root)

    return pairs


def read_file(path: str) -> str:
    """Read file contents, handling common encodings."""
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return Path(path).read_text(encoding=enc)
        except (UnicodeDecodeError, FileNotFoundError):
            continue
    return ""


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(results: List[ComparisonResult], output_path: str) -> None:
    """Write markdown report to disk."""
    lines = [
        "# BEFORE/AFTER Substance-Weighted Comparison",
        "",
        f"**Model:** Bayesian beta-binomial, substance {int(SUBSTANCE_WEIGHT*100)}% / "
        f"style {int(STYLE_WEIGHT*100)}%",
        f"**Thresholds:** AFTER ≥ {RECOMMEND_AFTER}, BEFORE ≤ {RECOMMEND_BEFORE}, "
        f"else HUMAN REVIEW",
        "",
        "---",
        "",
    ]

    for r in results:
        lines.append(f"## {r.pair_name}")
        lines.append("")
        lines.append(f"**BEFORE:** `{r.before_path}`")
        lines.append(f"**AFTER:** `{r.after_path}`")
        lines.append("")

        # Substance metrics table
        lines.append("### Substance Metrics (70% weight)")
        lines.append("")
        lines.append("| Metric | BEFORE | AFTER | Delta | Winner | Weight |")
        lines.append("|--------|--------|-------|-------|--------|--------|")
        for m in r.metrics:
            if m.tier != "substance":
                continue
            winner = "AFTER" if m.after_wins else "BEFORE"
            lines.append(
                f"| {m.name} | {m.before_value:.3f} | {m.after_value:.3f} | "
                f"{m.delta:+.3f} | {winner} | {m.weight:.0%} |"
            )
        lines.append("")

        # Style metrics table
        lines.append("### Style Metrics (30% weight)")
        lines.append("")
        lines.append("| Metric | BEFORE | AFTER | Delta | Winner | Weight |")
        lines.append("|--------|--------|-------|-------|--------|--------|")
        for m in r.metrics:
            if m.tier != "style":
                continue
            winner = "AFTER" if m.after_wins else "BEFORE"
            b_val = f"{m.before_value:.3f}" if m.before_value < 100 else f"{m.before_value:.0f}"
            a_val = f"{m.after_value:.3f}" if m.after_value < 100 else f"{m.after_value:.0f}"
            lines.append(
                f"| {m.name} | {b_val} | {a_val} | "
                f"{m.delta:+.3f} | {winner} | {m.weight:.0%} |"
            )
        lines.append("")

        # Scores
        lines.append("### Scores")
        lines.append("")
        lines.append(f"- **Substance sub-score:** {r.substance_score:.2f} (AFTER wins "
                      f"{r.substance_score:.0%} of substance weight)")
        lines.append(f"- **Style sub-score:** {r.style_score:.2f} (AFTER wins "
                      f"{r.style_score:.0%} of style weight)")
        lines.append(f"- **Composite:** {r.composite_score:.3f}")
        lines.append(f"- **Bayesian posterior P(AFTER better):** {r.posterior:.3f}")
        lines.append("")
        lines.append(f"**Recommendation:** {r.recommendation}")
        lines.append(f"**Rationale:** {r.rationale}")
        lines.append("")
        lines.append("---")
        lines.append("")

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Pair | Substance | Style | P(AFTER) | Recommendation |")
    lines.append("|------|-----------|-------|----------|----------------|")
    for r in results:
        lines.append(
            f"| {r.pair_name} | {r.substance_score:.2f} | {r.style_score:.2f} | "
            f"{r.posterior:.3f} | {r.recommendation} |"
        )
    lines.append("")
    lines.append(f"*Substance weight: {SUBSTANCE_WEIGHT:.0%} | "
                 f"Style weight: {STYLE_WEIGHT:.0%}*")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def compare_pair(pair_name: str, before_path: str, after_path: str) -> ComparisonResult:
    before_raw = read_file(before_path)
    after_raw = read_file(after_path)

    if not before_raw or not after_raw:
        return ComparisonResult(
            pair_name=pair_name,
            before_path=before_path,
            after_path=after_path,
            recommendation="ERROR",
            rationale="Could not read one or both files.",
        )

    metrics = compute_metrics(before_raw, after_raw)
    posterior = bayesian_posterior(metrics)
    sub_score = substance_subscore(metrics)
    sty_score = style_subscore(metrics)
    composite = SUBSTANCE_WEIGHT * sub_score + STYLE_WEIGHT * sty_score

    if posterior >= RECOMMEND_AFTER:
        rec = "AFTER"
    elif posterior <= RECOMMEND_BEFORE:
        rec = "BEFORE"
    else:
        rec = "HUMAN REVIEW"

    # Build rationale
    sub_wins = [m.name for m in metrics if m.tier == "substance" and m.after_wins]
    sub_losses = [m.name for m in metrics if m.tier == "substance" and not m.after_wins]
    if rec == "AFTER":
        rationale = (f"AFTER wins on substance ({', '.join(sub_wins) or 'none'}) "
                     f"with P={posterior:.2f}.")
    elif rec == "BEFORE":
        rationale = (f"BEFORE wins on substance ({', '.join(sub_losses) or 'none'}) "
                     f"with P={posterior:.2f}.")
    else:
        rationale = (f"Mixed results — substance wins: {', '.join(sub_wins) or 'none'}; "
                     f"substance losses: {', '.join(sub_losses) or 'none'}. "
                     f"P={posterior:.2f} falls in review zone.")

    return ComparisonResult(
        pair_name=pair_name,
        before_path=before_path,
        after_path=after_path,
        metrics=metrics,
        posterior=posterior,
        recommendation=rec,
        rationale=rationale,
        substance_score=sub_score,
        style_score=sty_score,
        composite_score=composite,
    )


def main():
    parser = argparse.ArgumentParser(description="BEFORE/AFTER document comparator")
    _data_dir = os.environ.get("CIVIC_AI_DATA_DIR", str(Path(__file__).parent.parent / "data"))
    parser.add_argument("--dir", default=_data_dir,
                        help="Directory to scan for BEFORE/AFTER pairs")
    parser.add_argument("--output", default=str(Path(_data_dir) / "before_after_substance_analysis.md"),
                        help="Output report path")
    parser.add_argument("--threshold-after", type=float, default=RECOMMEND_AFTER,
                        help="P threshold to recommend AFTER")
    parser.add_argument("--threshold-before", type=float, default=RECOMMEND_BEFORE,
                        help="P threshold to recommend BEFORE")
    parser.add_argument("--substance-weight", type=float, default=SUBSTANCE_WEIGHT,
                        help="Weight for substance tier (0-1)")
    parser.add_argument("--json", action="store_true", help="Also emit JSON results")
    args = parser.parse_args()

    global SUBSTANCE_WEIGHT, STYLE_WEIGHT, RECOMMEND_AFTER, RECOMMEND_BEFORE
    SUBSTANCE_WEIGHT = args.substance_weight
    STYLE_WEIGHT = 1.0 - SUBSTANCE_WEIGHT
    RECOMMEND_AFTER = args.threshold_after
    RECOMMEND_BEFORE = args.threshold_before

    pairs = find_pairs(args.dir)
    if not pairs:
        print(f"No BEFORE/AFTER pairs found in {args.dir}")
        sys.exit(1)

    print(f"Found {len(pairs)} pair(s). Substance weight: {SUBSTANCE_WEIGHT:.0%}")

    results = [compare_pair(name, b, a) for name, b, a in pairs]
    generate_report(results, args.output)

    if args.json:
        json_path = args.output.replace(".md", ".json")
        json_results = []
        for r in results:
            json_results.append({
                "pair": r.pair_name,
                "posterior": r.posterior,
                "substance_score": r.substance_score,
                "style_score": r.style_score,
                "composite": r.composite_score,
                "recommendation": r.recommendation,
                "metrics": [
                    {
                        "name": m.name,
                        "tier": m.tier,
                        "before": m.before_value,
                        "after": m.after_value,
                        "delta": m.delta,
                        "after_wins": m.after_wins,
                        "weight": m.weight,
                    }
                    for m in r.metrics
                ],
            })
        Path(json_path).write_text(json.dumps(json_results, indent=2))
        print(f"JSON written to {json_path}")

    # Print summary
    print("\n--- Summary ---")
    for r in results:
        print(f"  {r.pair_name}: P(AFTER)={r.posterior:.3f} → {r.recommendation}")


if __name__ == "__main__":
    main()
