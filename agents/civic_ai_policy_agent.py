#!/usr/bin/env python3
"""
Civic AI Policy Agent - Government AI implementation guidance
Provides policy guidance, implementation frameworks, and best practices
for county/municipal AI adoption
"""

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from loguru import logger


@dataclass
class PolicyFramework:
    """AI policy framework structure"""

    phase: str
    duration: str
    activities: List[str]
    deliverables: List[str]
    stakeholders: List[str]


@dataclass
class KBSection:
    """A searchable section from the knowledge base"""

    title: str
    category: str
    topics: List[str]
    content: str
    source_file: str
    word_count: int = 0


class CivicAIPolicyAgent:
    """
    Comprehensive civic AI policy guidance:
    - Government AI implementation frameworks
    - NIST AI Risk Management Framework guidance
    - County/municipal policy best practices
    - Real-world case study analysis (Miami-Dade, San Diego, Georgia, NJ, MA, etc.)
    - Tech lead implementation roadmaps
    - Compliance and security requirements
    - 13+ government AI policy documents (AI Bill of Rights, NIST RMF, state guidelines, etc.)
    """

    # Knowledge base paths
    KNOWLEDGE_BASE = Path(__file__).parent.parent / "knowledge_base" / "COUNTY_AI_POLICY_RESEARCH.md"
    DOCUMENTS_KB = Path(__file__).parent.parent / "knowledge_base" / "GOVERNMENT_AI_DOCUMENTS_KB.md"

    # Implementation phases
    IMPLEMENTATION_PHASES = {
        "phase_1": {
            "name": "Discovery & Assessment",
            "duration": "2-3 months",
            "key_activities": [
                "Survey all departments on current AI usage",
                "Identify shadow IT and unauthorized tools",
                "Document existing AI projects",
                "Assess data governance maturity",
                "Apply NIST AI Risk Management Framework",
            ],
            "deliverables": [
                "AI Usage Inventory Report",
                "Shadow IT Assessment",
                "Data Governance Gap Analysis",
                "AI Risk Register",
            ],
        },
        "phase_2": {
            "name": "Policy & Governance",
            "duration": "1-2 months",
            "key_activities": [
                "Draft AI usage policy",
                "Create AI procurement guidelines",
                "Establish AI Governance Board",
                "Develop security guidelines",
                "Define data handling requirements",
            ],
            "deliverables": [
                "AI Usage Policy",
                "Governance Charter",
                "Security Guidelines",
                "Vendor Assessment Checklist",
            ],
        },
        "phase_3": {
            "name": "Pilot Programs",
            "duration": "3-6 months",
            "key_activities": [
                "Select 2-3 pilot projects",
                "Procure and configure AI tools",
                "Train pilot users",
                "Monitor usage and performance",
                "Collect feedback and iterate",
            ],
            "deliverables": [
                "Pilot Evaluation Report",
                "Lessons Learned Document",
                "Policy Updates",
                "Expansion Plan",
            ],
        },
        "phase_4": {
            "name": "Training & Change Management",
            "duration": "Ongoing",
            "key_activities": [
                "Develop training curriculum",
                "Conduct employee training sessions",
                "Establish Communities of Practice",
                "Partner with local universities",
                "Address concerns and resistance",
            ],
            "deliverables": [
                "Training Materials",
                "Change Management Plan",
                "Partnership Agreements",
                "Employee Feedback Reports",
            ],
        },
        "phase_5": {
            "name": "Scale & Production",
            "duration": "6-12 months",
            "key_activities": [
                "Phased rollout by department",
                "Establish AI operations team",
                "Implement monitoring and alerting",
                "Conduct regular audits",
                "Report to leadership",
            ],
            "deliverables": [
                "Production Deployment Plan",
                "Operations Playbook",
                "Quarterly Board Reports",
                "Public Transparency Portal",
            ],
        },
    }

    # Key resources and citations
    PRIMARY_RESOURCES = [
        {
            "name": "NACo AI County Compass",
            "url": (
                "https://www.naco.org/resource/"
                "ai-county-compass-comprehensive-toolkit-"
                "local-governance-and-implementation-artificial"
            ),
            "description": "Most comprehensive county-specific AI toolkit available",
        },
        {
            "name": "National Academies Report: AI Integration Strategies",
            "url": "https://www.nationalacademies.org/read/29152/chapter/3",
            "description": (
                "Strategies for integrating AI into state and " "local government decision making"
            ),
        },
        {
            "name": "Miami-Dade County AI Reports",
            "urls": [
                (
                    "https://documents.miamidade.gov/mayor/memos/"
                    "03.22.24-Report-on-Miami-Dade-Countys-Policy-"
                    "on-Artificial-Intelligence-Directive-"
                    "No-231203.pdf"
                ),
                (
                    "https://www.miamidade.gov/technology/library/"
                    "artificial-intelligence-report-2025.pdf"
                ),
            ],
            "description": ("Real-world implementation case study " "from pilot to production"),
        },
        {
            "name": "NIST AI Risk Management Framework",
            "url": "https://www.ai.gov/",
            "description": "Federal risk management framework for AI systems",
        },
    ]

    # 2026 Implementation priorities
    PRIORITIES_2026 = [
        "Production deployment of Generative AI (beyond pilots)",
        "Agentic AI systems (AI agents that take actions)",
        "AI-powered citizen services (chatbots, virtual agents)",
        "Internal productivity tools (document summarization, meeting transcription)",
        "Operational efficiency (automated form processing, data entry)",
    ]

    # Critical success factors
    SUCCESS_FACTORS = [
        "Executive Support & Governance - Board/Commission buy-in and oversight",
        "Policy Before Technology - Clear guidelines before deployment",
        "Start Small, Scale Smart - Pilot in limited scope first",
        "Invest in People - Comprehensive training and change management",
        "Measure Everything - Usage metrics, ROI, user satisfaction",
        "Partner Extensively - Universities, peer counties, vendors",
        "Communicate Transparently - Regular updates and public transparency",
    ]

    # Common pitfalls
    COMMON_PITFALLS = {
        "technology_first": {
            "problem": "Deploying AI tools without clear use cases or policies",
            "solution": "Start with community needs and policy framework",
        },
        "inadequate_training": {
            "problem": "Rolling out tools without proper user preparation",
            "solution": "Invest heavily in training and change management",
        },
        "shadow_it": {
            "problem": "Employees using unauthorized AI tools",
            "solution": "Establish clear policies and easy-to-use approved tools",
        },
        "vendor_lockin": {
            "problem": "Over-dependence on single vendor",
            "solution": "Evaluate data portability and open standards",
        },
        "privacy_gaps": {
            "problem": "Inadequate data protection measures",
            "solution": "Apply NIST framework and conduct thorough risk assessments",
        },
        "no_metrics": {
            "problem": "Unable to demonstrate value or identify issues",
            "solution": "Define KPIs upfront and monitor continuously",
        },
        "poor_governance": {
            "problem": "No clear decision-making authority or oversight",
            "solution": "Establish AI governance board with executive representation",
        },
    }

    def __init__(self):
        logger.info("Civic AI Policy Agent initialized")
        self.kb_sections: List[KBSection] = []
        self._load_knowledge_base()
        self._load_documents_kb()

    def _load_knowledge_base(self):
        """Load the civic AI policy research document"""
        try:
            if self.KNOWLEDGE_BASE.exists():
                self.knowledge_content = self.KNOWLEDGE_BASE.read_text()
                logger.info(
                    f"Loaded knowledge base: {self.KNOWLEDGE_BASE} "
                    f"({len(self.knowledge_content):,} chars)"
                )
            else:
                logger.warning(f"Knowledge base not found: {self.KNOWLEDGE_BASE}")
                self.knowledge_content = None
        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")
            self.knowledge_content = None

    def _load_documents_kb(self):
        """Load and index the government AI documents knowledge base"""
        try:
            if not self.DOCUMENTS_KB.exists():
                logger.warning(f"Documents KB not found: {self.DOCUMENTS_KB}")
                return

            content = self.DOCUMENTS_KB.read_text()
            # Parse sections by ## headers (each document is a ## section)
            # Use ^ anchor with MULTILINE to find section boundaries at line starts
            section_pattern = re.compile(
                r"^## \d+\.\s+(.+?)\n\n"
                r"\*\*Category:\*\*\s*(.+?)\n"
                r"\*\*Topics:\*\*\s*(.+?)\n"
                r"\*\*Word Count:\*\*\s*[\d,]+\n\n"
                r"### Content\n\n"
                r"(.*?)(?=^## \d+\.|\Z)",
                re.MULTILINE | re.DOTALL,
            )

            for match in section_pattern.finditer(content):
                title = match.group(1).strip()
                category = match.group(2).strip()
                topics = [t.strip() for t in match.group(3).split(",")]
                section_content = match.group(4).strip()

                section = KBSection(
                    title=title,
                    category=category,
                    topics=topics,
                    content=section_content,
                    source_file="GOVERNMENT_AI_DOCUMENTS_KB.md",
                    word_count=len(section_content.split()),
                )
                self.kb_sections.append(section)

            logger.info(f"Loaded {len(self.kb_sections)} document sections from documents KB")
        except Exception as e:
            logger.error(f"Error loading documents KB: {e}")

    async def get_implementation_framework(
        self, county_size: str = "medium", current_maturity: str = "planning"
    ) -> Dict:
        """
        Get county AI implementation framework

        Args:
            county_size: Small, medium, or large county
            current_maturity: planning, pilot, or production phase

        Returns:
            Complete implementation framework with phases
        """

        framework: Dict[str, Any] = {
            "overview": "5-Phase Implementation Framework for County AI Policy",
            "total_timeline": "12-18 months from planning to production",
            "phases": [],
        }

        # Add all phases
        for phase_key, phase_data in self.IMPLEMENTATION_PHASES.items():
            framework["phases"].append(
                {  # type: ignore
                    "phase": phase_data["name"],
                    "duration": phase_data["duration"],
                    "activities": phase_data["key_activities"],
                    "deliverables": phase_data["deliverables"],
                    "status": (
                        "current"
                        if self._is_current_phase(
                            phase_data["name"],  # type: ignore[arg-type]
                            current_maturity,
                        )
                        else "future"
                    ),
                }
            )

        framework["current_phase"] = self._get_phase_by_maturity(current_maturity)
        framework["success_factors"] = self.SUCCESS_FACTORS
        framework["priorities_2026"] = self.PRIORITIES_2026

        return framework

    async def analyze_case_study(self, county: str = "Miami-Dade") -> Dict:
        """
        Get detailed case study analysis

        Args:
            county: County name (Miami-Dade, San Diego, etc.)

        Returns:
            Case study details with timeline and lessons learned
        """

        case_studies = {
            "Miami-Dade": {
                "timeline": {
                    "July 2023": "Board of County Commissioners adopted AI directive",
                    "2024": "Assessment phase - surveys, governance, security guidelines",
                    "2025": "Production deployment - Microsoft 365 Copilot, workforce training",
                },
                "key_achievements": [
                    "Established Community of AI Practice (cross-departmental)",
                    "Created data governance model with documented inventories",
                    "Rolled out security guidelines and usage metrics",
                    "Launched Microsoft 365 Copilot for all employees",
                    "Partnered with InnovateUS for workforce development",
                ],
                "success_factors": [
                    "Executive support from County Mayor",
                    "Board oversight and accountability",
                    "Iterative approach with quarterly reporting",
                    "Investment in employee training",
                    "Partnership with vendors and academia",
                ],
                "lessons_learned": [
                    "Start with organizational assessment to identify shadow IT",
                    "Establish governance BEFORE deployment",
                    "Training and change management are critical",
                    "Regular reporting builds trust and accountability",
                ],
            },
            "San Diego": {
                "timeline": {"2025": "Policy examination and planning phase"},
                "status": "Assessing current AI usage and planning governance structure",
                "key_activities": [
                    "Department-by-department AI usage assessment",
                    "Identifying needed policy changes",
                    "Planning governance structure",
                ],
            },
            "Georgia": {
                "timeline": {
                    "2024": "Developed AI Responsible Use Guidelines",
                    "2025": "Measurable outcomes across multiple state agencies",
                },
                "key_achievements": [
                    "Published comprehensive Responsible AI Use Guidelines",
                    "Established risk-based classification for AI systems",
                    "Defined acceptable use categories with clear guardrails",
                    "Created state-wide AI governance framework",
                ],
                "success_factors": [
                    "Risk-based approach to AI classification",
                    "Clear guidelines for different risk levels",
                    "State-level coordination across agencies",
                ],
            },
            "New Jersey": {
                "timeline": {"2024-2025": "NJ AI Agent development and deployment"},
                "key_achievements": [
                    "Launched NJ AI Agent for citizen services",
                    "Integrated AI into state service delivery",
                    "Developed state-specific AI implementation framework",
                ],
                "focus_areas": [
                    "Citizen-facing AI services",
                    "State government operations",
                    "Public sector AI innovation",
                ],
            },
            "Massachusetts": {
                "timeline": {"2024-2025": "Massachusetts Genie AI assistant development"},
                "key_achievements": [
                    "Developed Massachusetts Genie AI assistant",
                    "AI-powered government service delivery",
                    "Innovation in public sector AI applications",
                ],
                "focus_areas": [
                    "AI assistants for government operations",
                    "Citizen engagement through AI",
                    "State-level AI innovation",
                ],
            },
            "Boston": {
                "timeline": {"2024-2025": "Boston AI Launchpad initiative"},
                "key_achievements": [
                    "Launched Boston AI Launchpad program",
                    "City-level AI innovation initiative",
                    "Public-private partnership for AI in government",
                ],
            },
            "Manatee County": {
                "timeline": {"2025": "AI Readiness Assessment completed"},
                "status": "Assessment phase with structured readiness evaluation",
                "key_achievements": [
                    "Completed comprehensive AI readiness assessment",
                    "Evaluated department-level AI maturity",
                    "Identified priority use cases for AI adoption",
                ],
                "focus_areas": [
                    "Organizational readiness evaluation",
                    "Department-by-department assessment",
                    "Prioritized AI implementation plan",
                ],
            },
            "Maryland": {
                "timeline": {
                    "2024": "DoIT AI Strategy development",
                    "2025": "Implementation and InnovateUS partnership",
                },
                "key_achievements": [
                    "Published Department of IT AI Strategy",
                    "Partnered with InnovateUS for workforce development",
                    "Created comprehensive AI training programs",
                    "Established state-level AI governance",
                ],
                "focus_areas": [
                    "State IT modernization",
                    "Workforce AI training",
                    "AI governance frameworks",
                ],
            },
            "San Jose": {
                "timeline": {
                    "2023-2024": "GovAI Coalition formation and GenAI guidelines",
                    "2025": "Templates and resources published",
                },
                "key_achievements": [
                    "Co-founded the GovAI Coalition",
                    "Published comprehensive GenAI guidelines for city use",
                    "Created Algorithmic Impact Assessment template",
                    "Developed reusable governance templates for other cities",
                ],
                "success_factors": [
                    "Coalition-based approach sharing resources across cities",
                    "Open-source policy templates",
                    "Comprehensive generative AI usage guidelines",
                ],
            },
        }

        if county in case_studies:
            result = {
                "county": county,
                "case_study": case_studies[county],
                "sources": [
                    r
                    for r in self.PRIMARY_RESOURCES
                    if county in r.get("name", "")  # type: ignore[attr-defined]
                ],
            }
            # Enrich with document KB content if available
            doc = await self.get_document_by_topic(county)
            if doc.get("found"):
                result["additional_context"] = {
                    "document": doc["title"],
                    "excerpt": doc["content"][:1000],
                }
            return result
        else:
            # Try dynamic lookup from document KB
            doc = await self.get_document_by_topic(county)
            if doc.get("found"):
                return {
                    "county": county,
                    "case_study": {
                        "source_document": doc["title"],
                        "category": doc["category"],
                        "content": doc["content"][:2000],
                        "note": "Derived from indexed government AI document",
                    },
                }
            return {
                "error": f"Case study for {county} not available",
                "available_case_studies": list(case_studies.keys()),
            }

    async def get_nist_framework_guidance(self) -> Dict:
        """
        Get NIST AI Risk Management Framework guidance

        Returns:
            NIST framework overview and implementation guidance
        """

        return {
            "framework": "NIST AI Risk Management Framework",
            "purpose": (
                "Comprehensive framework for identifying, " "assessing, and mitigating AI risks"
            ),
            "url": "https://www.ai.gov/",
            "key_components": [
                "Govern - Establish governance and accountability structures",
                "Map - Identify and document AI risks and contexts",
                "Measure - Assess and analyze identified risks",
                "Manage - Implement and monitor risk mitigation strategies",
            ],
            "county_application": [
                "Use framework during Phase 1 (Discovery & Assessment)",
                "Create AI Risk Register documenting all identified risks",
                "Apply to all AI procurement and deployment decisions",
                "Integrate into vendor security assessment criteria",
                "Reference in AI usage policies and guidelines",
            ],
            "best_practices": [
                "Start with high-risk use cases (e.g., citizen-facing decisions)",
                "Document risk assessments for transparency",
                "Revisit risk assessments quarterly",
                "Train governance board on NIST framework",
                "Use framework to communicate with elected officials",
            ],
        }

    async def get_policy_checklist(self, focus_area: str = "all") -> Dict:
        """
        Get policy implementation checklist

        Args:
            focus_area: all, governance, security, training, or deployment

        Returns:
            Actionable checklist for policy implementation
        """

        checklists = {
            "governance": {
                "name": "Governance & Policy Setup",
                "items": [
                    "☐ Review NACo AI County Compass toolkit",
                    "☐ Read National Academies report on AI integration",
                    "☐ Survey current AI usage across all departments",
                    "☐ Identify executive sponsor and get board buy-in",
                    "☐ Assemble AI governance board/committee",
                    "☐ Draft AI usage policy (reference peer county policies)",
                    "☐ Create AI procurement guidelines",
                    "☐ Establish governance charter with RACI matrix",
                    "☐ Set meeting cadence for governance board",
                ],
            },
            "security": {
                "name": "Security & Compliance",
                "items": [
                    "☐ Apply NIST AI Risk Management Framework",
                    "☐ Create AI Risk Register",
                    "☐ Develop AI-specific security guidelines",
                    "☐ Create vendor security assessment checklist",
                    "☐ Establish data privacy requirements",
                    "☐ Define incident response protocols",
                    "☐ Set up audit and monitoring procedures",
                    "☐ Review compliance requirements (FISMA, HIPAA, etc.)",
                    "☐ Document data handling and classification policies",
                ],
            },
            "training": {
                "name": "Training & Change Management",
                "items": [
                    "☐ Develop training curriculum for all employees",
                    "☐ Create role-specific training (IT, leadership, power users)",
                    "☐ Partner with local universities for expertise",
                    "☐ Establish Community of AI Practice",
                    "☐ Schedule regular training sessions and workshops",
                    "☐ Create documentation and job aids",
                    "☐ Set up office hours/drop-in support",
                    "☐ Develop change management communication plan",
                    "☐ Identify and train department champions",
                ],
            },
            "deployment": {
                "name": "Pilot & Production Deployment",
                "items": [
                    "☐ Identify 2-3 pilot projects (low-risk, high-value)",
                    "☐ Define success metrics for pilots",
                    "☐ Procure and configure pilot AI tools",
                    "☐ Train pilot users",
                    "☐ Monitor usage and collect feedback",
                    "☐ Evaluate pilot results against success criteria",
                    "☐ Refine policies based on pilot learnings",
                    "☐ Create expansion plan",
                    "☐ Establish AI operations (AIOps) team",
                    "☐ Set up production monitoring and alerting",
                    "☐ Create quarterly reporting process",
                ],
            },
        }

        if focus_area == "all":
            return {
                "overview": "Complete AI Policy Implementation Checklist",
                "checklists": checklists,
                "timeline": "Use these checklists across Phases 1-5 of implementation",
            }
        elif focus_area in checklists:
            return {"focus_area": focus_area, "checklist": checklists[focus_area]}
        else:
            return {
                "error": f"Focus area '{focus_area}' not recognized",
                "available_areas": list(checklists.keys()) + ["all"],
            }

    async def get_common_pitfalls(self) -> Dict:
        """
        Get common pitfalls and solutions

        Returns:
            List of common pitfalls with solutions
        """

        return {
            "overview": "Common pitfalls in county AI implementation and how to avoid them",
            "pitfalls": self.COMMON_PITFALLS,
            "key_recommendation": (
                "Policy Before Technology - establish governance "
                "and guidelines before deploying tools"
            ),
        }

    async def get_2026_priorities(self) -> Dict:
        """
        Get 2026 AI implementation priorities for counties

        Returns:
            Current priorities and trends
        """

        return {
            "year": 2026,
            "overview": "Counties are moving from pilots to production deployment",
            "top_priorities": self.PRIORITIES_2026,
            "key_trends": [
                "90%+ of states have adopted responsible-use AI policies",
                "Focus shifted to risk management and transparency",
                "Enterprise-scale implementations (e.g., Microsoft 365 Copilot)",
                "Investment in workforce training and partnerships",
                "Public transparency portals and regular reporting",
            ],
            "success_stories": [
                "Miami-Dade County: Full production deployment with Microsoft 365 Copilot",
                "Georgia: Measurable outcomes across multiple state agencies",
                "North Dakota: Quick wins approach showing rapid value",
            ],
            "recommended_focus": (
                "Start with internal productivity tools (low-risk) "
                "to build confidence and expertise"
            ),
        }

    async def search_knowledge_base(self, query: str, max_results: int = 5) -> Dict:
        """
        Search across all knowledge base documents using keyword + topic matching.

        Args:
            query: Search query
            max_results: Maximum number of results to return

        Returns:
            Relevant sections from knowledge base with context
        """
        results = []
        query_lower = query.lower()
        keywords = [k for k in query_lower.split() if len(k) > 2]

        # 1. Search indexed document sections (from the 13 PDF documents)
        for section in self.kb_sections:
            score = 0
            # Topic match (high weight)
            for topic in section.topics:
                if any(kw in topic.lower() for kw in keywords):
                    score += 3
            # Title match (high weight)
            if any(kw in section.title.lower() for kw in keywords):
                score += 3
            # Category match (medium weight)
            if any(kw in section.category.lower() for kw in keywords):
                score += 2
            # Content match (count keyword hits)
            content_lower = section.content.lower()
            for kw in keywords:
                score += content_lower.count(kw)

            if score > 0:
                # Extract the most relevant snippet (paragraph containing most keywords)
                paragraphs = section.content.split("\n\n")
                best_para = ""
                best_para_score = 0
                for para in paragraphs:
                    para_lower = para.lower()
                    para_score = sum(para_lower.count(kw) for kw in keywords)
                    if para_score > best_para_score:
                        best_para_score = para_score
                        best_para = para

                results.append(
                    {
                        "source": section.title,
                        "category": section.category,
                        "topics": section.topics,
                        "relevance_score": score,
                        "snippet": best_para[:500] if best_para else section.content[:500],
                        "word_count": section.word_count,
                    }
                )

        # 2. Also search the original research KB (line-based for backwards compat)
        if self.knowledge_content:
            lines = self.knowledge_content.split("\n")
            for i, line in enumerate(lines):
                if any(keyword in line.lower() for keyword in keywords):
                    start = max(0, i - 3)
                    end = min(len(lines), i + 4)
                    context = "\n".join(lines[start:end])
                    results.append(
                        {
                            "source": "County AI Policy Research",
                            "category": "Original Research",
                            "topics": [],
                            "relevance_score": 1,
                            "snippet": context[:500],
                            "word_count": len(context.split()),
                        }
                    )

        # Sort by relevance score, deduplicate, limit
        results.sort(
            key=lambda x: x["relevance_score"], reverse=True  # type: ignore[arg-type, return-value]
        )
        # Deduplicate by source
        seen_sources = set()
        unique_results = []
        for r in results:
            key = r["source"]
            if key not in seen_sources:
                seen_sources.add(key)
                unique_results.append(r)

        return {
            "query": query,
            "matches_found": len(unique_results),
            "results": unique_results[:max_results],
            "total_documents_indexed": len(self.kb_sections),
            "knowledge_bases": ["COUNTY_AI_POLICY_RESEARCH.md", "GOVERNMENT_AI_DOCUMENTS_KB.md"],
        }

    async def get_document_by_topic(self, topic: str) -> Dict:
        """
        Retrieve a full document section by topic or title match.

        Args:
            topic: Topic, state name, or document title to look up

        Returns:
            Full document content with metadata
        """
        topic_lower = topic.lower()

        for section in self.kb_sections:
            # Match on title, topics, or category
            if (
                topic_lower in section.title.lower()
                or any(topic_lower in t.lower() for t in section.topics)
                or topic_lower in section.category.lower()
            ):
                return {
                    "found": True,
                    "title": section.title,
                    "category": section.category,
                    "topics": section.topics,
                    "content": section.content,
                    "word_count": section.word_count,
                }

        return {
            "found": False,
            "query": topic,
            "available_documents": [s.title for s in self.kb_sections],
            "suggestion": "Try searching with a state name, framework name, or topic keyword",
        }

    async def list_documents(self) -> Dict:
        """
        List all indexed government AI documents.

        Returns:
            Summary of all documents in the knowledge base
        """
        docs = []
        for section in self.kb_sections:
            docs.append(
                {
                    "title": section.title,
                    "category": section.category,
                    "topics": section.topics,
                    "word_count": section.word_count,
                }
            )

        total_words = sum(s.word_count for s in self.kb_sections)
        return {
            "total_documents": len(docs),
            "total_words": total_words,
            "documents": docs,
            "categories": list(set(s.category for s in self.kb_sections)),
        }

    def _is_current_phase(self, phase_name: str, maturity: str) -> bool:
        """Determine if phase matches current maturity level"""
        maturity_map = {
            "planning": "Discovery & Assessment",
            "pilot": "Pilot Programs",
            "production": "Scale & Production",
        }
        return phase_name == maturity_map.get(maturity, "Discovery & Assessment")

    def _get_phase_by_maturity(self, maturity: str) -> str:
        """Get current phase name by maturity level"""
        maturity_map = {
            "planning": "Phase 1: Discovery & Assessment",
            "pilot": "Phase 3: Pilot Programs",
            "production": "Phase 5: Scale & Production",
        }
        return maturity_map.get(maturity, "Phase 1: Discovery & Assessment")


# Example usage
async def main():
    """Test the civic AI policy agent"""
    agent = CivicAIPolicyAgent()

    # Get implementation framework
    print("=== Implementation Framework ===\n")
    framework = await agent.get_implementation_framework(
        county_size="medium", current_maturity="planning"
    )
    print(f"Framework: {framework['overview']}")
    print(f"Timeline: {framework['total_timeline']}")
    print(f"Current Phase: {framework['current_phase']}\n")

    # Analyze case study
    print("\n=== Miami-Dade Case Study ===\n")
    case_study = await agent.analyze_case_study("Miami-Dade")
    print(f"County: {case_study['county']}")
    print("\nKey Achievements:")
    for achievement in case_study["case_study"]["key_achievements"]:
        print(f"  ✓ {achievement}")

    # Get NIST guidance
    print("\n=== NIST Framework Guidance ===\n")
    nist = await agent.get_nist_framework_guidance()
    print(f"Framework: {nist['framework']}")
    print("\nKey Components:")
    for component in nist["key_components"]:
        print(f"  • {component}")

    # Get policy checklist
    print("\n=== Policy Checklist (Governance) ===\n")
    checklist = await agent.get_policy_checklist(focus_area="governance")
    print(f"{checklist['checklist']['name']}:")
    for item in checklist["checklist"]["items"]:
        print(f"  {item}")

    # Get 2026 priorities
    print("\n=== 2026 Implementation Priorities ===\n")
    priorities = await agent.get_2026_priorities()
    print(f"Top Priorities for {priorities['year']}:")
    for i, priority in enumerate(priorities["top_priorities"], 1):
        print(f"  {i}. {priority}")


if __name__ == "__main__":
    asyncio.run(main())
