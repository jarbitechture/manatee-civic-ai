"""
Web Intelligence Agent — Legislation and peer county monitoring
Tracks state legislation, peer county AI initiatives, and federal framework updates.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

from agents.base_agent import AgentContext, AgentResult, BaseAgent


@dataclass
class LegislationAlert:
    """A tracked legislative item"""

    title: str
    jurisdiction: str  # "florida", "federal", "peer_county"
    url: str
    summary: str
    relevance_score: float = 0.0
    discovered_at: str = field(default_factory=lambda: datetime.now().isoformat())


class WebIntelligenceAgent(BaseAgent):
    """
    Monitors legislation, peer county AI programs, and federal frameworks.

    Capabilities:
    - Track Florida legislature AI-related bills
    - Monitor peer county AI initiatives (Miami-Dade, San Diego, etc.)
    - Watch NIST, GovAI Coalition, and NACo updates
    - Generate briefing summaries for leadership
    """

    # Peer counties to monitor
    PEER_COUNTIES = [
        {"name": "Miami-Dade County", "state": "FL", "url": "https://www.miamidade.gov"},
        {"name": "Hillsborough County", "state": "FL", "url": "https://www.hillsboroughcounty.org"},
        {"name": "Orange County", "state": "FL", "url": "https://www.orangecountyfl.net"},
        {"name": "San Diego County", "state": "CA", "url": "https://www.sandiegocounty.gov"},
        {"name": "Fairfax County", "state": "VA", "url": "https://www.fairfaxcounty.gov"},
    ]

    # Key legislative sources
    LEGISLATIVE_SOURCES = [
        {
            "name": "Florida Legislature",
            "url": "https://www.flsenate.gov",
            "keywords": ["artificial intelligence", "AI governance", "automated decision"],
        },
        {
            "name": "NIST AI RMF",
            "url": "https://www.nist.gov/artificial-intelligence",
            "keywords": ["risk management", "AI framework", "trustworthy AI"],
        },
        {
            "name": "GovAI Coalition",
            "url": "https://govaicoalition.org",
            "keywords": ["government AI", "municipal AI", "policy templates"],
        },
        {
            "name": "NACo AI Resources",
            "url": "https://www.naco.org",
            "keywords": ["county AI", "local government", "AI toolkit"],
        },
    ]

    def __init__(self, model_client: Any = None, http_client: Any = None):
        self.http_client = http_client
        self.alerts: List[LegislationAlert] = []
        super().__init__(
            name="Web Intelligence Agent",
            description="Legislation and peer county AI monitoring",
            agent_type="intelligence",
            capabilities=[
                "Track Florida AI legislation",
                "Monitor peer county AI programs",
                "Watch federal framework updates",
                "Generate leadership briefings",
            ],
            model_client=model_client,
        )

    def _register_tools(self):
        self.register_tool(
            "scan_legislation", self.scan_legislation, "Scan for new AI legislation"
        )
        self.register_tool(
            "monitor_peer_counties", self.monitor_peer_counties, "Check peer county AI updates"
        )
        self.register_tool(
            "generate_briefing", self.generate_briefing, "Generate intelligence briefing"
        )
        self.register_tool(
            "get_tracked_alerts", self.get_tracked_alerts, "Get all tracked alerts"
        )

    async def execute(self, context: AgentContext) -> AgentResult:
        request = context.request.lower()

        try:
            if "legislation" in request or "bill" in request or "statute" in request:
                result = await self.scan_legislation()
                return AgentResult(success=True, output=result, metadata={"operation": "legislation"})

            elif "peer" in request or "county" in request or "benchmark" in request:
                result = await self.monitor_peer_counties()
                return AgentResult(success=True, output=result, metadata={"operation": "peer_counties"})

            elif "briefing" in request or "summary" in request or "report" in request:
                result = await self.generate_briefing()
                return AgentResult(success=True, output=result, metadata={"operation": "briefing"})

            else:
                result = await self.generate_briefing()
                return AgentResult(success=True, output=result, metadata={"operation": "default_briefing"})

        except Exception as e:
            logger.error(f"Web Intelligence Agent failed: {e}")
            return AgentResult(success=False, output=None, error=str(e))

    async def scan_legislation(self, jurisdiction: str = "florida") -> Dict[str, Any]:
        """
        Scan for AI-related legislation.

        Args:
            jurisdiction: "florida", "federal", or "all"

        Returns:
            Legislation scan results with relevance scoring
        """
        sources = self.LEGISLATIVE_SOURCES
        if jurisdiction == "florida":
            sources = [s for s in sources if "Florida" in s["name"]]

        results = {
            "jurisdiction": jurisdiction,
            "scanned_at": datetime.now().isoformat(),
            "sources_checked": [s["name"] for s in sources],
            "alerts": [],
            "status": "scan_complete",
        }

        if self.http_client:
            for source in sources:
                try:
                    content = await self._fetch_source(source["url"])
                    if content:
                        matches = self._score_content(content, source["keywords"])
                        for match in matches:
                            alert = LegislationAlert(
                                title=match["title"],
                                jurisdiction=jurisdiction,
                                url=source["url"],
                                summary=match["snippet"],
                                relevance_score=match["score"],
                            )
                            self.alerts.append(alert)
                            results["alerts"].append(
                                {
                                    "title": alert.title,
                                    "source": source["name"],
                                    "relevance": alert.relevance_score,
                                    "summary": alert.summary,
                                }
                            )
                except Exception as e:
                    logger.warning(f"Failed to scan {source['name']}: {e}")
        else:
            results["note"] = (
                "No HTTP client configured. Provide an http_client to enable live scanning. "
                "Current data is from the knowledge base only."
            )

        return results

    async def monitor_peer_counties(self) -> Dict[str, Any]:
        """Monitor peer county AI initiatives"""
        results = {
            "monitored_counties": [],
            "scanned_at": datetime.now().isoformat(),
        }

        for county in self.PEER_COUNTIES:
            entry = {
                "name": county["name"],
                "state": county["state"],
                "url": county["url"],
                "status": "tracked",
            }

            if self.http_client:
                try:
                    content = await self._fetch_source(county["url"])
                    if content:
                        ai_mentions = self._score_content(
                            content,
                            ["artificial intelligence", "AI policy", "machine learning", "automation"],
                        )
                        entry["ai_activity_score"] = len(ai_mentions)
                        entry["recent_mentions"] = ai_mentions[:3]
                except Exception as e:
                    logger.warning(f"Failed to scan {county['name']}: {e}")
                    entry["status"] = "scan_failed"
            else:
                entry["status"] = "no_http_client"

            results["monitored_counties"].append(entry)

        return results

    async def generate_briefing(self) -> Dict[str, Any]:
        """Generate an intelligence briefing from all tracked alerts"""
        return {
            "briefing_date": datetime.now().isoformat(),
            "total_alerts": len(self.alerts),
            "by_jurisdiction": self._group_alerts_by_jurisdiction(),
            "top_alerts": sorted(
                [
                    {
                        "title": a.title,
                        "jurisdiction": a.jurisdiction,
                        "relevance": a.relevance_score,
                        "summary": a.summary,
                    }
                    for a in self.alerts
                ],
                key=lambda x: x["relevance"],
                reverse=True,
            )[:10],
            "peer_counties_tracked": len(self.PEER_COUNTIES),
            "sources_tracked": len(self.LEGISLATIVE_SOURCES),
        }

    async def get_tracked_alerts(self) -> Dict[str, Any]:
        """Return all tracked alerts"""
        return {
            "total": len(self.alerts),
            "alerts": [
                {
                    "title": a.title,
                    "jurisdiction": a.jurisdiction,
                    "url": a.url,
                    "relevance": a.relevance_score,
                    "discovered": a.discovered_at,
                }
                for a in self.alerts
            ],
        }

    def _group_alerts_by_jurisdiction(self) -> Dict[str, int]:
        groups: Dict[str, int] = {}
        for alert in self.alerts:
            groups[alert.jurisdiction] = groups.get(alert.jurisdiction, 0) + 1
        return groups

    async def _fetch_source(self, url: str) -> Optional[str]:
        """Fetch content from a URL using the configured HTTP client"""
        if not self.http_client:
            return None
        try:
            response = await self.http_client.get(url, timeout=15)
            return response.text if hasattr(response, "text") else str(response)
        except Exception as e:
            logger.warning(f"HTTP fetch failed for {url}: {e}")
            return None

    def _score_content(self, content: str, keywords: List[str]) -> List[Dict[str, Any]]:
        """Score content against keywords and extract relevant snippets"""
        content_lower = content.lower()
        matches = []

        for keyword in keywords:
            idx = 0
            while True:
                idx = content_lower.find(keyword.lower(), idx)
                if idx == -1:
                    break
                start = max(0, idx - 100)
                end = min(len(content), idx + len(keyword) + 200)
                snippet = content[start:end].strip()

                matches.append(
                    {
                        "title": keyword,
                        "snippet": snippet,
                        "score": content_lower.count(keyword.lower()) / max(len(content) / 1000, 1),
                    }
                )
                break  # one match per keyword

        return matches
