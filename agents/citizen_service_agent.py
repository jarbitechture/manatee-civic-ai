"""
Citizen Service Agent Implementation
Public-facing chatbot for government services
"""

from datetime import datetime
from typing import Any, Dict

from loguru import logger

from agents.base_agent import AgentContext, AgentResult, BaseAgent


class CitizenServiceAgent(BaseAgent):
    """
    Citizen service agent for public-facing government chatbot.
    Handles citizen inquiries with professional, helpful responses.
    """

    def __init__(self, model_client: Any = None):
        super().__init__(
            name="Citizen Service Agent",
            description="Public-facing chatbot for Manatee County services",
            agent_type="customer_service",
            capabilities=[
                "Answer county service questions",
                "Direct citizens to resources",
                "Form assistance",
                "311 service requests",
                "Department information",
                "Professional, friendly tone",
            ],
            model_client=model_client,
        )

        # Load knowledge base
        self.knowledge_base = self._load_knowledge_base()

    def _load_knowledge_base(self) -> Dict[str, Any]:
        """Load county services knowledge base"""
        # This would typically load from a database or file
        return {
            "departments": {
                "public_works": {
                    "name": "Public Works",
                    "phone": "(941) 749-3030",
                    "services": ["road maintenance", "drainage", "traffic signals"],
                    "hours": "Mon-Fri 8:00 AM - 5:00 PM",
                },
                "utilities": {
                    "name": "Utilities Department",
                    "phone": "(941) 792-8811",
                    "services": ["water service", "sewer", "billing"],
                    "hours": "Mon-Fri 8:00 AM - 5:00 PM",
                },
                "building": {
                    "name": "Building Department",
                    "phone": "(941) 748-4501",
                    "services": ["permits", "inspections", "code compliance"],
                    "hours": "Mon-Fri 8:00 AM - 5:00 PM",
                },
                "parks": {
                    "name": "Parks and Recreation",
                    "phone": "(941) 742-5923",
                    "services": ["park reservations", "programs", "facilities"],
                    "hours": "Mon-Fri 8:00 AM - 5:00 PM",
                },
                "sheriff": {
                    "name": "Sheriff's Office",
                    "phone": "(941) 747-3011",
                    "emergency": "911",
                    "services": ["non-emergency reports", "records", "community services"],
                },
            },
            "common_services": {
                "311": "For non-emergency county services, call 311 or (941) 748-4501",
                "permits": "Building permits can be applied for online at mymanatee.org",
                "taxes": "Property tax information: Tax Collector (941) 741-4800",
                "voting": "Elections Office: (941) 741-3823",
                "trash": "Waste collection schedules at mymanatee.org/solidwaste",
            },
            "faqs": [
                {
                    "question": "How do I pay my water bill?",
                    "answer": (
                        "You can pay online at mymanatee.org, "
                        "by phone at (941) 792-8811, by mail, "
                        "or in person at the Utilities office."
                    ),
                },
                {
                    "question": "How do I report a pothole?",
                    "answer": (
                        "Report potholes by calling 311 or using "
                        "the myManatee app. Include the location "
                        "and size if possible."
                    ),
                },
                {
                    "question": "Where can I get a building permit?",
                    "answer": (
                        "Building permits can be obtained online "
                        "at mymanatee.org or at the Building "
                        "Department, 1112 Manatee Ave W."
                    ),
                },
            ],
        }

    def _register_tools(self):
        """Register citizen service tools"""
        self.register_tool("answer_inquiry", self.answer_inquiry, "Answer citizen inquiry")
        self.register_tool("find_department", self.find_department, "Find department for service")
        self.register_tool(
            "create_311_request", self.create_311_request, "Create 311 service request"
        )
        self.register_tool(
            "get_department_info", self.get_department_info, "Get department information"
        )
        self.register_tool("check_faq", self.check_faq, "Check FAQ for answer")
        self.register_tool("escalate_to_human", self.escalate_to_human, "Escalate to human agent")

    async def execute(self, context: AgentContext) -> AgentResult:
        """Execute citizen service operations"""
        request = context.request.lower()

        try:
            # Check for department inquiries
            if "department" in request or any(
                dept in request
                for dept in ["public works", "utilities", "building", "parks", "sheriff"]
            ):
                dept = self._identify_department(request)
                result = await self.get_department_info(dept)
                return AgentResult(
                    success=True, output=result, metadata={"operation": "department_info"}
                )

            # Check for 311 request
            elif "311" in request or "report" in request or "request" in request:
                result = await self.create_311_request(context.request, context.user_id)
                return AgentResult(
                    success=True, output=result, metadata={"operation": "311_request"}
                )

            # Check FAQ
            elif "how" in request or "where" in request or "what" in request:
                faq_result = await self.check_faq(context.request)
                if faq_result.get("found"):
                    return AgentResult(
                        success=True, output=faq_result, metadata={"operation": "faq"}
                    )

            # Check for escalation request
            elif (
                "human" in request
                or "person" in request
                or "agent" in request
                or "representative" in request
            ):
                result = await self.escalate_to_human(context.request, context.user_id)
                return AgentResult(success=True, output=result, metadata={"operation": "escalate"})

            # Default: general inquiry
            result = await self.answer_inquiry(context.request)
            return AgentResult(success=True, output=result, metadata={"operation": "inquiry"})

        except Exception as e:
            logger.error(f"Citizen Service Agent execution failed: {e}")
            return AgentResult(success=False, output=None, error=str(e))

    def _identify_department(self, text: str) -> str:
        """Identify which department is relevant"""
        text = text.lower()

        mappings = {
            "public_works": ["road", "pothole", "drainage", "traffic", "street", "sidewalk"],
            "utilities": ["water", "sewer", "bill", "utility", "meter"],
            "building": ["permit", "inspection", "code", "construction", "building"],
            "parks": ["park", "recreation", "sports", "trail", "facility"],
            "sheriff": ["police", "crime", "safety", "emergency", "report"],
        }

        for dept, keywords in mappings.items():
            if any(kw in text for kw in keywords):
                return dept

        return "general"

    async def answer_inquiry(self, inquiry: str) -> Dict[str, Any]:
        """Answer a general citizen inquiry"""
        # Check FAQ first
        faq_result = await self.check_faq(inquiry)
        if faq_result.get("found"):
            return faq_result

        # Identify relevant department
        dept = self._identify_department(inquiry)
        dept_info = self.knowledge_base["departments"].get(dept, {})

        # Generate response with LLM
        response = await self.call_llm(
            prompt=f"""You are a helpful Manatee County government assistant.

Citizen inquiry: "{inquiry}"

Relevant department: {dept_info.get('name', 'General Services')}
Department phone: {dept_info.get('phone', '311')}
Services: {dept_info.get('services', [])}

Provide a helpful, professional response. Include:
1. Direct answer to their question if possible
2. Relevant contact information
3. Next steps they can take

Be friendly but professional. Use "we" when referring to the county.""",
            system_message="""You are a Manatee County citizen service representative.
Be helpful, professional, and friendly. Always provide actionable information.
If you don't know something specific, direct them to call 311 or the appropriate department.""",
        )

        return {
            "success": True,
            "inquiry": inquiry,
            "response": response,
            "department": dept_info.get("name", "General Services"),
            "contact": dept_info.get("phone", "311"),
        }

    async def find_department(self, service_needed: str) -> Dict[str, Any]:
        """Find the appropriate department for a service"""
        dept = self._identify_department(service_needed)
        dept_info = self.knowledge_base["departments"].get(dept, {})

        if dept_info:
            return {
                "success": True,
                "service": service_needed,
                "department": dept_info.get("name"),
                "phone": dept_info.get("phone"),
                "hours": dept_info.get("hours", "Contact for hours"),
                "services": dept_info.get("services", []),
            }
        else:
            return {
                "success": True,
                "service": service_needed,
                "recommendation": "For general inquiries, please call 311 or (941) 748-4501",
                "note": "A representative can direct you to the appropriate department",
            }

    async def create_311_request(self, description: str, user_id: str) -> Dict[str, Any]:
        """Create a 311 service request"""
        # In production, this would integrate with the 311 system
        request_id = f"SR-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        dept = self._identify_department(description)

        return {
            "success": True,
            "request_id": request_id,
            "status": "submitted",
            "description": description,
            "assigned_department": self.knowledge_base["departments"]
            .get(dept, {})
            .get("name", "General Services"),
            "message": (
                f"Your service request {request_id} has been "
                "submitted. You will receive a confirmation "
                "and updates via email."
            ),
            "next_steps": [
                "A representative will review your request within 1 business day",
                "You may track your request at mymanatee.org/311",
                "For urgent matters, please call 311 directly",
            ],
        }

    async def get_department_info(self, department: str) -> Dict[str, Any]:
        """Get information about a department"""
        dept_key = department.lower().replace(" ", "_")
        dept_info = self.knowledge_base["departments"].get(dept_key)

        if dept_info:
            return {
                "success": True,
                "department": dept_info.get("name"),
                "phone": dept_info.get("phone"),
                "hours": dept_info.get("hours", "Contact for hours"),
                "services": dept_info.get("services", []),
                "emergency": dept_info.get("emergency"),
            }
        else:
            # List all departments
            all_depts = [
                {"name": info.get("name"), "phone": info.get("phone")}
                for info in self.knowledge_base["departments"].values()
            ]
            return {
                "success": True,
                "message": "Here are our main departments:",
                "departments": all_depts,
                "general_info": "For general inquiries, call 311",
            }

    async def check_faq(self, question: str) -> Dict[str, Any]:
        """Check FAQ for a matching answer"""
        question_lower = question.lower()

        for faq in self.knowledge_base["faqs"]:
            # Simple keyword matching (could be enhanced with embeddings)
            faq_words = set(faq["question"].lower().split())
            question_words = set(question_lower.split())

            overlap = len(faq_words & question_words)
            if overlap >= 3:  # At least 3 words match
                return {
                    "success": True,
                    "found": True,
                    "matched_question": faq["question"],
                    "answer": faq["answer"],
                }

        return {"success": True, "found": False, "message": "No exact FAQ match found"}

    async def escalate_to_human(self, reason: str, user_id: str) -> Dict[str, Any]:
        """Escalate to a human agent"""
        return {
            "success": True,
            "escalated": True,
            "message": (
                "I understand you'd like to speak with a " "representative. Here are your options:"
            ),
            "options": [
                {
                    "method": "Phone",
                    "contact": "311 or (941) 748-4501",
                    "hours": "Mon-Fri 8:00 AM - 5:00 PM",
                },
                {
                    "method": "In Person",
                    "location": (
                        "Manatee County Administration Building, " "1112 Manatee Ave W, Bradenton"
                    ),
                    "hours": "Mon-Fri 8:00 AM - 5:00 PM",
                },
                {
                    "method": "Email",
                    "contact": "info@mymanatee.org",
                    "response_time": "Within 1-2 business days",
                },
            ],
            "ticket_created": True,
            "ticket_id": f"ESC-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "note": (
                "A representative will follow up within 1 "
                "business day if you've provided contact "
                "information."
            ),
        }
