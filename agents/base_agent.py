"""
Base Agent Class
Foundation for all specialized agents with common functionality
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from loguru import logger


class AgentStatus(Enum):
    """Agent execution status"""

    IDLE = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    WAITING = "waiting"


@dataclass
class AgentContext:
    """Context passed between agents"""

    user_id: str
    session_id: str
    request: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    shared_state: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    """Result from agent execution"""

    success: bool
    output: Any
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    agent_name: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class BaseAgent(ABC):
    """
    Abstract base class for all specialized agents.
    Provides common functionality for execution, error handling, and tool management.
    """

    def __init__(
        self,
        name: str,
        description: str,
        agent_type: str,
        capabilities: List[str],
        model_client: Any = None,
    ):
        self.name = name
        self.description = description
        self.agent_type = agent_type
        self.capabilities = capabilities
        self.model_client = model_client
        self.status = AgentStatus.IDLE
        self.tools: Dict[str, Any] = {}
        self._register_tools()

        logger.info(f"Initialized {self.name} with {len(self.tools)} tools")

    @abstractmethod
    def _register_tools(self):
        """Register agent-specific tools. Must be implemented by subclasses."""

    @abstractmethod
    async def execute(self, context: AgentContext) -> AgentResult:
        """Execute the agent's main task. Must be implemented by subclasses."""

    def register_tool(self, name: str, func: Callable, description: str = ""):
        """Register a tool that this agent can use"""
        self.tools[name] = {"function": func, "description": description}
        logger.debug(f"{self.name}: Registered tool '{name}'")

    async def call_tool(self, tool_name: str, **kwargs) -> Any:
        """Call a registered tool"""
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not registered for {self.name}")

        tool = self.tools[tool_name]
        func = tool["function"]

        try:
            if asyncio.iscoroutinefunction(func):
                return await func(**kwargs)
            else:
                return func(**kwargs)
        except Exception as e:
            logger.error(f"{self.name}: Tool '{tool_name}' failed: {e}")
            raise

    async def run(self, context: AgentContext) -> AgentResult:
        """
        Main entry point for agent execution.
        Handles status tracking, timing, and error handling.
        """
        self.status = AgentStatus.RUNNING
        start_time = datetime.now()

        try:
            logger.info(f"{self.name}: Starting execution for request: {context.request[:100]}...")

            result = await self.execute(context)
            result.agent_name = self.name
            result.execution_time = (datetime.now() - start_time).total_seconds()

            self.status = AgentStatus.SUCCESS if result.success else AgentStatus.FAILED

            logger.info(
                f"{self.name}: Completed in "
                f"{result.execution_time:.2f}s - "
                f"{'SUCCESS' if result.success else 'FAILED'}"
            )

            return result

        except Exception as e:
            self.status = AgentStatus.FAILED
            execution_time = (datetime.now() - start_time).total_seconds()

            logger.error(f"{self.name}: Execution failed: {e}")

            return AgentResult(
                success=False,
                output=None,
                error=str(e),
                agent_name=self.name,
                execution_time=execution_time,
            )

    def get_tool_descriptions(self) -> List[Dict[str, str]]:
        """Get descriptions of all tools for LLM prompting"""
        return [
            {"name": name, "description": info["description"]} for name, info in self.tools.items()
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert agent info to dictionary"""
        return {
            "name": self.name,
            "description": self.description,
            "agent_type": self.agent_type,
            "capabilities": self.capabilities,
            "status": self.status.value,
            "tools": list(self.tools.keys()),
        }

    async def call_llm(self, prompt: str, system_message: Optional[str] = None) -> str:
        """Call the LLM with a prompt"""
        if not self.model_client:
            logger.warning(f"{self.name}: No model client available, returning placeholder")
            return f"[{self.name}] LLM response would go here for: {prompt[:50]}..."

        try:
            # Handle different model client types
            if hasattr(self.model_client, "create"):
                # Autogen-style client
                messages = []
                if system_message:
                    messages.append({"role": "system", "content": system_message})
                messages.append({"role": "user", "content": prompt})

                response = await self.model_client.create(messages=messages)
                return response.content if hasattr(response, "content") else str(response)

            elif hasattr(self.model_client, "chat"):
                # Direct API client
                response = self.model_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_message or f"You are {self.name}"},
                        {"role": "user", "content": prompt},
                    ]
                )
                return response.choices[0].message.content

            else:
                logger.warning(f"{self.name}: Unknown model client type")
                return f"[{self.name}] Could not process: {prompt[:50]}..."

        except Exception as e:
            logger.error(f"{self.name}: LLM call failed: {e}")
            return f"[{self.name}] Error: {str(e)}"


class AgentPool:
    """
    Manages a pool of agents and handles multi-agent coordination.
    """

    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.execution_history: List[Dict[str, Any]] = []

    def register(self, agent: BaseAgent):
        """Register an agent in the pool"""
        self.agents[agent.name] = agent
        logger.info(f"AgentPool: Registered '{agent.name}'")

    def get(self, name: str) -> Optional[BaseAgent]:
        """Get an agent by name"""
        return self.agents.get(name)

    def list_agents(self) -> List[Dict[str, Any]]:
        """List all registered agents"""
        return [agent.to_dict() for agent in self.agents.values()]

    async def execute_single(self, agent_name: str, context: AgentContext) -> AgentResult:
        """Execute a single agent"""
        agent = self.get(agent_name)
        if not agent:
            return AgentResult(
                success=False,
                output=None,
                error=f"Agent '{agent_name}' not found",
                agent_name=agent_name,
            )

        return await agent.run(context)

    async def execute_sequential(
        self, agent_names: List[str], context: AgentContext
    ) -> List[AgentResult]:
        """Execute agents sequentially, passing results forward"""
        results = []

        for agent_name in agent_names:
            result = await self.execute_single(agent_name, context)
            results.append(result)

            # Add result to shared state for next agent
            context.shared_state[agent_name] = {"output": result.output, "success": result.success}

            # Stop on failure if critical
            if not result.success and result.metadata.get("critical", False):
                logger.warning(
                    f"AgentPool: Stopping sequence due to critical failure in {agent_name}"
                )
                break

        return results

    async def execute_parallel(
        self, agent_names: List[str], context: AgentContext
    ) -> List[AgentResult]:
        """Execute agents in parallel"""
        tasks = [self.execute_single(name, context) for name in agent_names]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to AgentResults
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(
                    AgentResult(
                        success=False, output=None, error=str(result), agent_name=agent_names[i]
                    )
                )
            else:
                processed_results.append(result)  # type: ignore[arg-type]

        return processed_results

    def record_execution(self, results: List[AgentResult], context: AgentContext):
        """Record execution history"""
        self.execution_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "request": context.request,
                "user_id": context.user_id,
                "results": [
                    {
                        "agent": r.agent_name,
                        "success": r.success,
                        "execution_time": r.execution_time,
                    }
                    for r in results
                ],
            }
        )
