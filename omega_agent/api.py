"""FastAPI REST API for OMEGA Agent."""

import asyncio
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field

from omega_agent import OmegaAgent, Config
from omega_agent.core.types import AgentResult

app = FastAPI(
    title="OMEGA Agent API",
    description="Multi-domain action-taking orchestrator",
    version="1.0.0",
)

_result_cache: Dict[str, Dict[str, Any]] = {}
_agent: Optional[OmegaAgent] = None


def get_agent() -> OmegaAgent:
    global _agent
    if _agent is None:
        _agent = OmegaAgent(config=Config())
    return _agent


class GoalRequest(BaseModel):
    goal: str = Field(..., min_length=1, description="Natural language goal")
    domain: Optional[str] = Field(None, description="Domain hint (auto-detected if omitted)")
    max_time: int = Field(300, ge=5, le=600, description="Max execution seconds")


class GoalResponse(BaseModel):
    request_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    cost: float = 0.0
    latency: float = 0.0


class HealthResponse(BaseModel):
    status: str
    version: str
    llm_mode: str
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    domains: list


@app.get("/health", response_model=HealthResponse)
async def health():
    agent = get_agent()
    provider = agent.config.active_llm_provider()
    return HealthResponse(
        status="ok",
        version="1.1.0",
        llm_mode="live" if provider else "mock",
        llm_provider=provider,
        llm_model=agent.config.primary_model if provider else None,
        domains=agent.list_tools(),
    )


@app.post("/v1/goals", response_model=GoalResponse)
async def submit_goal(req: GoalRequest, bg_tasks: BackgroundTasks):
    request_id = str(uuid.uuid4())
    _result_cache[request_id] = {"status": "running", "result": None}

    async def execute():
        agent = get_agent()
        result = await agent.run(goal=req.goal, domain=req.domain, max_time=req.max_time)
        _result_cache[request_id] = {
            "status": "completed" if result.success else "failed",
            "result": result.to_dict(),
            "cost": result.cost,
            "latency": result.latency,
        }

    bg_tasks.add_task(execute)
    return GoalResponse(request_id=request_id, status="running")


@app.post("/v1/goals/sync", response_model=GoalResponse)
async def submit_goal_sync(req: GoalRequest):
    """Synchronous goal execution — waits for result."""
    request_id = str(uuid.uuid4())
    agent = get_agent()
    result = await agent.run(goal=req.goal, domain=req.domain, max_time=req.max_time)
    _result_cache[request_id] = {
        "status": "completed" if result.success else "failed",
        "result": result.to_dict(),
    }
    return GoalResponse(
        request_id=request_id,
        status="completed" if result.success else "failed",
        result=result.to_dict(),
        cost=result.cost,
        latency=result.latency,
    )


@app.get("/v1/goals/{request_id}", response_model=GoalResponse)
async def get_goal_status(request_id: str):
    cached = _result_cache.get(request_id)
    if not cached:
        raise HTTPException(status_code=404, detail="Request not found")
    return GoalResponse(
        request_id=request_id,
        status=cached["status"],
        result=cached.get("result"),
        cost=cached.get("cost", 0.0),
        latency=cached.get("latency", 0.0),
    )


@app.get("/v1/metrics")
async def get_metrics():
    agent = get_agent()
    return {
        **agent.get_metrics(),
        "memory": agent.get_memory_stats(),
    }


@app.get("/v1/domains")
async def list_domains():
    agent = get_agent()
    return {
        "mode": "dynamic",
        "description": "Domains discovered per-goal via web search + LLM",
        "tools": agent.list_tools(),
    }
