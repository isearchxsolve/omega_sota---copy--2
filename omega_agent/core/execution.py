"""DAG execution engine with parallel wave scheduling."""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from omega_agent.core.config import Config
from omega_agent.core.dag_schedule import compute_execution_waves
from omega_agent.core.types import ExecutionContext, ExecutionStatus, TaskNode
from omega_agent.tools.executor import ToolExecutor

logger = logging.getLogger("omega_agent.execution")


class DAGExecutor:
    """Execute task DAGs in parallel waves with retries, timeouts, and partial failure."""

    def __init__(self, config: Config, tool_executor: ToolExecutor):
        self.config = config
        self.tool_executor = tool_executor

    async def execute(
        self,
        dag: List[TaskNode],
        ctx: ExecutionContext,
    ) -> Dict[str, Any]:
        waves = compute_execution_waves(dag)
        logger.info("Executing DAG: %d tasks in %d parallel wave(s)", len(dag), len(waves))

        results: Dict[str, Any] = {}
        cost_lock = asyncio.Lock()

        for wave_idx, wave in enumerate(waves):
            if ctx.is_timed_out():
                logger.warning("Timeout — skipping remaining waves")
                break

            logger.info("Wave %d: running %d task(s) in parallel", wave_idx + 1, len(wave))
            outcomes = await asyncio.gather(
                *[self._run_task_safe(task, ctx, cost_lock) for task in wave],
                return_exceptions=False,
            )

            for task, (result, error) in zip(wave, outcomes):
                if error:
                    task.status = ExecutionStatus.FAILED
                    task.error = str(error)
                    ctx.add_error(f"Task {task.id} failed: {error}")
                    logger.error("Task %s failed: %s", task.id, error)
                else:
                    results[task.id] = result
                    ctx.task_results[task.id] = result
                    task.status = ExecutionStatus.SUCCESS
                    task.result = result
                    logger.info("Task %s completed", task.id)

        return results

    async def _run_task_safe(
        self,
        task: TaskNode,
        ctx: ExecutionContext,
        cost_lock: asyncio.Lock,
    ) -> tuple[Any, Optional[Exception]]:
        if ctx.is_timed_out():
            return None, RuntimeError(f"Task {task.id} skipped — execution timed out")
        try:
            result = await self._execute_task(task, ctx, cost_lock)
            return result, None
        except Exception as e:
            return None, e

    async def _execute_task(
        self,
        task: TaskNode,
        ctx: ExecutionContext,
        cost_lock: asyncio.Lock,
        attempt: int = 1,
    ) -> Any:
        if attempt > task.max_retries:
            raise RuntimeError(f"Task {task.id} failed after {task.max_retries} retries")

        task.status = ExecutionStatus.RUNNING
        logger.info("Executing task %s (attempt %d)", task.id, attempt)

        try:
            resolved_args = self._resolve_args(task.arguments, ctx.task_results)
            result, cost = await asyncio.wait_for(
                self.tool_executor.execute(task.tool_name, resolved_args, ctx.domain or "general"),
                timeout=task.timeout,
            )
            async with cost_lock:
                ctx.add_cost(cost)
            return result
        except asyncio.TimeoutError:
            logger.warning("Task %s timed out (attempt %d)", task.id, attempt)
        except Exception as e:
            logger.warning("Task %s attempt %d failed: %s", task.id, attempt, e)

        if attempt < task.max_retries:
            await asyncio.sleep(1.0 * attempt)
            return await self._execute_task(task, ctx, cost_lock, attempt + 1)
        raise RuntimeError(f"Task {task.id} exhausted retries")

    @staticmethod
    def _resolve_args(arguments: Dict[str, Any], prior_results: Dict[str, Any]) -> Dict[str, Any]:
        resolved = {}
        for key, value in arguments.items():
            if isinstance(value, str) and value.startswith("$"):
                ref = value[1:]
                resolved[key] = prior_results.get(ref, value)
            elif isinstance(value, list):
                resolved[key] = [
                    prior_results.get(v[1:], v) if isinstance(v, str) and v.startswith("$") else v
                    for v in value
                ]
            else:
                resolved[key] = value
        return resolved
