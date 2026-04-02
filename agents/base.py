"""Base classes for agents and tools.

A Tool wraps an existing piece of inference code (predict.py, evaluate, etc.)
An Agent owns one or more Tools and implements a `run(state) -> state` contract.
"""

from __future__ import annotations

import logging
import time
import traceback
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from agents.state import TradingState

logger = logging.getLogger(__name__)


class BaseTool(ABC):
    """Wraps existing model inference code as a callable tool.

    Subclasses implement `execute(**kwargs) -> result`.
    Tools are stateless — all state flows through kwargs/return values.
    """

    name: str = "base_tool"
    description: str = ""

    def __init__(self, device: str = "auto", **kwargs):
        if device == "auto":
            import torch
            self.device_str = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device_str = device
        self._initialized = False

    def lazy_init(self):
        """Defer heavy imports / model loading until first call.

        Override this to load models, caches, etc.  Guarantees
        one-time initialization even if execute() is called many times.
        """
        self._initialized = True

    def __call__(self, **kwargs) -> Any:
        if not self._initialized:
            logger.info(f"[Tool:{self.name}] Initializing...")
            t0 = time.time()
            self.lazy_init()
            logger.info(f"[Tool:{self.name}] Ready ({time.time() - t0:.1f}s)")
        return self.execute(**kwargs)

    @abstractmethod
    def execute(self, *args: Any, **kwargs: Any) -> Any:
        """Run the tool.  Must be implemented by subclasses."""
        ...


class BaseAgent(ABC):
    """An agent reads TradingState, calls its Tools, and writes results back.

    Lifecycle:
        1. ``__init__`` — register tools
        2. ``run(state)`` — called by the pipeline orchestrator
        3. Agent reads relevant fields from ``state``
        4. Agent calls its tools
        5. Agent writes results to ``state``
        6. Returns the mutated state

    Agents should never import model code directly — only through Tools.
    """

    name: str = "base_agent"

    def __init__(self, tools: Optional[List[BaseTool]] = None, **kwargs):
        self.tools: Dict[str, BaseTool] = {}
        for tool in (tools or []):
            self.tools[tool.name] = tool
        self.config = kwargs

    def register_tool(self, tool: BaseTool):
        """Register a tool after __init__."""
        self.tools[tool.name] = tool

    def get_tool(self, name: str) -> BaseTool:
        if name not in self.tools:
            available = list(self.tools.keys())
            raise KeyError(f"Tool '{name}' not found. Available: {available}")
        return self.tools[name]

    def run(self, state: TradingState) -> TradingState:
        """Execute this agent.  Wraps _run with error handling + logging."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Agent: {self.name}")
        logger.info(f"{'='*60}")
        t0 = time.time()
        try:
            state = self._run(state)
        except Exception as e:
            logger.error(f"[{self.name}] FAILED: {e}")
            state.errors.append({
                "agent": self.name,
                "error": str(e),
                "traceback": traceback.format_exc(),
            })
        elapsed = time.time() - t0
        logger.info(f"[{self.name}] Completed in {elapsed:.1f}s")
        return state

    @abstractmethod
    def _run(self, state: TradingState) -> TradingState:
        """Implement agent logic here.  Mutate and return state."""
        ...
