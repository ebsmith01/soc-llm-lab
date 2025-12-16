
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, List


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: Dict[str, Any]
    fn: Callable[..., Any]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"Tool already registered: {spec.name}")
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name]

    def names(self) -> List[str]:
        return sorted(self._tools.keys())

    def specs(self) -> List[ToolSpec]:
        return [self._tools[k] for k in self.names()]

    def invoke(self, name: str, args: Optional[Dict[str, Any]] = None) -> Any:
        spec = self.get(name)
        args = args or {}
        return spec.fn(**args)