"""Abstract base class cho mọi stage trong pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from core.config import RunConfig, StageConfig
from core.types import StageOutput


class BaseStage(ABC):
    def __init__(self, run_config: RunConfig, stage_config: StageConfig):
        self.run_config = run_config
        self.stage_config = stage_config

    @property
    def name(self) -> str:
        return self.stage_config.name

    @abstractmethod
    def run(self, context: dict[str, Any]) -> StageOutput: ...
