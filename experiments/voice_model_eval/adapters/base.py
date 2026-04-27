"""
Base adapter interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from schema import BackendConfig, EvalScenario, NormalizedSynthesisResult


class VoiceModelAdapter(ABC):
    def __init__(self, backend: BackendConfig):
        self.backend = backend

    @abstractmethod
    async def synthesize(self, scenario: EvalScenario) -> NormalizedSynthesisResult:
        raise NotImplementedError

