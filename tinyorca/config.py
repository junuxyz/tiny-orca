from __future__ import annotations

from dataclasses import dataclass, field

from tinyorca.core.request import SamplingConfig


@dataclass(slots=True)
class OrcaConfig:
    """Shared runtime configuration for tinyorca submodules."""

    model: str | None = None
    max_batch_size: int = 4
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    gpu_utilization: float = 0.8

    def __post_init__(self) -> None:
        if self.max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        if not 0.0 < self.gpu_utilization <= 1.0:
            raise ValueError("gpu_utilization must be in (0, 1]")
