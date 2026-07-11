"""Built-in processing stages (paper §2.4).

Importing this package is what populates the stage registry, so every stage below
becomes available by name to YAML configs and to remote clients.
"""

from .base import (  # noqa: F401
    Pipeline,
    PipelineSpecError,
    SeededStage,
    Stage,
    available_stages,
    build_pipeline,
    build_stage,
    pipeline_key,
    register,
)
from .delay import DelayStage, DropoutStage, PassthroughStage  # noqa: F401
from .filters import (  # noqa: F401
    ExponentialStage,
    KalmanStage,
    MovingAverageStage,
    SavitzkyGolayStage,
)
from .missing import (  # noqa: F401
    MissingFillStage,
    MissingInjectStage,
    ValidityMaskStage,
)
from .noise import ConstantNoiseStage, GaussianNoiseStage, UniformNoiseStage  # noqa: F401

__all__ = [
    "Pipeline",
    "PipelineSpecError",
    "Stage",
    "SeededStage",
    "available_stages",
    "build_pipeline",
    "build_stage",
    "pipeline_key",
    "register",
    "ConstantNoiseStage",
    "DelayStage",
    "DropoutStage",
    "ExponentialStage",
    "GaussianNoiseStage",
    "KalmanStage",
    "MissingFillStage",
    "MissingInjectStage",
    "MovingAverageStage",
    "PassthroughStage",
    "SavitzkyGolayStage",
    "UniformNoiseStage",
    "ValidityMaskStage",
]
