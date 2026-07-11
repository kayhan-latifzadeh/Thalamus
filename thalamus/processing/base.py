"""The processing-stage abstraction.

A :class:`Stage` takes one :class:`~thalamus.protocol.Sample` and returns zero,
one, or several samples. All three cases are real:

* a filter returns one sample (possibly a *previous* one, if it needs lookahead);
* a delay returns zero samples until its buffer fills, then one per input;
* a dropout returns zero samples for a lost packet.

Returning a list rather than a single sample is what lets buffering stages exist
without a scheduler, and what lets the whole pipeline stay synchronous and
therefore trivially unit-testable.

Stages are *stateful* and *not* thread-safe: each pipeline owns its own stage
instances. The Core guarantees a pipeline is only ever driven from one task.

New stages register themselves with :func:`register`, which is all it takes for
them to become available to YAML configs and to remote clients by name.
"""

from __future__ import annotations

import copy
import json
import random
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Sequence, Type

from ..protocol import MISSING, Sample

_REGISTRY: Dict[str, Type[Stage]] = {}


def register(name: str):
    """Class decorator that makes a stage constructible by name from a config."""

    def decorator(cls: Type[Stage]) -> Type[Stage]:
        if name in _REGISTRY:
            raise ValueError(f"stage {name!r} is already registered to {_REGISTRY[name].__name__}")
        cls.stage_name = name
        _REGISTRY[name] = cls
        return cls

    return decorator


def available_stages() -> List[str]:
    """Names of every registered stage, for error messages and ``thalamus stages``."""
    return sorted(_REGISTRY)


class Stage(ABC):
    """Base class for a single step of a processing pipeline.

    Subclasses implement :meth:`apply`. Override :meth:`flush` only if the stage
    buffers samples that must be released at end of stream.

    ``channels`` restricts the stage to a subset of the signal's channels; when
    it is ``None`` the stage acts on every numeric channel it sees. Non-numeric
    and :data:`~thalamus.protocol.MISSING` values are passed through untouched by
    :meth:`map_values`, so a filter never has to think about gaps.
    """

    #: Set by @register.
    stage_name: str = "unnamed"

    def __init__(self, *, channels: Optional[Sequence[str]] = None) -> None:
        self.channels = list(channels) if channels is not None else None

    @abstractmethod
    def apply(self, sample: Sample) -> Iterable[Sample]:
        """Transform one sample into zero or more output samples."""

    def flush(self) -> Iterable[Sample]:
        """Release anything still buffered. Called when a stream ends."""
        return ()

    def reset(self) -> None:  # noqa: B027 - optional hook: a stateless stage needs no body
        """Discard all state, as if freshly constructed.

        Deliberately not abstract: most stages hold no state, and forcing every one
        of them to write an empty override would be noise.
        """

    # -- helpers for subclasses ------------------------------------------------

    def targets(self, sample: Sample) -> List[str]:
        """The channels of ``sample`` this stage should act on."""
        if self.channels is None:
            return [k for k, v in sample.data.items() if isinstance(v, (int, float))]
        return [c for c in self.channels if c in sample.data]

    def map_values(self, sample: Sample, fn) -> Sample:
        """Return a copy of ``sample`` with ``fn`` applied to each target channel.

        Channels holding :data:`MISSING` or a non-numeric value are left alone:
        it is never meaningful to add noise to a gap or to smooth a string.
        """
        out = sample.copy()
        for channel in self.targets(sample):
            value = out.data[channel]
            if value is MISSING or not isinstance(value, (int, float)):
                continue
            out.data[channel] = fn(channel, float(value))
        return out

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.stage_name}>"


class SeededStage(Stage):
    """A stage whose behaviour is random, and therefore must be reproducible.

    Noise injection and dropout are only scientifically useful if a colleague can
    re-run your study and get the same stream. Every stochastic stage takes a
    ``seed`` and draws from its own :class:`random.Random`, never from the global
    one, so two stages in the same pipeline cannot perturb each other's sequence.
    """

    def __init__(self, *, seed: Optional[int] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.seed = seed
        self._rng = random.Random(seed)

    def reset(self) -> None:
        self._rng = random.Random(self.seed)


class Pipeline:
    """An ordered chain of stages applied to one device's stream.

    Samples fan out through the chain: if stage 1 emits two samples, both are fed
    to stage 2. An empty pipeline is the identity, which is the common case (most
    subscriptions want the raw stream).
    """

    def __init__(self, stages: Optional[Sequence[Stage]] = None) -> None:
        self.stages: List[Stage] = list(stages or ())

    def __bool__(self) -> bool:
        return bool(self.stages)

    def __len__(self) -> int:
        return len(self.stages)

    def process(self, sample: Sample) -> List[Sample]:
        """Push one sample through every stage in order."""
        current = [sample]
        for stage in self.stages:
            if not current:
                break
            nxt: List[Sample] = []
            for item in current:
                nxt.extend(stage.apply(item))
            current = nxt
        return current

    def flush(self) -> List[Sample]:
        """Drain buffered samples, feeding each stage's leftovers downstream."""
        pending: List[Sample] = []
        for index, stage in enumerate(self.stages):
            carried = list(pending)
            pending = []
            for item in carried:
                pending.extend(stage.apply(item))
            pending.extend(stage.flush())
            del index
        return pending

    def reset(self) -> None:
        for stage in self.stages:
            stage.reset()

    def __repr__(self) -> str:
        return f"Pipeline({[s.stage_name for s in self.stages]})"


class PipelineSpecError(ValueError):
    """A pipeline description could not be turned into stages."""


def build_stage(spec: Dict[str, Any]) -> Stage:
    """Construct one stage from a plain dict, e.g. from YAML or from a client.

    ``{"stage": "gaussian_noise", "sigma": 0.5}`` becomes a
    :class:`~thalamus.processing.noise.GaussianNoiseStage`. The dict is otherwise
    passed straight to the constructor as keyword arguments, so a stage's config
    surface is exactly its ``__init__`` signature and there is no second schema
    to keep in sync.
    """
    if not isinstance(spec, dict):
        raise PipelineSpecError(f"a stage must be an object, got {type(spec).__name__}")

    params = dict(spec)
    name = params.pop("stage", None)
    if not name:
        raise PipelineSpecError(f"stage entry is missing a 'stage' key: {spec!r}")
    if name not in _REGISTRY:
        raise PipelineSpecError(
            f"unknown stage {name!r}; available stages: {', '.join(available_stages())}"
        )

    try:
        return _REGISTRY[name](**params)
    except PipelineSpecError:
        raise
    except TypeError as exc:
        raise PipelineSpecError(f"bad parameters for stage {name!r}: {exc}") from exc
    except (ValueError, ImportError) as exc:
        raise PipelineSpecError(f"could not build stage {name!r}: {exc}") from exc


def build_pipeline(spec: Optional[Iterable[Dict[str, Any]]]) -> Pipeline:
    """Construct a pipeline from a list of stage dicts. ``None`` yields a no-op."""
    if not spec:
        return Pipeline()
    if isinstance(spec, (dict, str)):
        raise PipelineSpecError("a pipeline must be a list of stage objects")
    return Pipeline([build_stage(item) for item in spec])


def pipeline_key(spec: Optional[Iterable[Dict[str, Any]]]) -> str:
    """A canonical string for a pipeline spec, used to share work between clients.

    Two subscriptions asking for the same device with the same pipeline must be
    served by a *single* pipeline instance, computed once and fanned out — that
    is the efficiency argument in §3.1 of the paper. Sorting keys makes the
    canonical form insensitive to how the spec happened to be written.
    """
    if not spec:
        return "[]"
    return json.dumps(copy.deepcopy(list(spec)), sort_keys=True, separators=(",", ":"))
