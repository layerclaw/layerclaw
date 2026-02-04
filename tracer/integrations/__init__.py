"""Framework integrations for Tracer."""

from tracer.integrations.transformers import TracerCallback as TransformersCallback

try:
    from tracer.integrations.lightning import TracerCallback as LightningCallback
except ImportError:
    LightningCallback = None  # type: ignore

__all__ = [
    "TransformersCallback",
    "LightningCallback",
]
