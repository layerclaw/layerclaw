"""PyTorch Lightning integration."""

from typing import Any, Dict, Optional

try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import Callback
    
    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False
    Callback = object  # type: ignore

import tracer


class TracerCallback(Callback):
    """
    Tracer callback for PyTorch Lightning.

    Automatically integrates Tracer with Lightning training workflows.

    Example:
        >>> import pytorch_lightning as pl
        >>> from tracer.integrations import LightningCallback
        >>>
        >>> trainer = pl.Trainer(
        ...     callbacks=[LightningCallback(project="my-project")]
        ... )
        >>> trainer.fit(model, datamodule=dm)
    """

    def __init__(
        self,
        project: str,
        run_name: Optional[str] = None,
        checkpoint_interval: int = 100,
        **kwargs: Any,
    ) -> None:
        """
        Initialize TracerCallback.

        Args:
            project: Project name
            run_name: Optional run name
            checkpoint_interval: Steps between checkpoints
            **kwargs: Additional Tracer configuration
        """
        if not LIGHTNING_AVAILABLE:
            raise ImportError(
                "pytorch-lightning is not installed. "
                "Install it with: pip install pytorch-lightning"
            )

        self.project = project
        self.run_name = run_name
        self.checkpoint_interval = checkpoint_interval
        self.tracer_kwargs = kwargs
        self.tracer_instance = None

    def on_train_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        """Initialize Tracer at training start."""
        self.tracer_instance = tracer.init(
            project=self.project,
            run_name=self.run_name,
            checkpoint_interval=self.checkpoint_interval,
            track_gradients=True,
            **self.tracer_kwargs,
        )

        tracer.watch(pl_module)

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Log metrics and step after each batch."""
        if self.tracer_instance:
            # Log metrics from trainer
            metrics: Dict[str, float] = {}
            
            if trainer.callback_metrics:
                metrics.update(
                    {
                        k: v.item() if hasattr(v, "item") else float(v)
                        for k, v in trainer.callback_metrics.items()
                        if isinstance(v, (int, float)) or hasattr(v, "item")
                    }
                )

            if metrics:
                tracer.log(metrics)

            tracer.step()

    def on_train_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        """Finalize Tracer at training end."""
        if self.tracer_instance:
            tracer.finish()
