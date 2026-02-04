"""HuggingFace Transformers integration."""

from typing import Any, Dict, Optional

try:
    from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
    
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    TrainerCallback = object  # type: ignore

import tracer


class TracerCallback(TrainerCallback):
    """
    Tracer callback for HuggingFace Transformers Trainer.

    Automatically integrates Tracer with HuggingFace training workflows.

    Example:
        >>> from transformers import Trainer, TrainingArguments
        >>> from tracer.integrations import TransformersCallback
        >>>
        >>> trainer = Trainer(
        ...     model=model,
        ...     args=training_args,
        ...     train_dataset=dataset,
        ...     callbacks=[TransformersCallback(project="my-project")]
        ... )
        >>> trainer.train()
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
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers is not installed. Install it with: pip install transformers"
            )

        self.project = project
        self.run_name = run_name
        self.checkpoint_interval = checkpoint_interval
        self.tracer_kwargs = kwargs
        self.tracer_instance = None

    def on_train_begin(
        self,
        args: "TrainingArguments",
        state: "TrainerState",
        control: "TrainerControl",
        model: Any = None,
        **kwargs: Any,
    ) -> None:
        """Initialize Tracer at training start."""
        self.tracer_instance = tracer.init(
            project=self.project,
            run_name=self.run_name,
            checkpoint_interval=self.checkpoint_interval,
            track_gradients=True,
            **self.tracer_kwargs,
        )

        if model is not None:
            tracer.watch(model)

    def on_log(
        self,
        args: "TrainingArguments",
        state: "TrainerState",
        control: "TrainerControl",
        logs: Optional[Dict[str, float]] = None,
        **kwargs: Any,
    ) -> None:
        """Log metrics to Tracer."""
        if logs and self.tracer_instance:
            # Filter out non-numeric values
            numeric_logs = {
                k: v for k, v in logs.items() if isinstance(v, (int, float))
            }
            if numeric_logs:
                tracer.log(numeric_logs)

    def on_step_end(
        self,
        args: "TrainingArguments",
        state: "TrainerState",
        control: "TrainerControl",
        **kwargs: Any,
    ) -> None:
        """Increment Tracer step."""
        if self.tracer_instance:
            tracer.step()

    def on_train_end(
        self,
        args: "TrainingArguments",
        state: "TrainerState",
        control: "TrainerControl",
        **kwargs: Any,
    ) -> None:
        """Finalize Tracer at training end."""
        if self.tracer_instance:
            tracer.finish()
