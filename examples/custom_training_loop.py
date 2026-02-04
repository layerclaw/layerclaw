"""
Custom training loop example with Tracer.

Demonstrates advanced features like custom metrics, gradient clipping,
and learning rate scheduling.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

import tracer


class AdvancedModel(nn.Module):
    """More complex model with multiple layers."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(20, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.3)
        )
        self.hidden = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.2)
        )
        self.decoder = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.hidden(x)
        return self.decoder(x)


def main() -> None:
    """Run advanced training example."""
    # Initialize Tracer with custom configuration
    tracer.init(
        project="advanced-example",
        checkpoint_interval=50,
        track_gradients=True,
        track_system_metrics=True,
        tags=["experiment", "gradient-clipping", "cosine-schedule"],
        notes="Testing gradient clipping with cosine annealing scheduler",
    )

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AdvancedModel().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=100)
    criterion = nn.MSELoss()

    # Attach Tracer
    tracer.watch(model)

    # Create dataset
    X = torch.randn(2000, 20)
    y = torch.randn(2000, 1)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Training loop
    num_epochs = 5
    max_grad_norm = 1.0  # Gradient clipping threshold

    print(f"\nTraining on {device}...")

    for epoch in range(num_epochs):
        model.train()
        epoch_metrics = {"total_loss": 0.0, "batches": 0, "grad_norm": 0.0}

        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()

            # Log detailed metrics
            tracer.log(
                {
                    "loss": loss.item(),
                    "epoch": epoch,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "gradient_norm": grad_norm.item(),
                    "weight_decay": optimizer.param_groups[0]["weight_decay"],
                }
            )

            tracer.step()

            # Track epoch metrics
            epoch_metrics["total_loss"] += loss.item()
            epoch_metrics["grad_norm"] += grad_norm.item()
            epoch_metrics["batches"] += 1

        # Step scheduler
        scheduler.step()

        # Print epoch summary
        avg_loss = epoch_metrics["total_loss"] / epoch_metrics["batches"]
        avg_grad_norm = epoch_metrics["grad_norm"] / epoch_metrics["batches"]
        print(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Loss: {avg_loss:.4f}, "
            f"Grad Norm: {avg_grad_norm:.4f}, "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )

    # Finish
    tracer.finish()

    print("\nTraining complete!")
    print("Analyze with:")
    print("  tracer show <run-name>")
    print("  tracer anomalies <run-name> --auto")


if __name__ == "__main__":
    main()
