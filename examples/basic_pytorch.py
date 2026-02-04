"""
Basic PyTorch training example with Tracer.

This example demonstrates the simplest use case: tracking a basic PyTorch
training loop with gradient monitoring and metrics logging.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import tracer


class SimpleModel(nn.Module):
    """Simple feedforward neural network."""

    def __init__(self, input_dim: int = 10, hidden_dim: int = 64, output_dim: int = 1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def main() -> None:
    """Run basic training example."""
    # Initialize Tracer
    tracer.init(
        project="basic-example",
        run_name="simple-training",
        checkpoint_interval=100,  # Checkpoint every 100 steps
        track_gradients=True,  # Track gradient statistics
        track_system_metrics=True,  # Track CPU/GPU/memory
    )

    # Create model and optimizer
    model = SimpleModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Attach Tracer hooks to model (for gradient tracking)
    tracer.watch(model)

    # Create dummy dataset
    X = torch.randn(1000, 10)
    y = torch.randn(1000, 1)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Training loop
    num_epochs = 10
    print(f"\nTraining for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Log metrics
            tracer.log(
                {
                    "loss": loss.item(),
                    "epoch": epoch,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Step tracer (checkpoints automatically)
            tracer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

    # Finish tracer
    tracer.finish()

    print("\nTraining complete!")
    print("View results with: tracer show simple-training")
    print("Or list all runs: tracer list")


if __name__ == "__main__":
    main()
