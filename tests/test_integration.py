"""Integration tests for full workflow."""

import torch
import torch.nn as nn

import tracer


class TestIntegration:
    """Test complete integration workflow."""

    def test_full_workflow(self, temp_dir):
        """Test complete training workflow with Tracer."""
        # Initialize
        tracer.init(
            project="integration_test",
            storage_path=str(temp_dir),
            checkpoint_interval=5,
            track_gradients=True,
        )

        # Create model
        model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))

        # Attach hooks
        tracer.watch(model)

        # Simulate training
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        for i in range(20):
            # Dummy data
            x = torch.randn(4, 10)
            y = torch.randn(4, 1)

            # Forward
            output = model(x)
            loss = criterion(output, y)

            # Log
            tracer.log({"loss": loss.item(), "step": i})

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Step
            tracer.step()

        # Finish
        tracer.finish()

        # Verify data was saved
        from tracer.analysis.query import QueryEngine

        query = QueryEngine(str(temp_dir))
        runs = query.storage.list_runs()

        assert len(runs) == 1
        assert runs[0]["project_name"] == "integration_test"
        assert runs[0]["total_steps"] == 20

    def test_multiple_runs(self, temp_dir):
        """Test multiple sequential runs."""
        for i in range(2):
            tracer.init(
                project="multi_run_test",
                run_name=f"run_{i}",
                storage_path=str(temp_dir),
                checkpoint_interval=5,
            )

            model = nn.Linear(5, 1)
            tracer.watch(model)

            for _ in range(10):
                x = torch.randn(2, 5)
                loss = model(x).sum()
                tracer.log({"loss": loss.item()})
                loss.backward()
                tracer.step()

            tracer.finish()

        # Verify both runs exist
        from tracer.analysis.query import QueryEngine

        query = QueryEngine(str(temp_dir))
        runs = query.storage.list_runs()

        assert len(runs) == 2
