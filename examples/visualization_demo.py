"""
Demonstration of the new visualization system.
Shows how to use TrainingHistoryVisualizer, ComparisonVisualizer, and DivergenceVisualizer.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from fed_playground import (
    Environment,
    ClosedFormLinearRegressionModel,
    NoEncryption,
    MeanAggregation,
    DataLoader,
    TrainingHistoryVisualizer,
    ComparisonVisualizer,
    DivergenceVisualizer
)
from fed_playground.src.utils_data import generate_linear_data


def demo_training_history_visualizer():
    """Demonstrate TrainingHistoryVisualizer with simulated training history."""
    print("\n=== TrainingHistoryVisualizer Demo ===")

    # Initialize environment
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "test_data.csv")
    loader = DataLoader(file_path=data_path)
    env = Environment(
        n_parties=3,
        encryption_scheme=NoEncryption(),
        aggregation_strategy=MeanAggregation(),
        model_class=ClosedFormLinearRegressionModel,
        data_loader=loader
    )

    # Run simulation and collect history
    env.setup()
    history = env.run_simulation(rounds=10)

    # Prepare data for visualization
    training_data = {
        "Global Loss": history["global_loss"],
        "Avg Party Loss": history["party_loss"]
    }

    # Visualize training history
    visualizer = TrainingHistoryVisualizer(save_dir="demo_results")
    visualizer.plot(
        data=training_data,
        title="Federated Learning Training History",
        xlabel="Round",
        ylabel="Loss (MSE)",
        filename="training_history.png"
    )

    print("✓ Training history plot saved to demo_results/training_history.png")


def demo_comparison_visualizer():
    """Demonstrate ComparisonVisualizer comparing different models."""
    print("\n=== ComparisonVisualizer Demo ===")

    # Create synthetic data
    X, y = generate_linear_data(n_samples=1000, n_features=5, noise=0.1)
    X_train, y_train = X[:800], y[:800]
    X_test, y_test = X[800:], y[800:]

    # Train centralized model
    centralized_model = ClosedFormLinearRegressionModel(input_dim=5)
    centralized_model.train(X_train, y_train)
    centralized_mse = centralized_model.evaluate(X_test, y_test)

    # Train federated model
    loader = DataLoader(X=X_train, y=y_train)
    env = Environment(
        n_parties=5,
        encryption_scheme=NoEncryption(),
        aggregation_strategy=MeanAggregation(),
        model_class=ClosedFormLinearRegressionModel,
        data_loader=loader
    )
    env.setup()
    env.run_simulation(rounds=5, test_data=(X_test, y_test))

    # Get federated model performance
    federated_model = ClosedFormLinearRegressionModel(input_dim=5)
    federated_model.set_parameters(env.orchestrator.global_model_params)
    federated_mse = federated_model.evaluate(X_test, y_test)

    # Train local model (just party 0)
    local_model = env.parties[0].model
    local_mse = local_model.evaluate(X_test, y_test)

    # Compare models
    comparison_data = {
        "Centralized": centralized_mse,
        "Federated": federated_mse,
        "Local (Party 0)": local_mse
    }

    visualizer = ComparisonVisualizer(save_dir="demo_results")
    visualizer.plot(
        data=comparison_data,
        title="Model Performance Comparison",
        xlabel="Model Type",
        ylabel="Test MSE",
        filename="model_comparison.png",
        color=['green', 'blue', 'orange']
    )

    print("✓ Model comparison plot saved to demo_results/model_comparison.png")


def demo_divergence_visualizer():
    """Demonstrate DivergenceVisualizer for analyzing federated learning divergence."""
    print("\n=== DivergenceVisualizer Demo ===")

    # Create synthetic data
    X, y = generate_linear_data(n_samples=2000, n_features=5, noise=0.1)
    X_train, y_train = X[:1600], y[:1600]
    X_test, y_test = X[1600:], y[1600:]

    # Initialize visualizer
    visualizer = DivergenceVisualizer(save_dir="demo_results")

    # Run experiments with different numbers of parties
    for n_parties in [2, 4, 6, 8]:
        print(f"  Running experiment with {n_parties} parties...")

        # Train centralized model
        centralized_model = ClosedFormLinearRegressionModel(input_dim=5)
        centralized_model.train(X_train, y_train)
        gen_mse = centralized_model.evaluate(X_test, y_test)

        # Run multiple rounds
        metrics_per_round = []
        for round_num in range(5):
            # Train federated model
            loader = DataLoader(X=X_train, y=y_train)
            env = Environment(
                n_parties=n_parties,
                encryption_scheme=NoEncryption(),
                aggregation_strategy=MeanAggregation(),
                model_class=ClosedFormLinearRegressionModel,
                data_loader=loader
            )
            env.setup()
            env.run_round()

            # Evaluate federated model
            fed_model = ClosedFormLinearRegressionModel(input_dim=5)
            fed_model.set_parameters(env.orchestrator.global_model_params)
            fed_mse = fed_model.evaluate(X_test, y_test)

            # Calculate divergence metrics
            w_fed = fed_model.get_parameters()
            w_gen = centralized_model.get_parameters()
            norm_diff = np.linalg.norm(w_fed - w_gen)
            mse_diff = fed_mse - gen_mse
            mse_ratio = fed_mse / gen_mse if gen_mse != 0 else np.inf

            # Collect local model metrics
            local_mse = []
            local_normdiff = []
            local_msediff = []
            local_mseratio = []

            for party in env.parties:
                local_model = party.model
                l_mse = local_model.evaluate(X_test, y_test)
                w_local = local_model.get_parameters()
                l_norm_diff = np.linalg.norm(w_local - w_gen)
                l_mse_diff = l_mse - gen_mse
                l_mse_ratio = l_mse / gen_mse if gen_mse != 0 else np.inf

                local_mse.append(l_mse)
                local_normdiff.append(l_norm_diff)
                local_msediff.append(l_mse_diff)
                local_mseratio.append(l_mse_ratio)

            round_metrics = {
                "mse": fed_mse,
                "general_mse": gen_mse,
                "normdiff": norm_diff,
                "msediff": mse_diff,
                "mseratio": mse_ratio,
                "local_mse": local_mse,
                "local_normdiff": local_normdiff,
                "local_msediff": local_msediff,
                "local_mseratio": local_mseratio
            }
            metrics_per_round.append(round_metrics)

        visualizer.add_result(n_parties, metrics_per_round)

    # Generate all divergence plots
    visualizer.plot(
        x_label="Number of Parties",
        title_suffix="Party Count",
        show_local_models=True
    )

    print("✓ Divergence plots saved to demo_results/")
    print("  - norm_difference.png")
    print("  - mse_difference.png")
    print("  - mse_ratio.png")


def main():
    """Run all visualization demos."""
    print("=" * 60)
    print("Federated Learning Visualization Demo")
    print("=" * 60)

    # Create demo results directory
    os.makedirs("demo_results", exist_ok=True)

    # Run demos
    demo_training_history_visualizer()
    demo_comparison_visualizer()
    demo_divergence_visualizer()

    print("\n" + "=" * 60)
    print("All demos completed! Check the 'demo_results' directory.")
    print("=" * 60)


if __name__ == "__main__":
    main()
