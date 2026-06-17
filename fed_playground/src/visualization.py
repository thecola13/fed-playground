import abc
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


class Visualizer(abc.ABC):
    """
    Abstract base class for visualization components.
    Follows the library pattern of ABC-based interfaces.
    """

    def __init__(
        self, save_dir: str | None = None, figsize: tuple[int, int] = (10, 6)
    ) -> None:
        """
        Initialize visualizer.

        Args:
            save_dir: Directory to save plots. If None, plots are displayed.
            figsize: Default figure size as (width, height).
        """
        self.save_dir = save_dir
        self.figsize = figsize

    @abc.abstractmethod
    def plot(self, data: Any, **kwargs) -> Figure:
        """
        Create a plot from the provided data.

        Args:
            data: Input data for visualization (format depends on visualizer type).
            **kwargs: Additional plotting parameters.

        Returns:
            Matplotlib Figure object.
        """
        pass

    def save_or_show(self, fig: Figure, filename: str) -> None:
        """
        Save figure to file or display it.

        Args:
            fig: Matplotlib figure to save or show.
            filename: Name of file to save (used if save_dir is set).
        """
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            filepath = os.path.join(self.save_dir, filename)
            fig.savefig(filepath, bbox_inches="tight", dpi=300)
            plt.close(fig)
        else:
            plt.show()


class TrainingHistoryVisualizer(Visualizer):
    """
    Visualizer for training history metrics over rounds.
    Plots loss curves, accuracy, or other metrics tracked during training.
    """

    def plot(
        self,
        data: dict[str, list[float]],
        title: str = "Training History",
        xlabel: str = "Round",
        ylabel: str = "Value",
        filename: str = "training_history.png",
        **kwargs,
    ) -> Figure:
        """
        Plot training history metrics.

        Args:
            data: Dictionary mapping metric names to lists of values.
                  Example: {"global_loss": [0.5, 0.3, 0.2], "party_0_loss": [0.6, 0.4, 0.3]}
            title: Plot title.
            xlabel: X-axis label.
            ylabel: Y-axis label.
            filename: Filename for saving.
            **kwargs: Additional matplotlib parameters.

        Returns:
            Matplotlib Figure object.
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        for metric_name, values in data.items():
            rounds = range(1, len(values) + 1)
            ax.plot(rounds, values, marker="o", label=metric_name, **kwargs)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        self.save_or_show(fig, filename)

        return fig


class ComparisonVisualizer(Visualizer):
    """
    Visualizer for comparing metrics across different models or configurations.
    Creates bar plots, grouped comparisons, or side-by-side visualizations.
    """

    def plot(
        self,
        data: dict[str, float],
        title: str = "Model Comparison",
        xlabel: str = "Model",
        ylabel: str = "Metric",
        filename: str = "comparison.png",
        **kwargs,
    ) -> Figure:
        """
        Create a bar plot comparing metrics across models.

        Args:
            data: Dictionary mapping model names to metric values.
                  Example: {"Federated": 0.25, "Centralized": 0.20, "Local": 0.35}
            title: Plot title.
            xlabel: X-axis label.
            ylabel: Y-axis label.
            filename: Filename for saving.
            **kwargs: Additional matplotlib parameters.

        Returns:
            Matplotlib Figure object.
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        models = list(data.keys())
        values = list(data.values())

        bars = ax.bar(models, values, **kwargs)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.4f}",
                ha="center",
                va="bottom",
            )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        self.save_or_show(fig, filename)

        return fig


class DivergenceVisualizer(Visualizer):
    """
    Visualizer for analyzing divergence between federated and centralized models.
    Specialized for federated learning experiments comparing model parameters and performance.
    """

    def __init__(
        self, save_dir: str | None = None, figsize: tuple[int, int] = (15, 10)
    ) -> None:
        """
        Initialize divergence visualizer.

        Args:
            save_dir: Directory to save plots.
            figsize: Figure size (default larger for complex plots).
        """
        super().__init__(save_dir, figsize)
        self.results: dict[Any, list[dict]] = {}

    def add_result(self, x_value: Any, metrics_per_round: list[dict]) -> None:
        """
        Add experimental results for a specific x-axis value.

        Args:
            x_value: X-axis value (e.g., number of parties, data size).
            metrics_per_round: List of metric dictionaries, one per round.
                Expected keys in each dict:
                - "mse": Aggregated model MSE
                - "general_mse": Centralized model MSE
                - "normdiff": Parameter norm difference
                - "msediff": MSE difference
                - "mseratio": MSE ratio
                - "local_mse": List of local model MSEs (optional)
                - "local_normdiff": List of local norm differences (optional)
                - "local_msediff": List of local MSE differences (optional)
                - "local_mseratio": List of local MSE ratios (optional)
        """
        self.results[x_value] = metrics_per_round

    def clear_results(self) -> None:
        """Clear stored results."""
        self.results = {}

    def plot(
        self,
        x_label: str = "X Value",
        title_suffix: str = "Experiment",
        show_local_models: bool = True,
        metric_configs: list[tuple[str, str, str]] | None = None,
    ) -> list[Figure]:
        """
        Generate divergence plots for all metrics.

        Args:
            x_label: Label for x-axis.
            title_suffix: Suffix for plot titles.
            show_local_models: Whether to show individual local model metrics.
            metric_configs: List of (metric_key, label, filename) tuples.
                          If None, uses default metrics.

        Returns:
            List of matplotlib Figure objects.
        """
        if not self.results:
            raise ValueError("No results to plot. Call add_result() first.")

        if metric_configs is None:
            metric_configs = [
                ("normdiff", "Norm Difference", "norm_difference.png"),
                ("msediff", "MSE Difference", "mse_difference.png"),
                ("mseratio", "MSE Ratio", "mse_ratio.png"),
            ]

        figures = []
        for key, label, fname in metric_configs:
            fig = self._plot_divergence_metric(
                metric_key=key,
                metric_label=label,
                title=f"{label} vs {title_suffix}",
                filename=fname,
                x_label=x_label,
                show_local_models=show_local_models,
            )
            figures.append(fig)

        return figures

    def _plot_divergence_metric(
        self,
        metric_key: str,
        metric_label: str,
        title: str,
        filename: str,
        x_label: str,
        show_local_models: bool,
    ) -> Figure:
        """
        Create a divergence plot for a single metric.

        Args:
            metric_key: Key for the metric in results dictionaries.
            metric_label: Label for the metric (y-axis).
            title: Plot title.
            filename: Filename for saving.
            x_label: X-axis label.
            show_local_models: Whether to include local model metrics.

        Returns:
            Matplotlib Figure object.
        """
        x_values = sorted(self.results.keys())
        base = np.arange(len(x_values), dtype=float)
        width = 0.35

        # Collect aggregated and local metrics
        agg_series = self._collect_metric_series(x_values, metric_key, use_local=False)
        local_series = (
            self._collect_metric_series(x_values, metric_key, use_local=True)
            if show_local_models
            else None
        )

        # Collect MSE values for line plots
        agg_mse = [np.mean([r["mse"] for r in self.results[x]]) for x in x_values]
        gen_mse = [self.results[x][0]["general_mse"] for x in x_values]

        # Create figure with two y-axes
        fig, ax_box = plt.subplots(figsize=self.figsize)

        # Plot boxplots for divergence metrics
        positions_agg = base - (width / 2 if show_local_models else 0.0)
        bp_agg = ax_box.boxplot(
            agg_series,
            positions=positions_agg,
            widths=width,
            patch_artist=True,
            tick_labels=None,
        )
        for box in bp_agg["boxes"]:
            box.set_alpha(0.7)
            box.set_facecolor("lightblue")

        if show_local_models and local_series:
            positions_local = base + width / 2
            bp_local = ax_box.boxplot(
                local_series,
                positions=positions_local,
                widths=width,
                patch_artist=True,
                tick_labels=None,
            )
            for box in bp_local["boxes"]:
                box.set_alpha(0.4)
                box.set_facecolor("orange")

        # Configure boxplot axis
        ax_box.set_xticks(base)
        ax_box.set_xticklabels(x_values)
        ax_box.set_xlabel(x_label)
        ax_box.set_ylabel(metric_label)
        ax_box.set_title(title)
        ax_box.grid(axis="y", alpha=0.3)

        # Create secondary axis for MSE lines
        ax_mse = ax_box.twinx()
        ax_mse.set_ylabel("MSE")

        (line_gen,) = ax_mse.plot(
            base,
            gen_mse,
            color="green",
            marker="x",
            linestyle="--",
            label="Centralized MSE",
            linewidth=2,
        )
        (line_agg,) = ax_mse.plot(
            base,
            agg_mse,
            color="blue",
            marker="o",
            linestyle=":",
            label="Federated MSE",
            linewidth=2,
        )

        # Build legend
        handles = [
            plt.Line2D(
                [0], [0], color="lightblue", lw=6, alpha=0.7, label="Federated Model"
            ),
            line_gen,
            line_agg,
        ]

        if show_local_models:
            local_mse = []
            for x in x_values:
                all_local = []
                for r in self.results[x]:
                    if "local_mse" in r:
                        all_local.extend(r["local_mse"])
                local_mse.append(np.mean(all_local) if all_local else np.nan)

            (line_local,) = ax_mse.plot(
                base,
                local_mse,
                color="orange",
                marker="s",
                linestyle=":",
                label="Local Models MSE",
                linewidth=2,
            )
            handles.extend(
                [
                    plt.Line2D(
                        [0], [0], color="orange", lw=6, alpha=0.4, label="Local Models"
                    ),
                    line_local,
                ]
            )

        ax_box.legend(
            handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.1)
        )

        plt.tight_layout()
        self.save_or_show(fig, filename)

        return fig

    def _collect_metric_series(
        self, x_values: list[Any], metric_key: str, use_local: bool
    ) -> list[list[float]]:
        """
        Collect metric values across x_values and rounds.

        Args:
            x_values: Sorted list of x-axis values.
            metric_key: Metric key to extract.
            use_local: If True, collect local model metrics; else aggregated.

        Returns:
            List of lists containing metric values for each x_value.
        """
        series = []
        for x in x_values:
            rounds_list = self.results[x]
            if use_local:
                # Collect all local values across rounds
                values = []
                for metrics_dict in rounds_list:
                    local_key = f"local_{metric_key}"
                    if local_key in metrics_dict:
                        values.extend(metrics_dict[local_key])
                series.append(values if values else [np.nan])
            else:
                # Collect aggregated values across rounds
                values = [metrics_dict[metric_key] for metrics_dict in rounds_list]
                series.append(values)

        return series


class PrivacyUtilityVisualizer(Visualizer):
    """Plot the privacy/utility trade-off curve for differentially-private FL.

    Visualises how model utility degrades as the privacy budget ``ε`` tightens —
    the canonical figure of the DP-ML literature (e.g. Abadi et al., "Deep
    Learning with Differential Privacy", ACM CCS 2016): utility on the y-axis
    against ``ε`` on a logarithmic x-axis, with one line per mechanism and an
    optional non-private baseline.

    Smaller ``ε`` means stronger privacy and more noise, so utility curves
    typically *worsen* to the left — the shape practitioners use to pick an
    operating point.
    """

    def plot(
        self,
        data: dict[str, dict[float, float]],
        title: str = "Privacy-Utility Trade-off",
        xlabel: str = "Privacy budget ε (log scale)",
        ylabel: str = "Utility (e.g. test MSE or accuracy)",
        filename: str = "privacy_utility.png",
        baseline: float | None = None,
        lower_is_better: bool = True,
        **kwargs,
    ) -> Figure:
        """Plot utility vs ε for one or more DP mechanisms.

        Args:
            data: Mapping ``mechanism_name -> {epsilon: utility}``.  Example:
                ``{"Laplace": {0.2: 74.0, 1.0: 3.0, 5.0: 0.13}}``.
            title: Plot title.
            xlabel: X-axis label.
            ylabel: Y-axis label.
            filename: Filename for saving.
            baseline: Optional non-private utility, drawn as a dashed reference line.
            lower_is_better: If ``True`` annotate that lower utility is better
                (MSE/deviance); set ``False`` for accuracy-style metrics.
            **kwargs: Additional matplotlib line parameters.

        Returns:
            Matplotlib Figure object.

        Raises:
            ValueError: If *data* is empty.
        """
        if not data:
            raise ValueError("No data to plot. Provide at least one mechanism.")

        fig, ax = plt.subplots(figsize=self.figsize)

        for name, eps_to_util in data.items():
            eps = sorted(eps_to_util)
            utils = [eps_to_util[e] for e in eps]
            ax.plot(eps, utils, marker="o", label=name, **kwargs)

        if baseline is not None:
            ax.axhline(
                baseline,
                color="grey",
                linestyle="--",
                linewidth=1.5,
                label="No privacy (baseline)",
            )

        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        direction = "lower is better" if lower_is_better else "higher is better"
        ax.set_ylabel(f"{ylabel}\n({direction})")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)

        plt.tight_layout()
        self.save_or_show(fig, filename)

        return fig


# Legacy alias for backwards compatibility
DivergencePlotter = DivergenceVisualizer
