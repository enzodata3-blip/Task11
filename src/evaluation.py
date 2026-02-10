"""
Model Evaluation Module
Comprehensive evaluation and visualization of model performance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """
    Evaluate and visualize model performance with statistical rigor.

    Inspired by broom's tidy model outputs: clean, interpretable metrics.
    """

    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model"):
        """
        Initialize evaluator.

        Parameters:
        -----------
        y_true : np.ndarray
            True target values
        y_pred : np.ndarray
            Predicted values
        model_name : str
            Name of the model
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.model_name = model_name
        self.residuals = y_true - y_pred

    def compute_metrics(self) -> Dict:
        """
        Compute comprehensive regression metrics.

        Returns:
        --------
        Dict
            Dictionary of metrics
        """
        metrics = {
            'model_name': self.model_name,
            'n_samples': len(self.y_true),
            'r2': r2_score(self.y_true, self.y_pred),
            'adjusted_r2': self._adjusted_r2(),
            'rmse': np.sqrt(mean_squared_error(self.y_true, self.y_pred)),
            'mae': mean_absolute_error(self.y_true, self.y_pred),
            'mape': mean_absolute_percentage_error(self.y_true, self.y_pred) * 100,
            'mse': mean_squared_error(self.y_true, self.y_pred),
            'max_error': np.max(np.abs(self.residuals)),
            'mean_residual': np.mean(self.residuals),
            'std_residual': np.std(self.residuals),
            'skew_residual': stats.skew(self.residuals),
            'kurtosis_residual': stats.kurtosis(self.residuals)
        }

        return metrics

    def _adjusted_r2(self, n_features: int = 1) -> float:
        """Calculate adjusted R²."""
        n = len(self.y_true)
        r2 = r2_score(self.y_true, self.y_pred)
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
        return adj_r2

    def residual_analysis(self) -> Dict:
        """
        Perform residual analysis to check model assumptions.

        Returns:
        --------
        Dict
            Residual analysis results
        """
        # Shapiro-Wilk test for normality
        shapiro_stat, shapiro_pval = stats.shapiro(self.residuals)

        # Durbin-Watson test for autocorrelation
        dw_stat = self._durbin_watson()

        # Homoscedasticity check (visual inspection recommended)
        analysis = {
            'normality_test': {
                'test': 'Shapiro-Wilk',
                'statistic': shapiro_stat,
                'p_value': shapiro_pval,
                'is_normal': shapiro_pval > 0.05
            },
            'autocorrelation': {
                'test': 'Durbin-Watson',
                'statistic': dw_stat,
                'interpretation': self._interpret_dw(dw_stat)
            },
            'residual_stats': {
                'mean': np.mean(self.residuals),
                'std': np.std(self.residuals),
                'min': np.min(self.residuals),
                'max': np.max(self.residuals),
                'q25': np.percentile(self.residuals, 25),
                'q50': np.percentile(self.residuals, 50),
                'q75': np.percentile(self.residuals, 75)
            }
        }

        return analysis

    def _durbin_watson(self) -> float:
        """Calculate Durbin-Watson statistic."""
        diff_resid = np.diff(self.residuals)
        dw = np.sum(diff_resid ** 2) / np.sum(self.residuals ** 2)
        return dw

    def _interpret_dw(self, dw_stat: float) -> str:
        """Interpret Durbin-Watson statistic."""
        if dw_stat < 1.5:
            return "Positive autocorrelation"
        elif dw_stat > 2.5:
            return "Negative autocorrelation"
        else:
            return "No significant autocorrelation"

    def plot_predictions(self, save_path: Optional[str] = None, figsize: Tuple[int, int] = (10, 6)):
        """
        Plot predicted vs actual values.

        Parameters:
        -----------
        save_path : Optional[str]
            Path to save figure
        figsize : Tuple[int, int]
            Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Scatter plot
        ax.scatter(self.y_true, self.y_pred, alpha=0.6, edgecolors='k', linewidth=0.5)

        # Perfect prediction line
        min_val = min(self.y_true.min(), self.y_pred.min())
        max_val = max(self.y_true.max(), self.y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

        # Labels and title
        ax.set_xlabel('Actual Values', fontsize=12)
        ax.set_ylabel('Predicted Values', fontsize=12)
        ax.set_title(f'{self.model_name}: Predicted vs Actual', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        # Add R² annotation
        r2 = r2_score(self.y_true, self.y_pred)
        ax.text(0.05, 0.95, f'R² = {r2:.4f}',
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        plt.show()

    def plot_residuals(self, save_path: Optional[str] = None, figsize: Tuple[int, int] = (14, 10)):
        """
        Create comprehensive residual plots.

        Parameters:
        -----------
        save_path : Optional[str]
            Path to save figure
        figsize : Tuple[int, int]
            Figure size
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # 1. Residuals vs Predicted
        axes[0, 0].scatter(self.y_pred, self.residuals, alpha=0.6, edgecolors='k', linewidth=0.5)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0, 0].set_xlabel('Predicted Values', fontsize=11)
        axes[0, 0].set_ylabel('Residuals', fontsize=11)
        axes[0, 0].set_title('Residuals vs Predicted', fontsize=12, fontweight='bold')
        axes[0, 0].grid(alpha=0.3)

        # 2. Residual histogram
        axes[0, 1].hist(self.residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[0, 1].set_xlabel('Residuals', fontsize=11)
        axes[0, 1].set_ylabel('Frequency', fontsize=11)
        axes[0, 1].set_title('Residual Distribution', fontsize=12, fontweight='bold')
        axes[0, 1].grid(alpha=0.3)

        # 3. Q-Q plot
        stats.probplot(self.residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot', fontsize=12, fontweight='bold')
        axes[1, 0].grid(alpha=0.3)

        # 4. Scale-Location plot (sqrt of standardized residuals)
        standardized_residuals = self.residuals / np.std(self.residuals)
        axes[1, 1].scatter(self.y_pred, np.sqrt(np.abs(standardized_residuals)),
                          alpha=0.6, edgecolors='k', linewidth=0.5)
        axes[1, 1].set_xlabel('Predicted Values', fontsize=11)
        axes[1, 1].set_ylabel('√|Standardized Residuals|', fontsize=11)
        axes[1, 1].set_title('Scale-Location Plot', fontsize=12, fontweight='bold')
        axes[1, 1].grid(alpha=0.3)

        plt.suptitle(f'{self.model_name}: Residual Analysis', fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Residual plots saved to {save_path}")

        plt.show()

    def plot_error_distribution(self, save_path: Optional[str] = None, figsize: Tuple[int, int] = (12, 5)):
        """
        Plot error distribution analysis.

        Parameters:
        -----------
        save_path : Optional[str]
            Path to save figure
        figsize : Tuple[int, int]
            Figure size
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Absolute errors
        abs_errors = np.abs(self.residuals)

        # 1. Error distribution boxplot
        axes[0].boxplot([abs_errors], vert=True, labels=['Absolute Errors'])
        axes[0].set_ylabel('Absolute Error', fontsize=12)
        axes[0].set_title('Error Distribution (Box Plot)', fontsize=12, fontweight='bold')
        axes[0].grid(alpha=0.3)

        # 2. Cumulative error distribution
        sorted_errors = np.sort(abs_errors)
        cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100

        axes[1].plot(sorted_errors, cumulative, linewidth=2)
        axes[1].set_xlabel('Absolute Error', fontsize=12)
        axes[1].set_ylabel('Cumulative Percentage (%)', fontsize=12)
        axes[1].set_title('Cumulative Error Distribution', fontsize=12, fontweight='bold')
        axes[1].grid(alpha=0.3)

        # Add percentile lines
        for pct in [50, 75, 90, 95]:
            error_at_pct = np.percentile(abs_errors, pct)
            axes[1].axvline(x=error_at_pct, color='r', linestyle='--', alpha=0.5)
            axes[1].text(error_at_pct, pct, f' {pct}%', fontsize=9)

        plt.suptitle(f'{self.model_name}: Error Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Error distribution plots saved to {save_path}")

        plt.show()

    def print_evaluation_report(self):
        """Print comprehensive evaluation report."""
        metrics = self.compute_metrics()
        residual_analysis = self.residual_analysis()

        print("\n" + "=" * 70)
        print(f"MODEL EVALUATION REPORT: {self.model_name}")
        print("=" * 70)

        print("\nPerformance Metrics:")
        print(f"  • R²: {metrics['r2']:.4f}")
        print(f"  • Adjusted R²: {metrics['adjusted_r2']:.4f}")
        print(f"  • RMSE: {metrics['rmse']:.4f}")
        print(f"  • MAE: {metrics['mae']:.4f}")
        print(f"  • MAPE: {metrics['mape']:.2f}%")
        print(f"  • Max Error: {metrics['max_error']:.4f}")

        print("\nResidual Statistics:")
        print(f"  • Mean: {metrics['mean_residual']:.4f} (should be ≈ 0)")
        print(f"  • Std Dev: {metrics['std_residual']:.4f}")
        print(f"  • Skewness: {metrics['skew_residual']:.4f} (should be ≈ 0)")
        print(f"  • Kurtosis: {metrics['kurtosis_residual']:.4f} (should be ≈ 0)")

        print("\nModel Assumptions:")
        print(f"  • Normality (Shapiro-Wilk):")
        print(f"    - p-value: {residual_analysis['normality_test']['p_value']:.4f}")
        print(f"    - Residuals normal: {residual_analysis['normality_test']['is_normal']}")
        print(f"  • Autocorrelation (Durbin-Watson):")
        print(f"    - Statistic: {residual_analysis['autocorrelation']['statistic']:.4f}")
        print(f"    - {residual_analysis['autocorrelation']['interpretation']}")

        print("\n" + "=" * 70 + "\n")

    def generate_tidy_metrics(self) -> pd.DataFrame:
        """
        Generate tidy dataframe of metrics (broom-style).

        Returns:
        --------
        pd.DataFrame
            Tidy metrics dataframe
        """
        metrics = self.compute_metrics()

        return pd.DataFrame([metrics])


def compare_multiple_models(evaluators: List[ModelEvaluator]) -> pd.DataFrame:
    """
    Compare multiple models side-by-side.

    Parameters:
    -----------
    evaluators : List[ModelEvaluator]
        List of model evaluators

    Returns:
    --------
    pd.DataFrame
        Comparison table
    """
    comparison = pd.concat([ev.generate_tidy_metrics() for ev in evaluators], ignore_index=True)
    comparison = comparison.sort_values('r2', ascending=False)

    print("\n" + "=" * 100)
    print("MULTI-MODEL COMPARISON")
    print("=" * 100)
    print(comparison[['model_name', 'r2', 'adjusted_r2', 'rmse', 'mae', 'mape']].to_string(index=False))
    print("=" * 100 + "\n")

    return comparison


if __name__ == "__main__":
    # Example usage
    print("Model Evaluation Module")
    print("Comprehensive evaluation and diagnostics for regression models.")
    print("\nUsage example:")
    print("""
    from evaluation import ModelEvaluator, compare_multiple_models

    # Single model evaluation
    evaluator = ModelEvaluator(y_true=y_test, y_pred=predictions, model_name='Random Forest')

    # Print report
    evaluator.print_evaluation_report()

    # Visualizations
    evaluator.plot_predictions(save_path='results/predictions.png')
    evaluator.plot_residuals(save_path='results/residuals.png')
    evaluator.plot_error_distribution(save_path='results/errors.png')

    # Compare multiple models
    evaluators = [evaluator1, evaluator2, evaluator3]
    comparison = compare_multiple_models(evaluators)
    """)
