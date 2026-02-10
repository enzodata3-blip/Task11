"""
Correlation Analysis Module
Analyze correlation matrices to identify potential interaction terms and multicollinearity issues.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, kendalltau
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class CorrelationAnalyzer:
    """
    Comprehensive correlation analysis for feature engineering and interaction term discovery.

    Inspired by tidymodels/broom approach: tidy, interpretable statistical outputs.
    """

    def __init__(self, data: pd.DataFrame, target_col: str):
        """
        Initialize the correlation analyzer.

        Parameters:
        -----------
        data : pd.DataFrame
            Input dataframe containing features and target
        target_col : str
            Name of the target variable column
        """
        self.data = data.copy()
        self.target_col = target_col
        self.features = [col for col in data.columns if col != target_col]
        self.correlation_matrix = None
        self.target_correlations = None

    def compute_correlation_matrix(self, method: str = 'pearson') -> pd.DataFrame:
        """
        Compute correlation matrix using specified method.

        Parameters:
        -----------
        method : str
            Correlation method: 'pearson', 'spearman', or 'kendall'

        Returns:
        --------
        pd.DataFrame
            Correlation matrix
        """
        print(f"Computing {method} correlation matrix...")

        if method == 'pearson':
            self.correlation_matrix = self.data[self.features].corr(method='pearson')
        elif method == 'spearman':
            self.correlation_matrix = self.data[self.features].corr(method='spearman')
        elif method == 'kendall':
            self.correlation_matrix = self.data[self.features].corr(method='kendall')
        else:
            raise ValueError(f"Unknown correlation method: {method}")

        return self.correlation_matrix

    def compute_target_correlations(self, method: str = 'pearson') -> pd.DataFrame:
        """
        Compute correlations between features and target variable.

        Parameters:
        -----------
        method : str
            Correlation method

        Returns:
        --------
        pd.DataFrame
            Tidy dataframe with feature-target correlations
        """
        print(f"Computing feature-target correlations ({method})...")

        correlations = []
        for feature in self.features:
            if method == 'pearson':
                corr = self.data[feature].corr(self.data[self.target_col])
            elif method == 'spearman':
                corr, _ = spearmanr(self.data[feature], self.data[self.target_col])
            elif method == 'kendall':
                corr, _ = kendalltau(self.data[feature], self.data[self.target_col])
            else:
                raise ValueError(f"Unknown correlation method: {method}")

            correlations.append({
                'feature': feature,
                'correlation': corr,
                'abs_correlation': abs(corr)
            })

        self.target_correlations = pd.DataFrame(correlations).sort_values(
            'abs_correlation', ascending=False
        )

        return self.target_correlations

    def identify_multicollinearity(self, threshold: float = 0.8) -> pd.DataFrame:
        """
        Identify highly correlated feature pairs (potential multicollinearity).

        Parameters:
        -----------
        threshold : float
            Correlation threshold for identifying multicollinearity

        Returns:
        --------
        pd.DataFrame
            Tidy dataframe with highly correlated feature pairs
        """
        if self.correlation_matrix is None:
            self.compute_correlation_matrix()

        print(f"Identifying multicollinearity (threshold: {threshold})...")

        # Get upper triangle of correlation matrix
        upper_tri = np.triu(np.abs(self.correlation_matrix.values), k=1)

        # Find pairs exceeding threshold
        high_corr_pairs = []
        for i in range(len(self.correlation_matrix)):
            for j in range(i + 1, len(self.correlation_matrix)):
                if upper_tri[i, j] >= threshold:
                    high_corr_pairs.append({
                        'feature_1': self.correlation_matrix.index[i],
                        'feature_2': self.correlation_matrix.columns[j],
                        'correlation': self.correlation_matrix.iloc[i, j],
                        'abs_correlation': upper_tri[i, j]
                    })

        if len(high_corr_pairs) == 0:
            # Return empty DataFrame with proper columns
            return pd.DataFrame(columns=['feature_1', 'feature_2', 'correlation', 'abs_correlation'])

        return pd.DataFrame(high_corr_pairs).sort_values('abs_correlation', ascending=False)

    def identify_interaction_candidates(
        self,
        target_corr_threshold: float = 0.2,
        feature_corr_range: Tuple[float, float] = (0.1, 0.7),
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        Identify promising feature pairs for interaction terms.

        Logic: Look for feature pairs that:
        1. Both have moderate-to-strong correlation with target
        2. Have moderate correlation with each other (not too high, not too low)
        3. Could capture non-linear relationships

        Parameters:
        -----------
        target_corr_threshold : float
            Minimum absolute correlation with target for a feature to be considered
        feature_corr_range : Tuple[float, float]
            Range of inter-feature correlation to consider (min, max)
        top_n : int
            Number of top interaction candidates to return

        Returns:
        --------
        pd.DataFrame
            Tidy dataframe with interaction candidates and scores
        """
        if self.correlation_matrix is None:
            self.compute_correlation_matrix()
        if self.target_correlations is None:
            self.compute_target_correlations()

        print(f"Identifying interaction term candidates...")

        # Features with sufficient correlation to target
        relevant_features = self.target_correlations[
            self.target_correlations['abs_correlation'] >= target_corr_threshold
        ]['feature'].tolist()

        # Filter to only features that exist in correlation matrix
        relevant_features = [f for f in relevant_features if f in self.correlation_matrix.index]

        interaction_candidates = []

        for i, feat1 in enumerate(relevant_features):
            for feat2 in relevant_features[i + 1:]:
                # Get correlation between features
                feat_corr = abs(self.correlation_matrix.loc[feat1, feat2])

                # Check if correlation is in desired range
                if feature_corr_range[0] <= feat_corr <= feature_corr_range[1]:
                    # Get target correlations
                    feat1_target_corr = abs(self.target_correlations[
                        self.target_correlations['feature'] == feat1
                    ]['correlation'].values[0])

                    feat2_target_corr = abs(self.target_correlations[
                        self.target_correlations['feature'] == feat2
                    ]['correlation'].values[0])

                    # Compute interaction score (heuristic)
                    # Higher score = both features correlated with target,
                    # moderate correlation with each other
                    interaction_score = (feat1_target_corr + feat2_target_corr) * feat_corr

                    interaction_candidates.append({
                        'feature_1': feat1,
                        'feature_2': feat2,
                        'feature_1_target_corr': feat1_target_corr,
                        'feature_2_target_corr': feat2_target_corr,
                        'inter_feature_corr': feat_corr,
                        'interaction_score': interaction_score
                    })

        if len(interaction_candidates) == 0:
            print("⚠️  No interaction candidates found with current thresholds")
            # Return empty DataFrame with proper columns
            return pd.DataFrame(columns=[
                'feature_1', 'feature_2', 'feature_1_target_corr',
                'feature_2_target_corr', 'inter_feature_corr', 'interaction_score'
            ])

        result = pd.DataFrame(interaction_candidates).sort_values(
            'interaction_score', ascending=False
        ).head(top_n)

        print(f"Found {len(result)} interaction candidates")
        return result

    def plot_correlation_heatmap(self, figsize: Tuple[int, int] = (12, 10),
                                  annot: bool = False, save_path: Optional[str] = None):
        """
        Visualize correlation matrix as a heatmap.

        Parameters:
        -----------
        figsize : Tuple[int, int]
            Figure size
        annot : bool
            Whether to annotate cells with correlation values
        save_path : Optional[str]
            Path to save the figure
        """
        if self.correlation_matrix is None:
            self.compute_correlation_matrix()

        plt.figure(figsize=figsize)
        sns.heatmap(
            self.correlation_matrix,
            annot=annot,
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            fmt='.2f' if annot else None
        )
        plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Heatmap saved to {save_path}")

        plt.show()

    def plot_target_correlations(self, top_n: int = 20,
                                   save_path: Optional[str] = None):
        """
        Plot feature correlations with target variable.

        Parameters:
        -----------
        top_n : int
            Number of top features to display
        save_path : Optional[str]
            Path to save the figure
        """
        if self.target_correlations is None:
            self.compute_target_correlations()

        top_features = self.target_correlations.head(top_n)

        plt.figure(figsize=(10, max(6, len(top_features) * 0.3)))
        colors = ['red' if x < 0 else 'green' for x in top_features['correlation']]

        plt.barh(top_features['feature'], top_features['correlation'], color=colors, alpha=0.7)
        plt.xlabel('Correlation with Target', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.title(f'Top {top_n} Feature Correlations with Target',
                  fontsize=14, fontweight='bold')
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Target correlation plot saved to {save_path}")

        plt.show()

    def generate_summary_report(self) -> Dict:
        """
        Generate a comprehensive summary report of correlation analysis.

        Returns:
        --------
        Dict
            Summary statistics and key findings
        """
        if self.correlation_matrix is None:
            self.compute_correlation_matrix()
        if self.target_correlations is None:
            self.compute_target_correlations()

        multicollinearity = self.identify_multicollinearity(threshold=0.8)
        interaction_candidates = self.identify_interaction_candidates()

        report = {
            'num_features': len(self.features),
            'num_samples': len(self.data),
            'target_variable': self.target_col,
            'correlation_summary': {
                'mean_abs_correlation': self.correlation_matrix.abs().mean().mean(),
                'max_correlation': self.correlation_matrix.abs().max().max(),
                'num_high_correlations_0.8': len(multicollinearity),
                'num_high_correlations_0.5': len(
                    self.identify_multicollinearity(threshold=0.5)
                )
            },
            'target_correlation_summary': {
                'mean_abs_correlation': self.target_correlations['abs_correlation'].mean(),
                'max_correlation': self.target_correlations['abs_correlation'].max(),
                'min_correlation': self.target_correlations['abs_correlation'].min(),
                'top_5_features': self.target_correlations.head(5)['feature'].tolist()
            },
            'interaction_candidates': {
                'num_candidates': len(interaction_candidates),
                'top_5_pairs': [
                    f"{row['feature_1']} × {row['feature_2']}"
                    for _, row in interaction_candidates.head(5).iterrows()
                ]
            }
        }

        return report

    def print_report(self):
        """Print a formatted summary report."""
        report = self.generate_summary_report()

        print("\n" + "=" * 70)
        print("CORRELATION ANALYSIS REPORT")
        print("=" * 70)
        print(f"\nDataset Information:")
        print(f"  • Number of features: {report['num_features']}")
        print(f"  • Number of samples: {report['num_samples']}")
        print(f"  • Target variable: {report['target_variable']}")

        print(f"\nFeature Correlation Summary:")
        print(f"  • Mean absolute correlation: {report['correlation_summary']['mean_abs_correlation']:.3f}")
        print(f"  • Maximum correlation: {report['correlation_summary']['max_correlation']:.3f}")
        print(f"  • High correlations (>0.8): {report['correlation_summary']['num_high_correlations_0.8']}")
        print(f"  • Moderate correlations (>0.5): {report['correlation_summary']['num_high_correlations_0.5']}")

        print(f"\nTarget Correlation Summary:")
        print(f"  • Mean absolute correlation: {report['target_correlation_summary']['mean_abs_correlation']:.3f}")
        print(f"  • Max correlation: {report['target_correlation_summary']['max_correlation']:.3f}")
        print(f"  • Min correlation: {report['target_correlation_summary']['min_correlation']:.3f}")
        print(f"  • Top 5 features: {', '.join(report['target_correlation_summary']['top_5_features'])}")

        print(f"\nInteraction Term Candidates:")
        print(f"  • Number of candidates identified: {report['interaction_candidates']['num_candidates']}")
        print(f"  • Top 5 interaction pairs:")
        for i, pair in enumerate(report['interaction_candidates']['top_5_pairs'], 1):
            print(f"    {i}. {pair}")

        print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    # Example usage
    print("Correlation Analysis Module")
    print("This module provides comprehensive correlation analysis for ML feature engineering.")
    print("\nUsage example:")
    print("""
    from correlation_analysis import CorrelationAnalyzer

    # Initialize analyzer
    analyzer = CorrelationAnalyzer(data=df, target_col='target')

    # Compute correlations
    analyzer.compute_correlation_matrix(method='pearson')
    analyzer.compute_target_correlations()

    # Identify interaction candidates
    candidates = analyzer.identify_interaction_candidates()

    # Generate visualizations
    analyzer.plot_correlation_heatmap(save_path='results/correlation_heatmap.png')
    analyzer.plot_target_correlations(top_n=20)

    # Print summary report
    analyzer.print_report()
    """)
