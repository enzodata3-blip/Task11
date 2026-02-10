"""
Advanced Statistical Analysis Module
Additional statistical techniques for deeper insights and model optimization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import PowerTransformer
from sklearn.feature_selection import (
    mutual_info_regression,
    SelectKBest,
    f_regression,
    RFE
)
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.model_selection import learning_curve
from scipy import stats
from scipy.stats import shapiro, jarque_bera, anderson, boxcox
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')


class AdvancedStatisticalAnalysis:
    """
    Advanced statistical analysis for ML optimization.

    Provides deep insights into data characteristics, feature relationships,
    and model behavior beyond basic correlation analysis.
    """

    def __init__(self, data: pd.DataFrame, target_col: str):
        """
        Initialize advanced analyzer.

        Parameters:
        -----------
        data : pd.DataFrame
            Input dataset
        target_col : str
            Target variable column name
        """
        self.data = data.copy()
        self.target_col = target_col
        self.features = [col for col in data.columns if col != target_col]
        self.numeric_features = data[self.features].select_dtypes(include=[np.number]).columns.tolist()

    def compute_vif(self, threshold: float = 10.0) -> pd.DataFrame:
        """
        Calculate Variance Inflation Factor (VIF) for multicollinearity detection.

        VIF measures how much the variance of a regression coefficient is inflated
        due to multicollinearity with other features.

        Rules of thumb:
        - VIF < 5: Low multicollinearity
        - VIF 5-10: Moderate multicollinearity
        - VIF > 10: High multicollinearity (problematic)

        Parameters:
        -----------
        threshold : float
            VIF threshold for flagging problematic features

        Returns:
        --------
        pd.DataFrame
            VIF scores for each feature
        """
        print("Computing Variance Inflation Factors (VIF)...")

        X = self.data[self.numeric_features].copy()

        # Remove any columns with zero variance
        X = X.loc[:, X.std() > 0]

        vif_data = []
        for i, col in enumerate(X.columns):
            vif = variance_inflation_factor(X.values, i)
            vif_data.append({
                'feature': col,
                'vif': vif,
                'multicollinearity': 'High' if vif > threshold else ('Moderate' if vif > 5 else 'Low')
            })

        vif_df = pd.DataFrame(vif_data).sort_values('vif', ascending=False)

        print(f"\nVIF Analysis Complete:")
        high_vif = vif_df[vif_df['vif'] > threshold]
        if len(high_vif) > 0:
            print(f"  ⚠️  {len(high_vif)} features with VIF > {threshold} (high multicollinearity)")
            for _, row in high_vif.iterrows():
                print(f"     • {row['feature']}: VIF = {row['vif']:.2f}")
        else:
            print(f"  ✓ No features with VIF > {threshold}")

        return vif_df

    def mutual_information_analysis(self, n_neighbors: int = 3) -> pd.DataFrame:
        """
        Calculate mutual information between features and target.

        Mutual information measures the dependency between variables, capturing
        both linear and non-linear relationships.

        Parameters:
        -----------
        n_neighbors : int
            Number of neighbors for MI estimation

        Returns:
        --------
        pd.DataFrame
            Mutual information scores
        """
        print(f"Computing mutual information scores (n_neighbors={n_neighbors})...")

        X = self.data[self.numeric_features]
        y = self.data[self.target_col]

        mi_scores = mutual_info_regression(X, y, n_neighbors=n_neighbors, random_state=42)

        mi_df = pd.DataFrame({
            'feature': self.numeric_features,
            'mutual_info': mi_scores,
            'mi_normalized': mi_scores / mi_scores.max() if mi_scores.max() > 0 else mi_scores
        }).sort_values('mutual_info', ascending=False)

        print(f"✓ Top 5 features by mutual information:")
        for _, row in mi_df.head(5).iterrows():
            print(f"  • {row['feature']}: MI = {row['mutual_info']:.4f}")

        return mi_df

    def normality_tests(self) -> pd.DataFrame:
        """
        Perform comprehensive normality tests on all numeric features.

        Tests performed:
        - Shapiro-Wilk test (best for small samples, n < 50)
        - Jarque-Bera test (based on skewness and kurtosis)
        - Anderson-Darling test (more sensitive than Shapiro-Wilk)

        Returns:
        --------
        pd.DataFrame
            Normality test results for all features
        """
        print("Performing normality tests on all numeric features...")

        results = []

        for col in self.numeric_features + [self.target_col]:
            data_col = self.data[col].dropna()

            # Shapiro-Wilk test
            if len(data_col) <= 5000:  # Shapiro-Wilk works best on smaller samples
                shapiro_stat, shapiro_p = shapiro(data_col)
            else:
                shapiro_stat, shapiro_p = np.nan, np.nan

            # Jarque-Bera test
            jb_stat, jb_p = jarque_bera(data_col)

            # Anderson-Darling test
            anderson_result = anderson(data_col, dist='norm')

            # Skewness and Kurtosis
            skewness = stats.skew(data_col)
            kurtosis = stats.kurtosis(data_col)

            results.append({
                'feature': col,
                'shapiro_statistic': shapiro_stat,
                'shapiro_pvalue': shapiro_p,
                'shapiro_normal': shapiro_p > 0.05 if not np.isnan(shapiro_p) else None,
                'jarque_bera_stat': jb_stat,
                'jarque_bera_pvalue': jb_p,
                'jarque_bera_normal': jb_p > 0.05,
                'anderson_statistic': anderson_result.statistic,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'interpretation': self._interpret_normality(shapiro_p, jb_p, skewness, kurtosis)
            })

        results_df = pd.DataFrame(results)

        non_normal = results_df[results_df['interpretation'] != 'Normal']
        print(f"\n✓ Normality tests complete:")
        print(f"  • {len(results_df) - len(non_normal)} features appear normally distributed")
        print(f"  • {len(non_normal)} features show non-normal distribution")

        return results_df

    def _interpret_normality(self, shapiro_p: float, jb_p: float,
                            skewness: float, kurtosis: float) -> str:
        """Interpret normality test results."""
        if np.isnan(shapiro_p):
            # Use JB test only
            if jb_p > 0.05:
                return 'Normal'
            elif abs(skewness) > 1:
                return 'Highly Skewed'
            elif abs(kurtosis) > 3:
                return 'Heavy Tails'
            else:
                return 'Non-Normal'
        else:
            # Use both tests
            if shapiro_p > 0.05 and jb_p > 0.05:
                return 'Normal'
            elif abs(skewness) > 1:
                return 'Highly Skewed'
            elif abs(kurtosis) > 3:
                return 'Heavy Tails'
            else:
                return 'Non-Normal'

    def recommend_transformations(self, normality_results: pd.DataFrame) -> Dict:
        """
        Recommend feature transformations based on normality tests.

        Parameters:
        -----------
        normality_results : pd.DataFrame
            Output from normality_tests()

        Returns:
        --------
        Dict
            Transformation recommendations
        """
        print("\nAnalyzing transformation recommendations...")

        recommendations = {}

        for _, row in normality_results.iterrows():
            feature = row['feature']

            if feature == self.target_col:
                continue

            if row['interpretation'] == 'Normal':
                recommendations[feature] = 'None - Already normal'
            elif row['interpretation'] == 'Highly Skewed':
                if row['skewness'] > 0:
                    if self.data[feature].min() > 0:
                        recommendations[feature] = 'Log transform (right skew, positive values)'
                    else:
                        recommendations[feature] = 'Yeo-Johnson transform (right skew, has negatives)'
                else:
                    recommendations[feature] = 'Reflect + Log or Yeo-Johnson (left skew)'
            elif row['interpretation'] == 'Heavy Tails':
                recommendations[feature] = 'Box-Cox or Yeo-Johnson (heavy tails)'
            else:
                recommendations[feature] = 'Power transform (general non-normality)'

        print(f"✓ Generated transformation recommendations for {len(recommendations)} features")

        return recommendations

    def apply_power_transform(self, method: str = 'yeo-johnson') -> pd.DataFrame:
        """
        Apply power transformation to normalize features.

        Box-Cox: Only for positive data
        Yeo-Johnson: Works with positive and negative data

        Parameters:
        -----------
        method : str
            'box-cox' or 'yeo-johnson'

        Returns:
        --------
        pd.DataFrame
            Transformed dataset
        """
        print(f"Applying {method} power transformation...")

        transformer = PowerTransformer(method=method, standardize=True)

        X = self.data[self.numeric_features].copy()
        y = self.data[self.target_col].copy()

        # Apply transformation
        X_transformed = transformer.fit_transform(X)
        X_transformed = pd.DataFrame(
            X_transformed,
            columns=self.numeric_features,
            index=self.data.index
        )

        # Combine with target
        transformed_data = X_transformed.copy()
        transformed_data[self.target_col] = y

        print(f"✓ Transformation complete")

        return transformed_data

    def feature_selection_comparison(self, X, y, n_features: int = 10) -> pd.DataFrame:
        """
        Compare multiple feature selection methods.

        Methods:
        - F-statistic (ANOVA F-value)
        - Mutual Information
        - RFE with Random Forest

        Parameters:
        -----------
        X : pd.DataFrame
            Features
        y : pd.Series
            Target
        n_features : int
            Number of top features to select

        Returns:
        --------
        pd.DataFrame
            Feature rankings from different methods
        """
        print(f"\nComparing feature selection methods (top {n_features})...")

        # Method 1: F-statistic
        f_selector = SelectKBest(score_func=f_regression, k=n_features)
        f_selector.fit(X, y)
        f_scores = pd.DataFrame({
            'feature': X.columns,
            'f_score': f_selector.scores_
        }).sort_values('f_score', ascending=False)

        # Method 2: Mutual Information
        mi_scores = mutual_info_regression(X, y, random_state=42)
        mi_df = pd.DataFrame({
            'feature': X.columns,
            'mi_score': mi_scores
        }).sort_values('mi_score', ascending=False)

        # Method 3: RFE
        from sklearn.ensemble import RandomForestRegressor
        rfe = RFE(
            estimator=RandomForestRegressor(n_estimators=50, random_state=42),
            n_features_to_select=n_features
        )
        rfe.fit(X, y)
        rfe_df = pd.DataFrame({
            'feature': X.columns,
            'rfe_rank': rfe.ranking_,
            'rfe_selected': rfe.support_
        }).sort_values('rfe_rank')

        # Combine results
        comparison = f_scores.merge(mi_df, on='feature').merge(rfe_df, on='feature')

        # Calculate consensus score (average normalized rank)
        comparison['f_rank'] = comparison['f_score'].rank(ascending=False)
        comparison['mi_rank'] = comparison['mi_score'].rank(ascending=False)
        comparison['avg_rank'] = (comparison['f_rank'] + comparison['mi_rank'] + comparison['rfe_rank']) / 3
        comparison = comparison.sort_values('avg_rank')

        print(f"\n✓ Top {min(n_features, 5)} features by consensus:")
        for _, row in comparison.head(min(n_features, 5)).iterrows():
            print(f"  • {row['feature']} (avg rank: {row['avg_rank']:.1f})")

        return comparison

    def analyze_feature_interactions_statistical(self) -> pd.DataFrame:
        """
        Statistical analysis of potential interactions using ANOVA.

        Tests for significant interaction effects between feature pairs.

        Returns:
        --------
        pd.DataFrame
            Statistical significance of interaction terms
        """
        print("Analyzing feature interactions with statistical tests...")

        from sklearn.preprocessing import StandardScaler
        from scipy.stats import f_oneway

        results = []
        features = self.numeric_features[:min(10, len(self.numeric_features))]  # Limit for computational efficiency

        for i, feat1 in enumerate(features):
            for feat2 in features[i+1:]:
                # Create interaction term
                interaction = self.data[feat1] * self.data[feat2]

                # Correlation with target
                corr_original = abs(self.data[feat1].corr(self.data[self.target_col]))
                corr_interaction = abs(interaction.corr(self.data[self.target_col]))

                # Additional correlation gain
                gain = corr_interaction - max(corr_original,
                                              abs(self.data[feat2].corr(self.data[self.target_col])))

                results.append({
                    'feature_1': feat1,
                    'feature_2': feat2,
                    'interaction_correlation': corr_interaction,
                    'correlation_gain': gain,
                    'promising': gain > 0.05
                })

        results_df = pd.DataFrame(results).sort_values('correlation_gain', ascending=False)

        promising = results_df[results_df['promising']]
        print(f"✓ Found {len(promising)} statistically promising interactions")

        return results_df

    def perform_pca_analysis(self, n_components: Optional[int] = None,
                            variance_threshold: float = 0.95) -> Dict:
        """
        Perform Principal Component Analysis for dimensionality reduction.

        Parameters:
        -----------
        n_components : Optional[int]
            Number of components (if None, determined by variance_threshold)
        variance_threshold : float
            Cumulative variance threshold for automatic component selection

        Returns:
        --------
        Dict
            PCA results including transformed data and explained variance
        """
        print(f"Performing PCA analysis...")

        X = self.data[self.numeric_features]

        # Fit PCA with all components first
        pca_full = PCA()
        pca_full.fit(X)

        # Determine n_components if not specified
        if n_components is None:
            cumsum = np.cumsum(pca_full.explained_variance_ratio_)
            n_components = np.argmax(cumsum >= variance_threshold) + 1

        # Fit PCA with selected components
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)

        # Create component names
        component_names = [f'PC{i+1}' for i in range(n_components)]
        X_pca_df = pd.DataFrame(X_pca, columns=component_names, index=self.data.index)

        # Component loadings
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=component_names,
            index=self.numeric_features
        )

        print(f"\n✓ PCA Results:")
        print(f"  • Original features: {len(self.numeric_features)}")
        print(f"  • Principal components: {n_components}")
        print(f"  • Variance explained: {pca.explained_variance_ratio_.sum():.1%}")

        return {
            'pca_model': pca,
            'transformed_data': X_pca_df,
            'loadings': loadings,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
            'n_components': n_components
        }

    def plot_pca_analysis(self, pca_results: Dict, save_path: Optional[str] = None):
        """
        Visualize PCA results.

        Parameters:
        -----------
        pca_results : Dict
            Output from perform_pca_analysis()
        save_path : Optional[str]
            Path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Scree plot
        components = range(1, len(pca_results['explained_variance_ratio']) + 1)
        axes[0, 0].bar(components, pca_results['explained_variance_ratio'], alpha=0.7)
        axes[0, 0].set_xlabel('Principal Component', fontsize=11)
        axes[0, 0].set_ylabel('Variance Explained', fontsize=11)
        axes[0, 0].set_title('Scree Plot', fontsize=12, fontweight='bold')
        axes[0, 0].grid(alpha=0.3)

        # 2. Cumulative variance
        axes[0, 1].plot(components, pca_results['cumulative_variance'],
                       marker='o', linewidth=2)
        axes[0, 1].axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
        axes[0, 1].set_xlabel('Number of Components', fontsize=11)
        axes[0, 1].set_ylabel('Cumulative Variance Explained', fontsize=11)
        axes[0, 1].set_title('Cumulative Variance Explained', fontsize=12, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

        # 3. Component loadings heatmap (first 3 components)
        n_show = min(3, pca_results['n_components'])
        loadings_subset = pca_results['loadings'].iloc[:, :n_show]
        sns.heatmap(loadings_subset, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, ax=axes[1, 0], cbar_kws={'label': 'Loading'})
        axes[1, 0].set_title(f'Component Loadings (First {n_show} PCs)',
                            fontsize=12, fontweight='bold')

        # 4. Biplot (PC1 vs PC2)
        if pca_results['n_components'] >= 2:
            transformed = pca_results['transformed_data']
            axes[1, 1].scatter(transformed['PC1'], transformed['PC2'], alpha=0.6)
            axes[1, 1].set_xlabel(f"PC1 ({pca_results['explained_variance_ratio'][0]:.1%})",
                                 fontsize=11)
            axes[1, 1].set_ylabel(f"PC2 ({pca_results['explained_variance_ratio'][1]:.1%})",
                                 fontsize=11)
            axes[1, 1].set_title('PCA Biplot (PC1 vs PC2)', fontsize=12, fontweight='bold')
            axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ PCA visualization saved to {save_path}")

        plt.show()

    def generate_comprehensive_report(self) -> Dict:
        """
        Generate comprehensive statistical analysis report.

        Returns:
        --------
        Dict
            Complete statistical analysis summary
        """
        print("\n" + "="*80)
        print("COMPREHENSIVE STATISTICAL ANALYSIS")
        print("="*80 + "\n")

        report = {}

        # 1. VIF Analysis
        try:
            report['vif'] = self.compute_vif()
        except Exception as e:
            print(f"⚠️  VIF analysis failed: {e}")
            report['vif'] = None

        # 2. Mutual Information
        try:
            report['mutual_info'] = self.mutual_information_analysis()
        except Exception as e:
            print(f"⚠️  Mutual information failed: {e}")
            report['mutual_info'] = None

        # 3. Normality Tests
        try:
            report['normality'] = self.normality_tests()
            report['transformations'] = self.recommend_transformations(report['normality'])
        except Exception as e:
            print(f"⚠️  Normality tests failed: {e}")
            report['normality'] = None

        # 4. Statistical Interaction Analysis
        try:
            report['statistical_interactions'] = self.analyze_feature_interactions_statistical()
        except Exception as e:
            print(f"⚠️  Statistical interaction analysis failed: {e}")
            report['statistical_interactions'] = None

        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80 + "\n")

        return report


if __name__ == "__main__":
    print("Advanced Statistical Analysis Module")
    print("=" * 80)
    print("\nThis module provides:")
    print("  • VIF analysis for multicollinearity detection")
    print("  • Mutual information for non-linear relationships")
    print("  • Comprehensive normality tests")
    print("  • Power transformations for normalization")
    print("  • Feature selection comparison")
    print("  • PCA for dimensionality reduction")
    print("  • Statistical interaction analysis")
    print("\nUsage example:")
    print("""
    from advanced_analysis import AdvancedStatisticalAnalysis

    analyzer = AdvancedStatisticalAnalysis(data=df, target_col='target')

    # Run comprehensive analysis
    report = analyzer.generate_comprehensive_report()

    # Individual analyses
    vif = analyzer.compute_vif()
    mi = analyzer.mutual_information_analysis()
    normality = analyzer.normality_tests()
    pca = analyzer.perform_pca_analysis()
    """)
