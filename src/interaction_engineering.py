"""
Interaction Term Engineering Module
Create and evaluate interaction terms to enhance model performance.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import cross_val_score
from typing import List, Tuple, Dict, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class InteractionEngineer:
    """
    Engineer interaction terms based on correlation analysis and domain knowledge.

    The human element: systematic exploration and evaluation of interaction terms
    to guide the model beyond simple linear relationships.
    """

    def __init__(self, data: pd.DataFrame, target_col: str):
        """
        Initialize the interaction engineer.

        Parameters:
        -----------
        data : pd.DataFrame
            Input dataframe
        target_col : str
            Target variable column name
        """
        self.data = data.copy()
        self.target_col = target_col
        self.features = [col for col in data.columns if col != target_col]
        self.interaction_history = []
        self.enhanced_data = None

    def create_multiplicative_interaction(
        self,
        feature1: str,
        feature2: str,
        interaction_name: Optional[str] = None
    ) -> pd.Series:
        """
        Create multiplicative interaction: feature1 * feature2.

        Parameters:
        -----------
        feature1, feature2 : str
            Feature names to interact
        interaction_name : Optional[str]
            Name for the interaction term

        Returns:
        --------
        pd.Series
            Interaction term
        """
        if interaction_name is None:
            interaction_name = f"{feature1}_×_{feature2}"

        interaction = self.data[feature1] * self.data[feature2]
        interaction.name = interaction_name

        return interaction

    def create_polynomial_interaction(
        self,
        feature: str,
        degree: int = 2,
        interaction_name: Optional[str] = None
    ) -> pd.Series:
        """
        Create polynomial interaction: feature^degree.

        Parameters:
        -----------
        feature : str
            Feature name
        degree : int
            Polynomial degree
        interaction_name : Optional[str]
            Name for the interaction term

        Returns:
        --------
        pd.Series
            Polynomial term
        """
        if interaction_name is None:
            interaction_name = f"{feature}^{degree}"

        interaction = self.data[feature] ** degree
        interaction.name = interaction_name

        return interaction

    def create_ratio_interaction(
        self,
        numerator: str,
        denominator: str,
        interaction_name: Optional[str] = None,
        epsilon: float = 1e-8
    ) -> pd.Series:
        """
        Create ratio interaction: numerator / (denominator + epsilon).

        Parameters:
        -----------
        numerator, denominator : str
            Feature names
        interaction_name : Optional[str]
            Name for the interaction term
        epsilon : float
            Small constant to avoid division by zero

        Returns:
        --------
        pd.Series
            Ratio term
        """
        if interaction_name is None:
            interaction_name = f"{numerator}_{denominator}"

        interaction = self.data[numerator] / (self.data[denominator] + epsilon)
        interaction.name = interaction_name

        return interaction

    def create_difference_interaction(
        self,
        feature1: str,
        feature2: str,
        interaction_name: Optional[str] = None
    ) -> pd.Series:
        """
        Create difference interaction: feature1 - feature2.

        Parameters:
        -----------
        feature1, feature2 : str
            Feature names
        interaction_name : Optional[str]
            Name for the interaction term

        Returns:
        --------
        pd.Series
            Difference term
        """
        if interaction_name is None:
            interaction_name = f"{feature1}_minus_{feature2}"

        interaction = self.data[feature1] - self.data[feature2]
        interaction.name = interaction_name

        return interaction

    def create_log_interaction(
        self,
        feature: str,
        interaction_name: Optional[str] = None,
        offset: float = 1.0
    ) -> pd.Series:
        """
        Create log interaction: log(feature + offset).

        Parameters:
        -----------
        feature : str
            Feature name
        interaction_name : Optional[str]
            Name for the interaction term
        offset : float
            Offset to ensure positive values

        Returns:
        --------
        pd.Series
            Log-transformed term
        """
        if interaction_name is None:
            interaction_name = f"log_{feature}"

        # Ensure positive values
        min_val = self.data[feature].min()
        if min_val <= 0:
            offset = abs(min_val) + 1.0

        interaction = np.log(self.data[feature] + offset)
        interaction.name = interaction_name

        return interaction

    def batch_create_interactions(
        self,
        interaction_pairs: List[Tuple[str, str]],
        interaction_type: str = 'multiplicative'
    ) -> pd.DataFrame:
        """
        Create multiple interaction terms at once.

        Parameters:
        -----------
        interaction_pairs : List[Tuple[str, str]]
            List of feature pairs to interact
        interaction_type : str
            Type of interaction: 'multiplicative', 'ratio', 'difference'

        Returns:
        --------
        pd.DataFrame
            DataFrame with all interaction terms
        """
        interactions = []

        for feat1, feat2 in interaction_pairs:
            if interaction_type == 'multiplicative':
                interaction = self.create_multiplicative_interaction(feat1, feat2)
            elif interaction_type == 'ratio':
                interaction = self.create_ratio_interaction(feat1, feat2)
            elif interaction_type == 'difference':
                interaction = self.create_difference_interaction(feat1, feat2)
            else:
                raise ValueError(f"Unknown interaction type: {interaction_type}")

            interactions.append(interaction)

            # Record in history
            self.interaction_history.append({
                'feature_1': feat1,
                'feature_2': feat2,
                'interaction_name': interaction.name,
                'interaction_type': interaction_type
            })

        if len(interactions) == 0:
            print("⚠️  No interactions to create (empty interaction_pairs list)")
            # Return empty DataFrame
            return pd.DataFrame(index=self.data.index)

        return pd.concat(interactions, axis=1)

    def create_polynomial_features(
        self,
        features: Optional[List[str]] = None,
        degree: int = 2,
        include_bias: bool = False
    ) -> pd.DataFrame:
        """
        Create polynomial features using sklearn's PolynomialFeatures.

        Parameters:
        -----------
        features : Optional[List[str]]
            Features to polynomialize (default: all features)
        degree : int
            Maximum polynomial degree
        include_bias : bool
            Whether to include bias column

        Returns:
        --------
        pd.DataFrame
            DataFrame with polynomial features
        """
        if features is None:
            features = self.features

        poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
        poly_features = poly.fit_transform(self.data[features])

        # Create meaningful column names
        feature_names = poly.get_feature_names_out(features)

        return pd.DataFrame(poly_features, columns=feature_names, index=self.data.index)

    def evaluate_interaction_importance(
        self,
        interaction_terms: pd.DataFrame,
        estimator,
        cv: int = 5,
        scoring: str = 'r2'
    ) -> pd.DataFrame:
        """
        Evaluate the importance of interaction terms using cross-validation.

        Parameters:
        -----------
        interaction_terms : pd.DataFrame
            DataFrame containing interaction terms
        estimator : sklearn estimator
            Model to use for evaluation
        cv : int
            Number of cross-validation folds
        scoring : str
            Scoring metric

        Returns:
        --------
        pd.DataFrame
            Tidy dataframe with interaction term importance scores
        """
        print(f"Evaluating {len(interaction_terms.columns)} interaction terms...")

        # Prepare base data
        X_base = self.data[self.features]
        y = self.data[self.target_col]

        # Baseline score (without interactions)
        baseline_scores = cross_val_score(estimator, X_base, y, cv=cv, scoring=scoring)
        baseline_mean = baseline_scores.mean()

        print(f"Baseline {scoring}: {baseline_mean:.4f} (±{baseline_scores.std():.4f})")

        # Evaluate each interaction term individually
        results = []

        for col in interaction_terms.columns:
            X_with_interaction = pd.concat([X_base, interaction_terms[[col]]], axis=1)

            scores = cross_val_score(estimator, X_with_interaction, y, cv=cv, scoring=scoring)
            mean_score = scores.mean()
            improvement = mean_score - baseline_mean

            results.append({
                'interaction_term': col,
                'baseline_score': baseline_mean,
                'score_with_interaction': mean_score,
                'improvement': improvement,
                'improvement_pct': (improvement / abs(baseline_mean)) * 100,
                'std': scores.std()
            })

        if len(results) == 0:
            print("\n⚠️  No interaction terms to evaluate")
            # Return empty DataFrame with proper columns
            return pd.DataFrame(columns=[
                'interaction_term', 'baseline_score', 'score_with_interaction',
                'improvement', 'improvement_pct', 'std'
            ])

        results_df = pd.DataFrame(results).sort_values('improvement', ascending=False)

        print(f"\nTop 5 most valuable interactions:")
        for i, row in results_df.head(5).iterrows():
            print(f"  {row['interaction_term']}: "
                  f"+{row['improvement']:.4f} ({row['improvement_pct']:+.2f}%)")

        return results_df

    def add_interactions_to_data(
        self,
        interaction_terms: pd.DataFrame,
        inplace: bool = False
    ) -> pd.DataFrame:
        """
        Add interaction terms to the dataset.

        Parameters:
        -----------
        interaction_terms : pd.DataFrame
            Interaction terms to add
        inplace : bool
            Whether to modify self.data

        Returns:
        --------
        pd.DataFrame
            Enhanced dataset with interaction terms
        """
        enhanced_data = pd.concat([self.data, interaction_terms], axis=1)

        if inplace:
            self.enhanced_data = enhanced_data

        return enhanced_data

    def select_best_interactions(
        self,
        interaction_importance: pd.DataFrame,
        threshold: float = 0.0,
        top_n: Optional[int] = None
    ) -> List[str]:
        """
        Select best interaction terms based on importance scores.

        Parameters:
        -----------
        interaction_importance : pd.DataFrame
            Output from evaluate_interaction_importance()
        threshold : float
            Minimum improvement threshold
        top_n : Optional[int]
            Select top N interactions

        Returns:
        --------
        List[str]
            List of selected interaction term names
        """
        filtered = interaction_importance[interaction_importance['improvement'] > threshold]

        if top_n is not None:
            filtered = filtered.head(top_n)

        selected = filtered['interaction_term'].tolist()

        print(f"\nSelected {len(selected)} interaction terms:")
        for term in selected:
            print(f"  • {term}")

        return selected

    def generate_interaction_report(self) -> Dict:
        """
        Generate a summary report of interaction engineering.

        Returns:
        --------
        Dict
            Summary of interaction engineering process
        """
        report = {
            'num_original_features': len(self.features),
            'num_interactions_created': len(self.interaction_history),
            'interaction_types': pd.Series([
                h['interaction_type'] for h in self.interaction_history
            ]).value_counts().to_dict(),
            'interaction_names': [h['interaction_name'] for h in self.interaction_history]
        }

        return report

    def print_report(self):
        """Print a formatted interaction engineering report."""
        report = self.generate_interaction_report()

        print("\n" + "=" * 70)
        print("INTERACTION ENGINEERING REPORT")
        print("=" * 70)
        print(f"\nOriginal Features: {report['num_original_features']}")
        print(f"Interactions Created: {report['num_interactions_created']}")

        if report['interaction_types']:
            print(f"\nInteraction Types:")
            for int_type, count in report['interaction_types'].items():
                print(f"  • {int_type}: {count}")

        if report['num_interactions_created'] <= 20:
            print(f"\nCreated Interactions:")
            for name in report['interaction_names']:
                print(f"  • {name}")

        print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    # Example usage
    print("Interaction Engineering Module")
    print("This module creates and evaluates interaction terms for ML models.")
    print("\nUsage example:")
    print("""
    from interaction_engineering import InteractionEngineer
    from sklearn.ensemble import RandomForestRegressor

    # Initialize engineer
    engineer = InteractionEngineer(data=df, target_col='target')

    # Create specific interactions
    interaction1 = engineer.create_multiplicative_interaction('feature1', 'feature2')

    # Batch create interactions from correlation analysis
    interaction_pairs = [('feat1', 'feat2'), ('feat3', 'feat4')]
    interactions = engineer.batch_create_interactions(interaction_pairs)

    # Evaluate importance
    importance = engineer.evaluate_interaction_importance(
        interactions,
        estimator=RandomForestRegressor(),
        cv=5
    )

    # Select best interactions
    best = engineer.select_best_interactions(importance, threshold=0.01)

    # Add to dataset
    enhanced_df = engineer.add_interactions_to_data(interactions[best])
    """)
