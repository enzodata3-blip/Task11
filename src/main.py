#!/usr/bin/env python3
"""
Main Orchestration Script
Complete workflow: data loading → correlation analysis → interaction engineering → model training → evaluation
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from data_processing import DataProcessor
from correlation_analysis import CorrelationAnalyzer
from interaction_engineering import InteractionEngineer
from model_training import ModelTrainer
from evaluation import ModelEvaluator, compare_multiple_models


class MLOptimizationPipeline:
    """
    Complete ML optimization pipeline with human-guided interaction term engineering.
    """

    def __init__(self, data_path: str, target_col: str, random_state: int = 42):
        """
        Initialize pipeline.

        Parameters:
        -----------
        data_path : str
            Path to dataset
        target_col : str
            Target variable column name
        random_state : int
            Random seed for reproducibility
        """
        self.data_path = data_path
        self.target_col = target_col
        self.random_state = random_state

        # Components
        self.processor = DataProcessor()
        self.analyzer = None
        self.engineer = None
        self.trainer = None

        # Results
        self.data = None
        self.enhanced_data = None
        self.results = {}

        print("\n" + "="*80)
        print("ML OPTIMIZATION PIPELINE INITIALIZED")
        print("="*80)
        print(f"Data: {data_path}")
        print(f"Target: {target_col}")
        print(f"Random State: {random_state}")
        print("="*80 + "\n")

    def step_1_load_and_process_data(
        self,
        missing_strategy: str = 'drop',
        handle_outliers: bool = True,
        outlier_method: str = 'iqr'
    ):
        """
        Step 1: Load and preprocess data.

        Parameters:
        -----------
        missing_strategy : str
            Strategy for missing values
        handle_outliers : bool
            Whether to handle outliers
        outlier_method : str
            Outlier detection method
        """
        print("\n" + "="*80)
        print("STEP 1: DATA LOADING AND PREPROCESSING")
        print("="*80 + "\n")

        # Load data
        self.data = self.processor.load_data(self.data_path)

        # Profile data
        self.processor.print_data_profile()

        # Handle missing values
        if self.processor.data.isnull().sum().sum() > 0:
            self.data = self.processor.handle_missing_values(strategy=missing_strategy)

        # Handle outliers
        if handle_outliers:
            self.data = self.processor.handle_outliers(method=outlier_method)

        # Encode categorical variables
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
        if self.target_col in categorical_cols:
            categorical_cols.remove(self.target_col)

        if categorical_cols:
            self.data = self.processor.encode_categorical_variables(
                columns=categorical_cols,
                method='onehot'
            )

        print(f"\n✓ Data preprocessing complete")
        print(f"  Final shape: {self.data.shape}")

        return self.data

    def step_2_correlation_analysis(
        self,
        method: str = 'pearson',
        plot: bool = True
    ):
        """
        Step 2: Analyze correlations and identify interaction candidates.

        Parameters:
        -----------
        method : str
            Correlation method
        plot : bool
            Whether to generate plots
        """
        print("\n" + "="*80)
        print("STEP 2: CORRELATION ANALYSIS")
        print("="*80 + "\n")

        self.analyzer = CorrelationAnalyzer(data=self.data, target_col=self.target_col)

        # Compute correlations
        self.analyzer.compute_correlation_matrix(method=method)
        self.analyzer.compute_target_correlations(method=method)

        # Identify interaction candidates
        interaction_candidates = self.analyzer.identify_interaction_candidates(
            target_corr_threshold=0.1,
            feature_corr_range=(0.1, 0.7),
            top_n=20
        )

        # Print report
        self.analyzer.print_report()

        # Generate plots
        if plot:
            os.makedirs('results', exist_ok=True)
            self.analyzer.plot_target_correlations(
                top_n=20,
                save_path='results/target_correlations.png'
            )
            self.analyzer.plot_correlation_heatmap(
                save_path='results/correlation_heatmap.png'
            )

        print(f"\n✓ Correlation analysis complete")
        print(f"  Interaction candidates identified: {len(interaction_candidates)}")

        self.results['interaction_candidates'] = interaction_candidates
        return interaction_candidates

    def step_3_engineer_interactions(
        self,
        top_n_interactions: int = 10,
        interaction_type: str = 'multiplicative'
    ):
        """
        Step 3: Create interaction terms.

        Parameters:
        -----------
        top_n_interactions : int
            Number of top interactions to create
        interaction_type : str
            Type of interactions
        """
        print("\n" + "="*80)
        print("STEP 3: INTERACTION ENGINEERING")
        print("="*80 + "\n")

        self.engineer = InteractionEngineer(data=self.data, target_col=self.target_col)

        # Get top interaction candidates
        candidates = self.results['interaction_candidates'].head(top_n_interactions)

        # Create interaction pairs
        interaction_pairs = [
            (row['feature_1'], row['feature_2'])
            for _, row in candidates.iterrows()
        ]

        # Batch create interactions
        interactions = self.engineer.batch_create_interactions(
            interaction_pairs,
            interaction_type=interaction_type
        )

        print(f"\n✓ Created {len(interactions.columns)} interaction terms")

        # Evaluate interaction importance
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=100, random_state=self.random_state, n_jobs=-1)

        importance = self.engineer.evaluate_interaction_importance(
            interactions,
            estimator=model,
            cv=5,
            scoring='r2'
        )

        # Select best interactions (positive improvement)
        best_interactions = self.engineer.select_best_interactions(
            importance,
            threshold=0.0,
            top_n=None
        )

        # Add to data
        if best_interactions:
            self.enhanced_data = self.engineer.add_interactions_to_data(
                interactions[best_interactions],
                inplace=False
            )
        else:
            print("Warning: No interactions improved the model. Using original data.")
            self.enhanced_data = self.data.copy()

        print(f"\n✓ Interaction engineering complete")
        print(f"  Best interactions selected: {len(best_interactions)}")

        self.results['interaction_importance'] = importance
        self.results['best_interactions'] = best_interactions

        return self.enhanced_data

    def step_4_train_models(self):
        """
        Step 4: Train baseline and enhanced models.
        """
        print("\n" + "="*80)
        print("STEP 4: MODEL TRAINING")
        print("="*80 + "\n")

        # Initialize trainer with baseline data
        self.trainer = ModelTrainer(
            data=self.data,
            target_col=self.target_col,
            test_size=0.2,
            random_state=self.random_state,
            scale_features=True
        )

        # Train baseline models
        print("Training baseline models...")
        baseline_results = self.trainer.train_baseline_models(cv=5)

        # Train enhanced model
        if self.enhanced_data is not None:
            print("\nTraining enhanced model with interactions...")
            enhanced_results = self.trainer.train_enhanced_model(
                enhanced_data=self.enhanced_data,
                model_name='Enhanced Random Forest',
                cv=5
            )

        # Print comparison
        self.trainer.print_comparison()

        print(f"\n✓ Model training complete")

        self.results['baseline_results'] = baseline_results
        if self.enhanced_data is not None:
            self.results['enhanced_results'] = enhanced_results

        return self.trainer

    def step_5_evaluate_models(self):
        """
        Step 5: Comprehensive model evaluation.
        """
        print("\n" + "="*80)
        print("STEP 5: MODEL EVALUATION")
        print("="*80 + "\n")

        os.makedirs('results', exist_ok=True)

        evaluators = []

        # Evaluate baseline models
        for name, results in self.trainer.baseline_results.items():
            evaluator = ModelEvaluator(
                y_true=self.trainer.y_test,
                y_pred=results['predictions_test'],
                model_name=f"Baseline - {name}"
            )
            evaluator.print_evaluation_report()
            evaluators.append(evaluator)

        # Evaluate enhanced models
        for name, results in self.trainer.enhanced_results.items():
            evaluator = ModelEvaluator(
                y_true=results['y_test'],
                y_pred=results['predictions_test'],
                model_name=name
            )
            evaluator.print_evaluation_report()

            # Generate plots for best enhanced model
            evaluator.plot_predictions(save_path=f'results/{name.replace(" ", "_")}_predictions.png')
            evaluator.plot_residuals(save_path=f'results/{name.replace(" ", "_")}_residuals.png')
            evaluator.plot_error_distribution(save_path=f'results/{name.replace(" ", "_")}_errors.png')

            evaluators.append(evaluator)

        # Compare all models
        comparison = compare_multiple_models(evaluators)

        print(f"\n✓ Model evaluation complete")
        print(f"  Best model: {comparison.iloc[0]['model_name']}")
        print(f"  Best R²: {comparison.iloc[0]['r2']:.4f}")

        self.results['model_comparison'] = comparison

        return comparison

    def step_6_feature_importance(self):
        """
        Step 6: Analyze feature importance.
        """
        print("\n" + "="*80)
        print("STEP 6: FEATURE IMPORTANCE ANALYSIS")
        print("="*80 + "\n")

        # Get feature importance for enhanced model
        if self.trainer.enhanced_results:
            model_name = list(self.trainer.enhanced_results.keys())[0]
            importance = self.trainer.get_feature_importance(model_name)

            if importance is not None:
                print(f"Top 20 Most Important Features:")
                print(importance.head(20).to_string(index=False))

                # Save to file
                importance.to_csv('results/feature_importance.csv', index=False)

                self.results['feature_importance'] = importance

                print(f"\n✓ Feature importance saved to results/feature_importance.csv")

        return importance

    def run_full_pipeline(
        self,
        missing_strategy: str = 'drop',
        handle_outliers: bool = True,
        correlation_method: str = 'pearson',
        top_n_interactions: int = 10
    ):
        """
        Run the complete optimization pipeline.

        Parameters:
        -----------
        missing_strategy : str
            Strategy for missing values
        handle_outliers : bool
            Whether to handle outliers
        correlation_method : str
            Correlation method
        top_n_interactions : int
            Number of interactions to create
        """
        print("\n" + "="*80)
        print("🚀 STARTING FULL ML OPTIMIZATION PIPELINE")
        print("="*80)

        # Step 1: Load and process data
        self.step_1_load_and_process_data(
            missing_strategy=missing_strategy,
            handle_outliers=handle_outliers
        )

        # Step 2: Correlation analysis
        self.step_2_correlation_analysis(
            method=correlation_method,
            plot=True
        )

        # Step 3: Engineer interactions
        self.step_3_engineer_interactions(
            top_n_interactions=top_n_interactions
        )

        # Step 4: Train models
        self.step_4_train_models()

        # Step 5: Evaluate models
        self.step_5_evaluate_models()

        # Step 6: Feature importance
        self.step_6_feature_importance()

        print("\n" + "="*80)
        print("✅ PIPELINE COMPLETE!")
        print("="*80)
        print("\nResults saved to:")
        print("  • results/target_correlations.png")
        print("  • results/correlation_heatmap.png")
        print("  • results/Enhanced_Random_Forest_predictions.png")
        print("  • results/Enhanced_Random_Forest_residuals.png")
        print("  • results/Enhanced_Random_Forest_errors.png")
        print("  • results/feature_importance.csv")
        print("\n" + "="*80 + "\n")

        return self.results


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='ML Optimization Pipeline')
    parser.add_argument('--data', type=str, required=True, help='Path to dataset')
    parser.add_argument('--target', type=str, required=True, help='Target column name')
    parser.add_argument('--interactions', type=int, default=10, help='Number of interactions')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Initialize and run pipeline
    pipeline = MLOptimizationPipeline(
        data_path=args.data,
        target_col=args.target,
        random_state=args.seed
    )

    results = pipeline.run_full_pipeline(
        top_n_interactions=args.interactions
    )

    print("Pipeline execution complete!")


if __name__ == "__main__":
    # If run without arguments, show usage
    if len(sys.argv) == 1:
        print("ML Optimization Pipeline")
        print("\nUsage:")
        print("  python main.py --data path/to/data.csv --target target_column")
        print("\nOptions:")
        print("  --data         Path to dataset (required)")
        print("  --target       Target column name (required)")
        print("  --interactions Number of interaction terms to create (default: 10)")
        print("  --seed         Random seed (default: 42)")
        print("\nExample:")
        print("  python main.py --data data/raw/housing.csv --target price --interactions 15")
    else:
        main()
