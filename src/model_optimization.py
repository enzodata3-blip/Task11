"""
Advanced Model Optimization Module
Sophisticated techniques for maximizing model performance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    cross_val_score,
    GridSearchCV,
    RandomizedSearchCV,
    learning_curve,
    validation_curve
)
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    VotingRegressor,
    StackingRegressor,
    BaggingRegressor,
    AdaBoostRegressor
)
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import uniform, randint
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')


class AdvancedModelOptimizer:
    """
    Advanced model optimization using ensemble methods, hyperparameter tuning,
    and interpretability techniques.
    """

    def __init__(self, X_train, X_test, y_train, y_test, random_state: int = 42):
        """
        Initialize optimizer.

        Parameters:
        -----------
        X_train, X_test : pd.DataFrame or np.ndarray
            Training and test features
        y_train, y_test : pd.Series or np.ndarray
            Training and test targets
        random_state : int
            Random seed
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.random_state = random_state
        self.models = {}
        self.results = {}

    def bayesian_hyperparameter_tuning(
        self,
        model_type: str = 'random_forest',
        n_iter: int = 50,
        cv: int = 5
    ) -> Dict:
        """
        Bayesian optimization for hyperparameter tuning (using RandomizedSearchCV as proxy).

        More efficient than GridSearch for large parameter spaces.

        Parameters:
        -----------
        model_type : str
            'random_forest', 'gradient_boosting', 'ridge', or 'lasso'
        n_iter : int
            Number of parameter settings sampled
        cv : int
            Cross-validation folds

        Returns:
        --------
        Dict
            Best parameters and model
        """
        print(f"\nBayesian hyperparameter optimization for {model_type}...")
        print(f"Testing {n_iter} parameter combinations with {cv}-fold CV...")

        if model_type == 'random_forest':
            model = RandomForestRegressor(random_state=self.random_state, n_jobs=-1)
            param_distributions = {
                'n_estimators': randint(50, 500),
                'max_depth': [10, 20, 30, 40, 50, None],
                'min_samples_split': randint(2, 20),
                'min_samples_leaf': randint(1, 10),
                'max_features': ['sqrt', 'log2', None],
                'bootstrap': [True, False]
            }
        elif model_type == 'gradient_boosting':
            model = GradientBoostingRegressor(random_state=self.random_state)
            param_distributions = {
                'n_estimators': randint(50, 300),
                'learning_rate': uniform(0.01, 0.29),
                'max_depth': randint(3, 10),
                'min_samples_split': randint(2, 20),
                'min_samples_leaf': randint(1, 10),
                'subsample': uniform(0.6, 0.4),
                'max_features': ['sqrt', 'log2', None]
            }
        elif model_type == 'ridge':
            model = Ridge(random_state=self.random_state)
            param_distributions = {
                'alpha': uniform(0.001, 100),
                'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sag', 'saga']
            }
        elif model_type == 'lasso':
            model = Lasso(random_state=self.random_state)
            param_distributions = {
                'alpha': uniform(0.001, 10),
                'selection': ['cyclic', 'random']
            }
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # RandomizedSearchCV
        random_search = RandomizedSearchCV(
            model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring='r2',
            n_jobs=-1,
            random_state=self.random_state,
            verbose=0
        )

        random_search.fit(self.X_train, self.y_train)

        # Evaluate best model
        best_model = random_search.best_estimator_
        y_pred_train = best_model.predict(self.X_train)
        y_pred_test = best_model.predict(self.X_test)

        results = {
            'model_name': f'Optimized {model_type}',
            'best_params': random_search.best_params_,
            'best_cv_score': random_search.best_score_,
            'model': best_model,
            'train_r2': r2_score(self.y_train, y_pred_train),
            'test_r2': r2_score(self.y_test, y_pred_test),
            'train_rmse': np.sqrt(mean_squared_error(self.y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(self.y_test, y_pred_test))
        }

        self.models[f'optimized_{model_type}'] = best_model
        self.results[f'optimized_{model_type}'] = results

        print(f"\n✓ Optimization complete:")
        print(f"  • Best CV R²: {results['best_cv_score']:.4f}")
        print(f"  • Test R²: {results['test_r2']:.4f}")
        print(f"  • Test RMSE: {results['test_rmse']:.4f}")
        print(f"\n  Best parameters:")
        for param, value in results['best_params'].items():
            print(f"    • {param}: {value}")

        return results

    def create_stacking_ensemble(
        self,
        base_models: Optional[List] = None,
        meta_model = None,
        cv: int = 5
    ) -> Dict:
        """
        Create stacking ensemble combining multiple models.

        Stacking trains a meta-model on the predictions of base models,
        often achieving better performance than any single model.

        Parameters:
        -----------
        base_models : Optional[List]
            List of (name, model) tuples for base models
        meta_model : Optional
            Meta-learner model (default: Ridge)
        cv : int
            Cross-validation folds for stacking

        Returns:
        --------
        Dict
            Stacking ensemble results
        """
        print("\nCreating stacking ensemble...")

        if base_models is None:
            base_models = [
                ('rf', RandomForestRegressor(n_estimators=100, random_state=self.random_state, n_jobs=-1)),
                ('gb', GradientBoostingRegressor(n_estimators=100, random_state=self.random_state)),
                ('ridge', Ridge(alpha=1.0)),
                ('lasso', Lasso(alpha=1.0, random_state=self.random_state))
            ]

        if meta_model is None:
            meta_model = Ridge(alpha=1.0)

        print(f"  • Base models: {len(base_models)}")
        print(f"  • Meta-learner: {meta_model.__class__.__name__}")
        print(f"  • CV folds: {cv}")

        # Create stacking regressor
        stacking = StackingRegressor(
            estimators=base_models,
            final_estimator=meta_model,
            cv=cv,
            n_jobs=-1
        )

        # Train
        stacking.fit(self.X_train, self.y_train)

        # Evaluate
        y_pred_train = stacking.predict(self.X_train)
        y_pred_test = stacking.predict(self.X_test)

        results = {
            'model_name': 'Stacking Ensemble',
            'base_models': [name for name, _ in base_models],
            'meta_model': meta_model.__class__.__name__,
            'model': stacking,
            'train_r2': r2_score(self.y_train, y_pred_train),
            'test_r2': r2_score(self.y_test, y_pred_test),
            'train_rmse': np.sqrt(mean_squared_error(self.y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(self.y_test, y_pred_test)),
            'train_mae': mean_absolute_error(self.y_train, y_pred_train),
            'test_mae': mean_absolute_error(self.y_test, y_pred_test)
        }

        self.models['stacking'] = stacking
        self.results['stacking'] = results

        print(f"\n✓ Stacking ensemble trained:")
        print(f"  • Train R²: {results['train_r2']:.4f}")
        print(f"  • Test R²: {results['test_r2']:.4f}")
        print(f"  • Test RMSE: {results['test_rmse']:.4f}")

        return results

    def create_voting_ensemble(
        self,
        models: Optional[List] = None,
        weights: Optional[List] = None
    ) -> Dict:
        """
        Create voting ensemble (averages predictions from multiple models).

        Simpler than stacking but often effective.

        Parameters:
        -----------
        models : Optional[List]
            List of (name, model) tuples
        weights : Optional[List]
            Weights for each model

        Returns:
        --------
        Dict
            Voting ensemble results
        """
        print("\nCreating voting ensemble...")

        if models is None:
            models = [
                ('rf', RandomForestRegressor(n_estimators=100, random_state=self.random_state, n_jobs=-1)),
                ('gb', GradientBoostingRegressor(n_estimators=100, random_state=self.random_state)),
                ('ridge', Ridge(alpha=1.0))
            ]

        print(f"  • Number of models: {len(models)}")
        if weights:
            print(f"  • Weights: {weights}")

        # Create voting regressor
        voting = VotingRegressor(estimators=models, weights=weights, n_jobs=-1)

        # Train
        voting.fit(self.X_train, self.y_train)

        # Evaluate
        y_pred_train = voting.predict(self.X_train)
        y_pred_test = voting.predict(self.X_test)

        results = {
            'model_name': 'Voting Ensemble',
            'models': [name for name, _ in models],
            'weights': weights,
            'model': voting,
            'train_r2': r2_score(self.y_train, y_pred_train),
            'test_r2': r2_score(self.y_test, y_pred_test),
            'train_rmse': np.sqrt(mean_squared_error(self.y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(self.y_test, y_pred_test))
        }

        self.models['voting'] = voting
        self.results['voting'] = results

        print(f"\n✓ Voting ensemble trained:")
        print(f"  • Test R²: {results['test_r2']:.4f}")
        print(f"  • Test RMSE: {results['test_rmse']:.4f}")

        return results

    def plot_learning_curves(
        self,
        model,
        model_name: str = 'Model',
        cv: int = 5,
        train_sizes: np.ndarray = None,
        save_path: Optional[str] = None
    ):
        """
        Plot learning curves to diagnose bias-variance tradeoff.

        Parameters:
        -----------
        model : sklearn estimator
            Model to evaluate
        model_name : str
            Name for plot title
        cv : int
            Cross-validation folds
        train_sizes : np.ndarray
            Training set sizes to evaluate
        save_path : Optional[str]
            Path to save figure
        """
        print(f"\nGenerating learning curves for {model_name}...")

        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)

        train_sizes_abs, train_scores, val_scores = learning_curve(
            model,
            self.X_train,
            self.y_train,
            train_sizes=train_sizes,
            cv=cv,
            scoring='r2',
            n_jobs=-1,
            random_state=self.random_state
        )

        # Calculate mean and std
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)

        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes_abs, train_mean, 'o-', color='r', label='Training score')
        plt.plot(train_sizes_abs, val_mean, 'o-', color='g', label='Cross-validation score')

        plt.fill_between(train_sizes_abs,
                        train_mean - train_std,
                        train_mean + train_std,
                        alpha=0.1, color='r')
        plt.fill_between(train_sizes_abs,
                        val_mean - val_std,
                        val_mean + val_std,
                        alpha=0.1, color='g')

        plt.xlabel('Training Set Size', fontsize=12)
        plt.ylabel('R² Score', fontsize=12)
        plt.title(f'Learning Curves: {model_name}', fontsize=14, fontweight='bold')
        plt.legend(loc='best')
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Learning curves saved to {save_path}")

        plt.show()

        # Interpretation
        final_gap = train_mean[-1] - val_mean[-1]
        if final_gap > 0.1:
            print(f"  ⚠️  High variance (overfitting): train-val gap = {final_gap:.3f}")
            print(f"     → Consider: regularization, more data, or simpler model")
        elif val_mean[-1] < 0.7:
            print(f"  ⚠️  High bias (underfitting): max CV score = {val_mean[-1]:.3f}")
            print(f"     → Consider: more features, complex model, or feature engineering")
        else:
            print(f"  ✓ Good fit: train-val gap = {final_gap:.3f}, CV score = {val_mean[-1]:.3f}")

    def compute_permutation_importance(
        self,
        model,
        model_name: str = 'Model',
        n_repeats: int = 10,
        random_state: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Compute permutation importance for model interpretability.

        Permutation importance shows the decrease in model performance when a
        feature's values are randomly shuffled.

        Parameters:
        -----------
        model : sklearn estimator
            Trained model
        model_name : str
            Model name for display
        n_repeats : int
            Number of times to permute each feature
        random_state : Optional[int]
            Random seed

        Returns:
        --------
        pd.DataFrame
            Permutation importance scores
        """
        print(f"\nComputing permutation importance for {model_name}...")
        print(f"  • Repeats: {n_repeats}")

        if random_state is None:
            random_state = self.random_state

        # Compute permutation importance
        perm_importance = permutation_importance(
            model,
            self.X_test,
            self.y_test,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=-1
        )

        # Create dataframe
        feature_names = self.X_train.columns if hasattr(self.X_train, 'columns') else \
                       [f'feature_{i}' for i in range(self.X_train.shape[1])]

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance_mean': perm_importance.importances_mean,
            'importance_std': perm_importance.importances_std
        }).sort_values('importance_mean', ascending=False)

        print(f"\n✓ Top 10 most important features:")
        for _, row in importance_df.head(10).iterrows():
            print(f"  • {row['feature']}: {row['importance_mean']:.4f} (±{row['importance_std']:.4f})")

        return importance_df

    def plot_partial_dependence(
        self,
        model,
        features: List,
        model_name: str = 'Model',
        save_path: Optional[str] = None
    ):
        """
        Plot partial dependence plots for key features.

        Shows the marginal effect of features on predictions.

        Parameters:
        -----------
        model : sklearn estimator
            Trained model
        features : List
            Feature indices or names to plot
        model_name : str
            Model name for display
        save_path : Optional[str]
            Path to save figure
        """
        print(f"\nGenerating partial dependence plots for {model_name}...")

        fig, ax = plt.subplots(figsize=(14, 4))

        display = PartialDependenceDisplay.from_estimator(
            model,
            self.X_train,
            features=features,
            ax=ax,
            n_jobs=-1,
            random_state=self.random_state
        )

        plt.suptitle(f'Partial Dependence Plots: {model_name}',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Partial dependence plots saved to {save_path}")

        plt.show()

    def compare_all_models(self) -> pd.DataFrame:
        """
        Compare performance of all trained models.

        Returns:
        --------
        pd.DataFrame
            Comparison table
        """
        if not self.results:
            print("⚠️  No models trained yet!")
            return pd.DataFrame()

        comparison = []
        for name, results in self.results.items():
            comparison.append({
                'Model': results['model_name'],
                'Train_R2': results['train_r2'],
                'Test_R2': results['test_r2'],
                'Train_RMSE': results['train_rmse'],
                'Test_RMSE': results['test_rmse'],
                'Overfitting': results['train_r2'] - results['test_r2']
            })

        df = pd.DataFrame(comparison).sort_values('Test_R2', ascending=False)

        print("\n" + "="*90)
        print("MODEL COMPARISON - ADVANCED OPTIMIZATION")
        print("="*90)
        print(df.to_string(index=False))
        print("="*90 + "\n")

        return df

    def generate_optimization_report(self) -> Dict:
        """
        Generate comprehensive optimization report.

        Returns:
        --------
        Dict
            Complete optimization summary
        """
        print("\n" + "="*80)
        print("ADVANCED MODEL OPTIMIZATION REPORT")
        print("="*80 + "\n")

        report = {
            'num_models_trained': len(self.models),
            'models': list(self.models.keys()),
            'results': self.results,
            'comparison': self.compare_all_models()
        }

        if not report['comparison'].empty:
            best_model_row = report['comparison'].iloc[0]
            print(f"🏆 Best Model: {best_model_row['Model']}")
            print(f"   Test R²: {best_model_row['Test_R2']:.4f}")
            print(f"   Test RMSE: {best_model_row['Test_RMSE']:.4f}")
            print(f"   Overfitting Gap: {best_model_row['Overfitting']:.4f}")

        print("\n" + "="*80 + "\n")

        return report


if __name__ == "__main__":
    print("Advanced Model Optimization Module")
    print("="*80)
    print("\nThis module provides:")
    print("  • Bayesian hyperparameter optimization")
    print("  • Stacking ensembles")
    print("  • Voting ensembles")
    print("  • Learning curve analysis")
    print("  • Permutation importance")
    print("  • Partial dependence plots")
    print("\nUsage example:")
    print("""
    from model_optimization import AdvancedModelOptimizer

    optimizer = AdvancedModelOptimizer(X_train, X_test, y_train, y_test)

    # Bayesian optimization
    rf_optimized = optimizer.bayesian_hyperparameter_tuning('random_forest', n_iter=50)

    # Create ensembles
    stacking = optimizer.create_stacking_ensemble()
    voting = optimizer.create_voting_ensemble()

    # Analyze
    optimizer.plot_learning_curves(model, 'My Model')
    importance = optimizer.compute_permutation_importance(model)

    # Compare all models
    comparison = optimizer.compare_all_models()
    """)
