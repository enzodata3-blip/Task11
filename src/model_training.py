"""
Model Training Module
Train baseline and interaction-enhanced models with comprehensive evaluation.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)
from typing import Dict, List, Tuple, Optional, Any
import joblib
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """
    Train and compare baseline vs interaction-enhanced models.

    Emphasis on statistical rigor and reproducibility (inspired by tidymodels).
    """

    def __init__(
        self,
        data: pd.DataFrame,
        target_col: str,
        test_size: float = 0.2,
        random_state: int = 42,
        scale_features: bool = True
    ):
        """
        Initialize the model trainer.

        Parameters:
        -----------
        data : pd.DataFrame
            Complete dataset
        target_col : str
            Target variable column name
        test_size : float
            Proportion of data for testing
        random_state : int
            Random seed for reproducibility
        scale_features : bool
            Whether to scale features
        """
        self.data = data.copy()
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state
        self.scale_features = scale_features

        self.feature_cols = [col for col in data.columns if col != target_col]
        self.scaler = StandardScaler() if scale_features else None

        # Train/test split
        self.X_train, self.X_test, self.y_train, self.y_test = self._prepare_data()

        # Model storage
        self.baseline_results = {}
        self.enhanced_results = {}
        self.best_model = None

    def _prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Prepare train/test split."""
        X = self.data[self.feature_cols]
        y = self.data[self.target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state
        )

        print(f"Data split: {len(X_train)} training samples, {len(X_test)} test samples")

        # Scale if requested
        if self.scale_features:
            X_train = pd.DataFrame(
                self.scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
            X_test = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
            print("Features scaled using StandardScaler")

        return X_train, X_test, y_train, y_test

    def train_baseline_models(
        self,
        models: Optional[Dict[str, Any]] = None,
        cv: int = 5
    ) -> Dict:
        """
        Train multiple baseline models for comparison.

        Parameters:
        -----------
        models : Optional[Dict[str, Any]]
            Dictionary of model names and instances
        cv : int
            Cross-validation folds

        Returns:
        --------
        Dict
            Results for all baseline models
        """
        if models is None:
            models = {
                'Linear Regression': LinearRegression(),
                'Ridge': Ridge(alpha=1.0),
                'Lasso': Lasso(alpha=1.0),
                'Random Forest': RandomForestRegressor(
                    n_estimators=100,
                    random_state=self.random_state,
                    n_jobs=-1
                ),
                'Gradient Boosting': GradientBoostingRegressor(
                    n_estimators=100,
                    random_state=self.random_state
                )
            }

        print("\n" + "=" * 70)
        print("TRAINING BASELINE MODELS")
        print("=" * 70)

        results = {}

        for name, model in models.items():
            print(f"\nTraining {name}...")

            # Cross-validation
            cv_results = cross_validate(
                model,
                self.X_train,
                self.y_train,
                cv=cv,
                scoring=['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'],
                return_train_score=True
            )

            # Train on full training set
            model.fit(self.X_train, self.y_train)

            # Predictions
            y_train_pred = model.predict(self.X_train)
            y_test_pred = model.predict(self.X_test)

            # Metrics
            results[name] = {
                'model': model,
                'cv_r2_mean': cv_results['test_r2'].mean(),
                'cv_r2_std': cv_results['test_r2'].std(),
                'train_r2': r2_score(self.y_train, y_train_pred),
                'test_r2': r2_score(self.y_test, y_test_pred),
                'train_rmse': np.sqrt(mean_squared_error(self.y_train, y_train_pred)),
                'test_rmse': np.sqrt(mean_squared_error(self.y_test, y_test_pred)),
                'train_mae': mean_absolute_error(self.y_train, y_train_pred),
                'test_mae': mean_absolute_error(self.y_test, y_test_pred),
                'predictions_train': y_train_pred,
                'predictions_test': y_test_pred
            }

            print(f"  CV R²: {results[name]['cv_r2_mean']:.4f} (±{results[name]['cv_r2_std']:.4f})")
            print(f"  Test R²: {results[name]['test_r2']:.4f}")
            print(f"  Test RMSE: {results[name]['test_rmse']:.4f}")

        self.baseline_results = results
        return results

    def train_enhanced_model(
        self,
        enhanced_data: pd.DataFrame,
        model_name: str = 'Random Forest',
        model: Optional[Any] = None,
        cv: int = 5
    ) -> Dict:
        """
        Train model with interaction-enhanced features.

        Parameters:
        -----------
        enhanced_data : pd.DataFrame
            Dataset with interaction terms
        model_name : str
            Name for this model
        model : Optional[Any]
            Model instance (default: RandomForestRegressor)
        cv : int
            Cross-validation folds

        Returns:
        --------
        Dict
            Enhanced model results
        """
        if model is None:
            model = RandomForestRegressor(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            )

        print("\n" + "=" * 70)
        print(f"TRAINING ENHANCED MODEL: {model_name}")
        print("=" * 70)

        # Prepare enhanced data
        feature_cols_enhanced = [col for col in enhanced_data.columns if col != self.target_col]
        X = enhanced_data[feature_cols_enhanced]
        y = enhanced_data[self.target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state
        )

        # Scale if needed
        if self.scale_features:
            scaler = StandardScaler()
            X_train = pd.DataFrame(
                scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
            X_test = pd.DataFrame(
                scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )

        print(f"Enhanced features: {len(feature_cols_enhanced)} "
              f"(+{len(feature_cols_enhanced) - len(self.feature_cols)} interactions)")

        # Cross-validation
        cv_results = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'],
            return_train_score=True
        )

        # Train on full training set
        model.fit(X_train, y_train)

        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Metrics
        results = {
            'model': model,
            'model_name': model_name,
            'num_features': len(feature_cols_enhanced),
            'num_interactions': len(feature_cols_enhanced) - len(self.feature_cols),
            'cv_r2_mean': cv_results['test_r2'].mean(),
            'cv_r2_std': cv_results['test_r2'].std(),
            'train_r2': r2_score(y_train, y_train_pred),
            'test_r2': r2_score(y_test, y_test_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'test_mae': mean_absolute_error(y_test, y_test_pred),
            'predictions_train': y_train_pred,
            'predictions_test': y_test_pred,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }

        print(f"  CV R²: {results['cv_r2_mean']:.4f} (±{results['cv_r2_std']:.4f})")
        print(f"  Test R²: {results['test_r2']:.4f}")
        print(f"  Test RMSE: {results['test_rmse']:.4f}")

        self.enhanced_results[model_name] = results
        return results

    def compare_models(self) -> pd.DataFrame:
        """
        Compare baseline and enhanced models.

        Returns:
        --------
        pd.DataFrame
            Comparison table
        """
        comparison = []

        # Baseline models
        for name, results in self.baseline_results.items():
            comparison.append({
                'Model': f"Baseline - {name}",
                'CV_R2': results['cv_r2_mean'],
                'CV_R2_Std': results['cv_r2_std'],
                'Test_R2': results['test_r2'],
                'Test_RMSE': results['test_rmse'],
                'Test_MAE': results['test_mae'],
                'Num_Features': len(self.feature_cols),
                'Type': 'Baseline'
            })

        # Enhanced models
        for name, results in self.enhanced_results.items():
            comparison.append({
                'Model': f"Enhanced - {name}",
                'CV_R2': results['cv_r2_mean'],
                'CV_R2_Std': results['cv_r2_std'],
                'Test_R2': results['test_r2'],
                'Test_RMSE': results['test_rmse'],
                'Test_MAE': results['test_mae'],
                'Num_Features': results['num_features'],
                'Type': 'Enhanced'
            })

        df = pd.DataFrame(comparison).sort_values('Test_R2', ascending=False)
        return df

    def print_comparison(self):
        """Print formatted model comparison."""
        comparison = self.compare_models()

        print("\n" + "=" * 100)
        print("MODEL COMPARISON")
        print("=" * 100)
        print(comparison.to_string(index=False))
        print("=" * 100 + "\n")

        # Best model
        best = comparison.iloc[0]
        print(f"🏆 Best Model: {best['Model']}")
        print(f"   Test R²: {best['Test_R2']:.4f}")
        print(f"   Test RMSE: {best['Test_RMSE']:.4f}")
        print(f"   Features: {best['Num_Features']}\n")

    def get_feature_importance(self, model_name: str = None) -> pd.DataFrame:
        """
        Get feature importance from tree-based models.

        Parameters:
        -----------
        model_name : str
            Name of the model (default: best enhanced model)

        Returns:
        --------
        pd.DataFrame
            Feature importance ranking
        """
        if model_name is None:
            if self.enhanced_results:
                model_name = list(self.enhanced_results.keys())[0]
            else:
                model_name = list(self.baseline_results.keys())[0]

        # Get model
        if model_name in self.enhanced_results:
            results = self.enhanced_results[model_name]
            feature_names = results['X_train'].columns
        else:
            results = self.baseline_results[model_name]
            feature_names = self.feature_cols

        model = results['model']

        # Check if model has feature_importances_
        if not hasattr(model, 'feature_importances_'):
            print(f"Model {model_name} does not support feature importance")
            return None

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        return importance_df

    def save_model(self, model_name: str, filepath: str):
        """
        Save trained model to disk.

        Parameters:
        -----------
        model_name : str
            Name of model to save
        filepath : str
            Path to save the model
        """
        if model_name in self.enhanced_results:
            model = self.enhanced_results[model_name]['model']
        elif model_name in self.baseline_results:
            model = self.baseline_results[model_name]['model']
        else:
            raise ValueError(f"Model '{model_name}' not found")

        joblib.dump(model, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load model from disk."""
        model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        return model


if __name__ == "__main__":
    # Example usage
    print("Model Training Module")
    print("Train and compare baseline vs interaction-enhanced models.")
    print("\nUsage example:")
    print("""
    from model_training import ModelTrainer

    # Initialize trainer
    trainer = ModelTrainer(data=df, target_col='target', test_size=0.2)

    # Train baseline models
    baseline_results = trainer.train_baseline_models()

    # Train enhanced model with interactions
    enhanced_data = df_with_interactions  # From InteractionEngineer
    enhanced_results = trainer.train_enhanced_model(enhanced_data)

    # Compare models
    trainer.print_comparison()

    # Get feature importance
    importance = trainer.get_feature_importance()

    # Save best model
    trainer.save_model('Enhanced - Random Forest', 'models/best_model.joblib')
    """)
