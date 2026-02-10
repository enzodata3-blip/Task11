#!/usr/bin/env python3
"""
Complete ML Optimization Pipeline Execution
Generates data, runs analysis, optimizes models, and saves all results.
"""

import sys
import os
sys.path.insert(0, 'src')
os.makedirs('results', exist_ok=True)
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)
os.makedirs('models', exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from data_processing import DataProcessor
from correlation_analysis import CorrelationAnalyzer
from interaction_engineering import InteractionEngineer
from model_training import ModelTrainer
from evaluation import ModelEvaluator, compare_multiple_models

# Set random seed
np.random.seed(42)
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('seaborn-whitegrid')
sns.set_palette('husl')

print("="*80)
print("🚀 ML OPTIMIZATION PIPELINE - FULL EXECUTION")
print("="*80)
print()

# ============================================================================
# STEP 1: Generate Synthetic Data with Known Interactions
# ============================================================================
print("STEP 1: Generating Synthetic Housing Data...")
print("-" * 80)

def generate_housing_data(n_samples=2000):
    """Generate synthetic housing data with interaction effects."""
    data = {
        'area': np.random.randint(800, 4000, n_samples),
        'bedrooms': np.random.randint(1, 6, n_samples),
        'bathrooms': np.random.randint(1, 4, n_samples),
        'age': np.random.randint(0, 50, n_samples),
        'garage': np.random.randint(0, 4, n_samples),
        'lot_size': np.random.randint(2000, 15000, n_samples),
        'stories': np.random.randint(1, 4, n_samples),
        'neighborhood_score': np.random.randint(1, 11, n_samples),
    }

    df = pd.DataFrame(data)

    # Price with LINEAR + INTERACTION effects
    price = (
        100000 +
        df['area'] * 150 +
        df['bedrooms'] * 10000 +
        df['bathrooms'] * 15000 +
        df['age'] * -2000 +
        df['garage'] * 8000 +
        df['lot_size'] * 5 +
        df['stories'] * 12000 +
        df['neighborhood_score'] * 20000 +
        # INTERACTION EFFECTS:
        df['area'] * df['neighborhood_score'] * 30 +
        df['bedrooms'] * df['bathrooms'] * 5000 +
        df['area'] * df['age'] * -0.5
    )

    price = price + np.random.normal(0, 50000, n_samples)
    price = np.maximum(price, 50000)
    df['price'] = price

    return df

# Generate data
housing_data = generate_housing_data(n_samples=2000)
housing_data.to_csv('data/raw/housing_data.csv', index=False)

print(f"✅ Generated {len(housing_data):,} samples")
print(f"   Features: {len(housing_data.columns)-1}")
print(f"   Target: price")
print(f"   Saved to: data/raw/housing_data.csv")
print()

# ============================================================================
# STEP 2: Correlation Analysis
# ============================================================================
print("STEP 2: Correlation Analysis...")
print("-" * 80)

analyzer = CorrelationAnalyzer(data=housing_data, target_col='price')
corr_matrix = analyzer.compute_correlation_matrix(method='pearson')
target_corr = analyzer.compute_target_correlations(method='pearson')

# Generate visualizations
analyzer.plot_correlation_heatmap(
    figsize=(12, 10),
    save_path='results/correlation_heatmap.png'
)
plt.close()

analyzer.plot_target_correlations(
    top_n=10,
    save_path='results/target_correlations.png'
)
plt.close()

# Identify interaction candidates
interaction_candidates = analyzer.identify_interaction_candidates(
    target_corr_threshold=0.1,
    feature_corr_range=(0.05, 0.7),
    top_n=20
)

print(f"✅ Top 10 correlations with price:")
for _, row in target_corr.head(10).iterrows():
    print(f"   {row['feature']:20s} → {row['correlation']:+.3f}")

print(f"\n✅ Identified {len(interaction_candidates)} interaction candidates")
print(f"   Saved visualizations to results/")
print()

# ============================================================================
# STEP 3: Interaction Engineering
# ============================================================================
print("STEP 3: Interaction Engineering...")
print("-" * 80)

engineer = InteractionEngineer(data=housing_data, target_col='price')

# Create interactions from top candidates
top_n = 15
interaction_pairs = [
    (row['feature_1'], row['feature_2'])
    for _, row in interaction_candidates.head(top_n).iterrows()
]

print(f"Creating {len(interaction_pairs)} interaction terms...")
interactions = engineer.batch_create_interactions(
    interaction_pairs,
    interaction_type='multiplicative'
)

# Evaluate importance
print("Evaluating interaction importance (5-fold CV)...")
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

importance = engineer.evaluate_interaction_importance(
    interactions,
    estimator=model,
    cv=5,
    scoring='r2'
)

# Visualize
plt.figure(figsize=(12, 7))
colors = ['green' if x > 0 else 'red' for x in importance['improvement']]
plt.barh(importance['interaction_term'], importance['improvement'], color=colors, alpha=0.7)
plt.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
plt.xlabel('R² Improvement', fontsize=12, fontweight='bold')
plt.ylabel('Interaction Term', fontsize=12)
plt.title('Interaction Terms by Performance Impact', fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('results/interaction_importance.png', dpi=300, bbox_inches='tight')
plt.close()

# Select best interactions
best_interactions = engineer.select_best_interactions(
    importance,
    threshold=0.0,
    top_n=None
)

print(f"\n✅ Selected {len(best_interactions)} beneficial interactions")
print(f"   Saved visualization to results/interaction_importance.png")

# Create enhanced dataset
enhanced_data = engineer.add_interactions_to_data(interactions[best_interactions])
enhanced_data.to_csv('data/processed/enhanced_housing_data.csv', index=False)

print(f"✅ Enhanced dataset saved: {enhanced_data.shape}")
print()

# ============================================================================
# STEP 4: Model Training with Hyperparameter Optimization
# ============================================================================
print("STEP 4: Model Training & Hyperparameter Optimization...")
print("-" * 80)

trainer = ModelTrainer(
    data=housing_data,
    target_col='price',
    test_size=0.2,
    random_state=42,
    scale_features=True
)

# Train baseline models
print("Training baseline models...")
baseline_results = trainer.train_baseline_models(cv=5)

# Train enhanced model with basic RF
print("\nTraining enhanced model (base parameters)...")
enhanced_results_basic = trainer.train_enhanced_model(
    enhanced_data=enhanced_data,
    model_name='Enhanced RF (base)',
    cv=5
)

# Hyperparameter optimization for enhanced model
print("\nOptimizing hyperparameters for enhanced model...")
print("(This may take 2-3 minutes...)")

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [15, 20, 25, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

# Prepare data
feature_cols_enhanced = [col for col in enhanced_data.columns if col != 'price']
X_enhanced = enhanced_data[feature_cols_enhanced]
y_enhanced = enhanced_data['price']

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train_enh, X_test_enh, y_train_enh, y_test_enh = train_test_split(
    X_enhanced, y_enhanced, test_size=0.2, random_state=42
)

scaler_enh = StandardScaler()
X_train_enh_scaled = scaler_enh.fit_transform(X_train_enh)
X_test_enh_scaled = scaler_enh.transform(X_test_enh)

# Grid search
rf_optimized = RandomForestRegressor(random_state=42, n_jobs=-1)
grid_search = GridSearchCV(
    rf_optimized,
    param_grid,
    cv=3,
    scoring='r2',
    n_jobs=-1,
    verbose=0
)

grid_search.fit(X_train_enh_scaled, y_train_enh)

print(f"\n✅ Best parameters found:")
for param, value in grid_search.best_params_.items():
    print(f"   {param}: {value}")

# Train final optimized model
best_model = grid_search.best_estimator_
y_train_pred_opt = best_model.predict(X_train_enh_scaled)
y_test_pred_opt = best_model.predict(X_test_enh_scaled)

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

optimized_results = {
    'model': best_model,
    'model_name': 'Enhanced RF (optimized)',
    'num_features': len(feature_cols_enhanced),
    'train_r2': r2_score(y_train_enh, y_train_pred_opt),
    'test_r2': r2_score(y_test_enh, y_test_pred_opt),
    'train_rmse': np.sqrt(mean_squared_error(y_train_enh, y_train_pred_opt)),
    'test_rmse': np.sqrt(mean_squared_error(y_test_enh, y_test_pred_opt)),
    'train_mae': mean_absolute_error(y_train_enh, y_train_pred_opt),
    'test_mae': mean_absolute_error(y_test_enh, y_test_pred_opt),
    'predictions_test': y_test_pred_opt,
    'y_test': y_test_enh,
    'X_test': X_test_enh_scaled,
    'best_params': grid_search.best_params_
}

print(f"\n✅ Optimized Model Performance:")
print(f"   Test R²:   {optimized_results['test_r2']:.4f}")
print(f"   Test RMSE: ${optimized_results['test_rmse']:,.2f}")
print(f"   Test MAE:  ${optimized_results['test_mae']:,.2f}")

# Save optimized model
import joblib
joblib.dump(best_model, 'models/optimized_model.joblib')
print(f"\n✅ Model saved to models/optimized_model.joblib")
print()

# ============================================================================
# STEP 5: Model Comparison
# ============================================================================
print("STEP 5: Model Comparison...")
print("-" * 80)

# Create comparison dataframe
comparison_data = []

for name, results in baseline_results.items():
    comparison_data.append({
        'Model': f"Baseline - {name}",
        'Test_R2': results['test_r2'],
        'Test_RMSE': results['test_rmse'],
        'Test_MAE': results['test_mae'],
        'Features': len(trainer.feature_cols),
        'Type': 'Baseline'
    })

comparison_data.append({
    'Model': 'Enhanced RF (base)',
    'Test_R2': enhanced_results_basic['test_r2'],
    'Test_RMSE': enhanced_results_basic['test_rmse'],
    'Test_MAE': enhanced_results_basic['test_mae'],
    'Features': enhanced_results_basic['num_features'],
    'Type': 'Enhanced'
})

comparison_data.append({
    'Model': 'Enhanced RF (optimized)',
    'Test_R2': optimized_results['test_r2'],
    'Test_RMSE': optimized_results['test_rmse'],
    'Test_MAE': optimized_results['test_mae'],
    'Features': optimized_results['num_features'],
    'Type': 'Enhanced + Optimized'
})

comparison_df = pd.DataFrame(comparison_data).sort_values('Test_R2', ascending=False)
comparison_df.to_csv('results/model_comparison.csv', index=False)

print("\n" + "="*90)
print("MODEL COMPARISON")
print("="*90)
print(comparison_df.to_string(index=False))
print("="*90)

best_model_row = comparison_df.iloc[0]
baseline_best = comparison_df[comparison_df['Type'] == 'Baseline'].iloc[0]
improvement = best_model_row['Test_R2'] - baseline_best['Test_R2']
improvement_pct = (improvement / baseline_best['Test_R2']) * 100

print(f"\n🏆 BEST MODEL: {best_model_row['Model']}")
print(f"   Test R²:        {best_model_row['Test_R2']:.4f}")
print(f"   Improvement:    {improvement:+.4f} ({improvement_pct:+.2f}%)")
print(f"   Features:       {best_model_row['Features']}")
print()

# ============================================================================
# STEP 6: Comprehensive Evaluation
# ============================================================================
print("STEP 6: Model Evaluation & Visualization...")
print("-" * 80)

evaluator = ModelEvaluator(
    y_true=optimized_results['y_test'],
    y_pred=optimized_results['predictions_test'],
    model_name='Optimized Enhanced RF'
)

# Generate all visualizations
evaluator.plot_predictions(save_path='results/final_predictions.png')
plt.close()

evaluator.plot_residuals(save_path='results/final_residuals.png')
plt.close()

evaluator.plot_error_distribution(save_path='results/final_errors.png')
plt.close()

metrics = evaluator.compute_metrics()
print(f"✅ Final Model Metrics:")
print(f"   R²:                {metrics['r2']:.4f}")
print(f"   Adjusted R²:       {metrics['adjusted_r2']:.4f}")
print(f"   RMSE:              ${metrics['rmse']:,.2f}")
print(f"   MAE:               ${metrics['mae']:,.2f}")
print(f"   MAPE:              {metrics['mape']:.2f}%")
print(f"   Max Error:         ${metrics['max_error']:,.2f}")
print(f"\n✅ All visualizations saved to results/")
print()

# ============================================================================
# STEP 7: Feature Importance
# ============================================================================
print("STEP 7: Feature Importance Analysis...")
print("-" * 80)

feature_importance_df = pd.DataFrame({
    'feature': X_enhanced.columns,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

feature_importance_df.to_csv('results/feature_importance.csv', index=False)

print(f"✅ Top 15 Most Important Features:\n")
for i, row in feature_importance_df.head(15).iterrows():
    marker = "🔗" if "×" in row['feature'] else "📊"
    print(f"   {marker} {row['feature']:30s} {row['importance']:.4f}")

# Visualize
plt.figure(figsize=(12, 8))
top_features = feature_importance_df.head(20)
colors = ['red' if '×' in feat else 'steelblue' for feat in top_features['feature']]
plt.barh(top_features['feature'], top_features['importance'], color=colors, alpha=0.7)
plt.xlabel('Feature Importance', fontsize=12, fontweight='bold')
plt.ylabel('Feature', fontsize=12)
plt.title('Top 20 Feature Importances (Red = Interactions)', fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('results/feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

interaction_count_top10 = sum(1 for feat in feature_importance_df.head(10)['feature'] if '×' in feat)
interaction_count_top20 = sum(1 for feat in feature_importance_df.head(20)['feature'] if '×' in feat)

print(f"\n✅ Interaction terms in top 10: {interaction_count_top10}")
print(f"✅ Interaction terms in top 20: {interaction_count_top20}")
print(f"✅ Feature importance saved to results/feature_importance.csv")
print()

# ============================================================================
# STEP 8: Generate Summary Report
# ============================================================================
print("STEP 8: Generating Summary Report...")
print("-" * 80)

summary = {
    'Pipeline Execution': 'Complete',
    'Total Samples': len(housing_data),
    'Training Samples': len(X_train_enh),
    'Test Samples': len(X_test_enh),
    'Original Features': len(housing_data.columns) - 1,
    'Enhanced Features': len(feature_cols_enhanced),
    'Interactions Added': len(best_interactions),
    'Baseline Best R2': baseline_best['Test_R2'],
    'Enhanced Base R2': enhanced_results_basic['test_r2'],
    'Enhanced Optimized R2': optimized_results['test_r2'],
    'Final Improvement': f"{improvement_pct:+.2f}%",
    'Best Model': best_model_row['Model'],
    'Interactions in Top 10': interaction_count_top10,
    'Interactions in Top 20': interaction_count_top20
}

summary_df = pd.DataFrame([summary]).T
summary_df.columns = ['Value']
summary_df.to_csv('results/pipeline_summary.csv')

print("\n" + "="*80)
print("PIPELINE EXECUTION SUMMARY")
print("="*80)
print(summary_df.to_string())
print("="*80)
print()

# ============================================================================
# COMPLETE
# ============================================================================
print("\n" + "="*80)
print("✅ PIPELINE EXECUTION COMPLETE!")
print("="*80)
print("\nAll outputs saved to:")
print("  📊 results/           - Visualizations and metrics")
print("  💾 data/processed/    - Enhanced dataset")
print("  🤖 models/            - Trained optimized model")
print("\nNext step:")
print("  📓 Open 'notebooks/RESULTS_NOTEBOOK.ipynb' to view comprehensive report")
print("="*80)
print()
