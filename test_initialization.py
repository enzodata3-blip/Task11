#!/usr/bin/env python3
"""
Quick initialization test to verify all modules work correctly.
"""

import sys
import os
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

print("="*80)
print("🔍 INITIALIZATION VERIFICATION TEST")
print("="*80)
print()

# Test 1: Import all modules
print("Test 1: Importing modules...")
try:
    from data_processing import DataProcessor
    from correlation_analysis import CorrelationAnalyzer
    from interaction_engineering import InteractionEngineer
    from model_training import ModelTrainer
    from evaluation import ModelEvaluator, compare_multiple_models
    from main import MLOptimizationPipeline
    print("✅ All modules imported successfully")
except Exception as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

print()

# Test 2: Create synthetic test data
print("Test 2: Creating synthetic test data...")
try:
    np.random.seed(42)
    n_samples = 100

    test_data = pd.DataFrame({
        'feature_1': np.random.randn(n_samples),
        'feature_2': np.random.randn(n_samples),
        'feature_3': np.random.randn(n_samples),
        'feature_4': np.random.randn(n_samples),
    })

    # Create target with interaction effect
    test_data['target'] = (
        2 * test_data['feature_1'] +
        3 * test_data['feature_2'] +
        1.5 * test_data['feature_1'] * test_data['feature_2'] +  # interaction
        np.random.randn(n_samples) * 0.5
    )

    print(f"✅ Created test dataset: {test_data.shape}")
except Exception as e:
    print(f"❌ Data creation error: {e}")
    sys.exit(1)

print()

# Test 3: DataProcessor
print("Test 3: Testing DataProcessor...")
try:
    processor = DataProcessor()
    processor.data = test_data.copy()
    profile = processor.generate_data_profile()
    print(f"✅ DataProcessor works - {profile['shape'][0]} rows, {profile['shape'][1]} columns")
except Exception as e:
    print(f"❌ DataProcessor error: {e}")
    sys.exit(1)

print()

# Test 4: CorrelationAnalyzer
print("Test 4: Testing CorrelationAnalyzer...")
try:
    analyzer = CorrelationAnalyzer(data=test_data, target_col='target')
    corr_matrix = analyzer.compute_correlation_matrix(method='pearson')
    target_corr = analyzer.compute_target_correlations()
    candidates = analyzer.identify_interaction_candidates(
        target_corr_threshold=0.1,
        feature_corr_range=(0.0, 0.9),
        top_n=5
    )
    print(f"✅ CorrelationAnalyzer works - {len(candidates)} candidates found")
except Exception as e:
    print(f"❌ CorrelationAnalyzer error: {e}")
    sys.exit(1)

print()

# Test 5: InteractionEngineer
print("Test 5: Testing InteractionEngineer...")
try:
    engineer = InteractionEngineer(data=test_data, target_col='target')
    interaction = engineer.create_multiplicative_interaction('feature_1', 'feature_2')
    print(f"✅ InteractionEngineer works - Created interaction: {interaction.name}")
except Exception as e:
    print(f"❌ InteractionEngineer error: {e}")
    sys.exit(1)

print()

# Test 6: ModelTrainer
print("Test 6: Testing ModelTrainer...")
try:
    trainer = ModelTrainer(
        data=test_data,
        target_col='target',
        test_size=0.2,
        random_state=42,
        scale_features=True
    )
    print(f"✅ ModelTrainer initialized - Train size: {len(trainer.X_train)}, Test size: {len(trainer.X_test)}")
except Exception as e:
    print(f"❌ ModelTrainer error: {e}")
    sys.exit(1)

print()

# Test 7: Quick model training
print("Test 7: Testing model training...")
try:
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(trainer.X_train, trainer.y_train)
    predictions = model.predict(trainer.X_test)
    from sklearn.metrics import r2_score
    r2 = r2_score(trainer.y_test, predictions)
    print(f"✅ Model training works - Test R²: {r2:.4f}")
except Exception as e:
    print(f"❌ Model training error: {e}")
    sys.exit(1)

print()

# Test 8: ModelEvaluator
print("Test 8: Testing ModelEvaluator...")
try:
    evaluator = ModelEvaluator(
        y_true=trainer.y_test.values,
        y_pred=predictions,
        model_name='Test Model'
    )
    metrics = evaluator.compute_metrics()
    print(f"✅ ModelEvaluator works - R²: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.4f}")
except Exception as e:
    print(f"❌ ModelEvaluator error: {e}")
    sys.exit(1)

print()
print("="*80)
print("✅ ALL TESTS PASSED - FRAMEWORK IS READY!")
print("="*80)
print()
print("Next steps:")
print("  1. Run: python run_full_analysis.py")
print("  2. Or explore notebooks: jupyter notebook notebooks/")
print("  3. Or use the pipeline: python src/main.py --data <your_data> --target <target_col>")
print()
