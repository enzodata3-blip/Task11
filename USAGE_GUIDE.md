# ML Optimization Framework - Usage Guide

## Overview

This framework implements a systematic approach to machine learning model optimization through human-guided interaction term engineering. It bridges the gap between automated ML and domain expertise by using correlation analysis to identify promising feature interactions.

## Philosophy: The Human Element

Machine learning models often reach a plateau without human guidance. This framework embodies the "human element" in ML optimization:

1. **Statistical Analysis**: Use correlation matrices to understand feature relationships
2. **Informed Feature Engineering**: Create interaction terms based on statistical insights
3. **Iterative Refinement**: Evaluate and select only beneficial interactions
4. **Interpretable Models**: Maintain model interpretability through careful feature selection

## Quick Start

### 1. Installation

```bash
# Clone or navigate to the project directory
cd model_a

# Install dependencies
pip install -r requirements.txt

# Or use a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Prepare Your Data

Place your dataset in the `data/raw/` directory. Supported formats:
- CSV (`.csv`)
- Excel (`.xlsx`, `.xls`)
- Parquet (`.parquet`)
- JSON (`.json`)

### 3. Run the Pipeline

#### Option A: Command Line (Automated)

```bash
python src/main.py --data data/raw/your_dataset.csv --target target_column --interactions 10
```

**Parameters**:
- `--data`: Path to your dataset (required)
- `--target`: Name of the target variable column (required)
- `--interactions`: Number of interaction terms to create (default: 10)
- `--seed`: Random seed for reproducibility (default: 42)

#### Option B: Jupyter Notebook (Interactive)

```bash
jupyter notebook notebooks/01_exploratory_analysis.ipynb
```

Update the data path and target column in the notebook, then run all cells.

#### Option C: Python Script (Custom)

```python
from src import MLOptimizationPipeline

pipeline = MLOptimizationPipeline(
    data_path='data/raw/housing.csv',
    target_col='price',
    random_state=42
)

results = pipeline.run_full_pipeline(
    missing_strategy='median',
    handle_outliers=True,
    correlation_method='pearson',
    top_n_interactions=15
)
```

## Workflow Steps

### Step 1: Data Loading & Preprocessing

The framework automatically:
- Loads data from various formats
- Generates a comprehensive data profile
- Handles missing values (drop, mean, median, mode, constant)
- Detects and removes outliers (IQR or Z-score methods)
- Encodes categorical variables (one-hot or label encoding)

**Customization**:
```python
processor = DataProcessor()
data = processor.load_data('data/raw/data.csv')
processor.print_data_profile()

# Custom preprocessing
data = processor.handle_missing_values(strategy='median')
data = processor.handle_outliers(method='iqr', threshold=1.5)
data = processor.encode_categorical_variables(method='onehot')
```

### Step 2: Correlation Analysis

**Purpose**: Understand feature relationships and identify interaction candidates.

The analyzer:
- Computes correlation matrices (Pearson, Spearman, or Kendall)
- Identifies feature-target correlations
- Detects multicollinearity issues
- Suggests promising feature pairs for interaction terms

**Heuristic for Interaction Candidates**:
- Both features have moderate-to-strong correlation with target (|r| > 0.2)
- Features have moderate inter-correlation (0.1 < |r| < 0.7)
  - Not too low: features should be somewhat related
  - Not too high: avoid redundancy/multicollinearity

**Customization**:
```python
analyzer = CorrelationAnalyzer(data=df, target_col='target')

# Compute correlations
analyzer.compute_correlation_matrix(method='pearson')
analyzer.compute_target_correlations()

# Identify candidates
candidates = analyzer.identify_interaction_candidates(
    target_corr_threshold=0.2,
    feature_corr_range=(0.1, 0.7),
    top_n=20
)

# Visualizations
analyzer.plot_correlation_heatmap(save_path='results/heatmap.png')
analyzer.plot_target_correlations(top_n=20)
analyzer.print_report()
```

### Step 3: Interaction Engineering

**Purpose**: Create and evaluate interaction terms.

Supported interaction types:
1. **Multiplicative**: f1 × f2 (captures joint effects)
2. **Polynomial**: f^n (captures non-linear relationships)
3. **Ratio**: f1 / f2 (captures relative relationships)
4. **Difference**: f1 - f2 (captures contrasts)
5. **Logarithmic**: log(f) (captures exponential relationships)

**Evaluation Process**:
- Train baseline model without interactions
- Add each interaction individually and measure improvement
- Select interactions that improve cross-validation performance
- Create enhanced dataset with beneficial interactions only

**Customization**:
```python
engineer = InteractionEngineer(data=df, target_col='target')

# Create specific interactions
int1 = engineer.create_multiplicative_interaction('age', 'income')
int2 = engineer.create_polynomial_interaction('distance', degree=2)
int3 = engineer.create_ratio_interaction('price', 'area')

# Batch create from candidates
interaction_pairs = [('feat1', 'feat2'), ('feat3', 'feat4')]
interactions = engineer.batch_create_interactions(
    interaction_pairs,
    interaction_type='multiplicative'
)

# Evaluate importance
from sklearn.ensemble import RandomForestRegressor
importance = engineer.evaluate_interaction_importance(
    interactions,
    estimator=RandomForestRegressor(),
    cv=5,
    scoring='r2'
)

# Select best
best = engineer.select_best_interactions(importance, threshold=0.0)
enhanced_df = engineer.add_interactions_to_data(interactions[best])
```

### Step 4: Model Training

**Purpose**: Train and compare baseline vs enhanced models.

The framework trains multiple baseline models:
- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest
- Gradient Boosting

Then trains enhanced versions with interaction terms.

**Customization**:
```python
trainer = ModelTrainer(
    data=df,
    target_col='target',
    test_size=0.2,
    random_state=42,
    scale_features=True
)

# Train baseline models
baseline_results = trainer.train_baseline_models(cv=5)

# Train enhanced model
enhanced_results = trainer.train_enhanced_model(
    enhanced_data=enhanced_df,
    model_name='Enhanced Random Forest',
    cv=5
)

# Compare
trainer.print_comparison()

# Save best model
trainer.save_model('Enhanced Random Forest', 'models/best_model.joblib')
```

### Step 5: Model Evaluation

**Purpose**: Comprehensive evaluation with statistical rigor.

Evaluation includes:
- **Performance Metrics**: R², Adjusted R², RMSE, MAE, MAPE
- **Residual Analysis**: Normality tests, autocorrelation, homoscedasticity
- **Visualizations**: Predicted vs actual, residual plots, Q-Q plots, error distributions

**Customization**:
```python
evaluator = ModelEvaluator(
    y_true=y_test,
    y_pred=predictions,
    model_name='Random Forest'
)

# Print comprehensive report
evaluator.print_evaluation_report()

# Generate visualizations
evaluator.plot_predictions(save_path='results/predictions.png')
evaluator.plot_residuals(save_path='results/residuals.png')
evaluator.plot_error_distribution(save_path='results/errors.png')

# Compare multiple models
comparison = compare_multiple_models([evaluator1, evaluator2, evaluator3])
```

### Step 6: Feature Importance Analysis

**Purpose**: Understand which features (including interactions) drive predictions.

The framework extracts feature importance from tree-based models and ranks:
- Original features
- Interaction terms
- Their relative contributions

This helps validate that interaction terms are meaningful and interpretable.

**Customization**:
```python
# Get feature importance
importance = trainer.get_feature_importance('Enhanced Random Forest')

# Identify top interaction terms
interaction_features = importance[importance['feature'].str.contains('×')]
print(interaction_features.head(10))
```

## Interpreting Results

### 1. Correlation Analysis

**What to look for**:
- Features with strong correlation to target (|r| > 0.3) are valuable
- High inter-feature correlation (|r| > 0.8) indicates multicollinearity
- Interaction candidates with scores > 0.1 are promising

**Action items**:
- Remove one feature from highly correlated pairs
- Prioritize interaction candidates with high scores
- Consider domain knowledge when interpreting correlations

### 2. Interaction Importance

**What to look for**:
- Positive improvement values indicate beneficial interactions
- Improvement > 0.01 (1%) is typically meaningful
- Compare to baseline performance

**Action items**:
- Select interactions with positive improvement
- Remove interactions that hurt performance
- Balance model complexity with performance gain

### 3. Model Comparison

**What to look for**:
- Enhanced model should outperform baseline
- Check for overfitting (train R² >> test R²)
- Verify cross-validation scores are consistent

**Action items**:
- If no improvement: try different interaction types or features
- If overfitting: reduce number of interactions or add regularization
- Document which interactions provide the most value

### 4. Residual Analysis

**What to look for**:
- Residuals should be normally distributed (Shapiro-Wilk p > 0.05)
- No patterns in residual plots (randomness indicates good fit)
- Durbin-Watson statistic ~2 (no autocorrelation)

**Action items**:
- Non-normal residuals: try log transformation of target
- Patterns in residuals: consider additional interactions or features
- Heteroscedasticity: try different model or transformation

## Advanced Usage

### Custom Interaction Types

```python
# Create custom interaction function
def custom_interaction(df, feat1, feat2):
    """Custom interaction: geometric mean"""
    return np.sqrt(df[feat1] * df[feat2])

# Apply to data
engineer.data['custom_int'] = custom_interaction(engineer.data, 'feat1', 'feat2')
```

### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Grid search
model = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Use best model
best_model = grid_search.best_estimator_
```

### Ensemble Methods

```python
from sklearn.ensemble import VotingRegressor

# Create ensemble
ensemble = VotingRegressor([
    ('rf', RandomForestRegressor()),
    ('gb', GradientBoostingRegressor()),
    ('ridge', Ridge())
])

# Train ensemble
ensemble.fit(X_train, y_train)
predictions = ensemble.predict(X_test)
```

## Troubleshooting

### Issue: No interactions improve performance

**Solutions**:
1. Try different correlation methods (Spearman, Kendall)
2. Adjust interaction candidate thresholds
3. Try different interaction types (ratio, polynomial)
4. Check for data quality issues (outliers, missing values)
5. Consider that the dataset may be inherently linear

### Issue: Overfitting with interactions

**Solutions**:
1. Reduce number of interactions
2. Add regularization (Ridge, Lasso)
3. Increase cross-validation folds
4. Use simpler models
5. Collect more training data

### Issue: High multicollinearity

**Solutions**:
1. Remove one feature from correlated pairs
2. Use PCA for dimensionality reduction
3. Apply Lasso regression (automatic feature selection)
4. Domain expertise: keep most interpretable feature

### Issue: Residuals not normal

**Solutions**:
1. Try log transformation: `np.log(target + 1)`
2. Try Box-Cox transformation
3. Remove outliers more aggressively
4. Consider non-parametric models

## Best Practices

1. **Always Start with EDA**: Understand your data before engineering features
2. **Use Cross-Validation**: Never rely solely on test set performance
3. **Check Assumptions**: Verify model assumptions through residual analysis
4. **Document Decisions**: Record why specific interactions were selected
5. **Domain Knowledge**: Combine statistical insights with domain expertise
6. **Iterative Process**: Feature engineering is rarely one-and-done
7. **Interpretability**: Prefer interpretable interactions over complex ones
8. **Validate Separately**: Use holdout data not seen during development

## Citation & References

This framework draws inspiration from:
- **tidymodels** (R): Unified modeling interface and tidy model outputs
- **broom** (R): Converting statistical models into tidy data frames
- **Feature Engineering for Machine Learning** by Alice Zheng & Amanda Casari
- **Applied Predictive Modeling** by Max Kuhn & Kjell Johnson

## Support

For issues, questions, or contributions:
- Check the code documentation in each module
- Review the Jupyter notebook for interactive examples
- Examine the output logs for debugging information

---

**Remember**: The goal is not just to improve metrics, but to build interpretable, robust models that leverage both statistical insights and domain knowledge. The "human element" is what transforms automated feature engineering into intelligent model optimization.
