# Project Structure

```
model_a/
│
├── README.md                          # Project overview and introduction
├── QUICKSTART.md                      # 3-step quick start guide
├── USAGE_GUIDE.md                     # Comprehensive usage documentation
├── PROJECT_STRUCTURE.md               # This file
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
│
├── data/                              # Data directory
│   ├── raw/                           # Raw data files (place your datasets here)
│   │   └── .gitkeep                   # Keeps empty directory in git
│   └── processed/                     # Processed/enhanced data outputs
│       └── .gitkeep
│
├── notebooks/                         # Jupyter notebooks for interactive analysis
│   ├── 00_demo_with_synthetic_data.ipynb    # ⭐ START HERE - Demo with synthetic data
│   └── 01_exploratory_analysis.ipynb        # Template for your own data
│
├── src/                               # Source code modules
│   ├── __init__.py                    # Package initialization
│   ├── data_processing.py             # Data loading, cleaning, preprocessing
│   ├── correlation_analysis.py        # Correlation matrices, interaction candidates
│   ├── interaction_engineering.py     # Create and evaluate interaction terms
│   ├── model_training.py              # Train baseline & enhanced models
│   ├── evaluation.py                  # Comprehensive model evaluation
│   └── main.py                        # Complete pipeline orchestration
│
├── models/                            # Saved model files
│   └── .gitkeep
│
├── results/                           # Outputs: plots, reports, metrics
│   └── .gitkeep
│
└── tests/                             # Unit tests (optional)
    └── .gitkeep
```

## Core Modules Overview

### 📊 `data_processing.py`
- Load data from CSV, Excel, Parquet, JSON
- Handle missing values (drop, mean, median, mode, constant)
- Detect and remove outliers (IQR, Z-score)
- Encode categorical variables
- Generate comprehensive data profiles

### 🔗 `correlation_analysis.py`
- Compute correlation matrices (Pearson, Spearman, Kendall)
- Identify feature-target correlations
- Detect multicollinearity
- Find promising interaction candidates
- Generate correlation visualizations

### 🔧 `interaction_engineering.py`
- Create multiplicative interactions (f1 × f2)
- Create polynomial features (f^n)
- Create ratio interactions (f1 / f2)
- Create difference interactions (f1 - f2)
- Evaluate interaction importance via cross-validation
- Select best interactions based on performance

### 🤖 `model_training.py`
- Train multiple baseline models (Linear, Ridge, Lasso, RF, GB)
- Train enhanced models with interactions
- Cross-validation with multiple metrics
- Compare baseline vs enhanced performance
- Save/load trained models

### 📈 `evaluation.py`
- Comprehensive metrics (R², RMSE, MAE, MAPE)
- Residual analysis (normality, autocorrelation)
- Visualizations (predictions, residuals, Q-Q plots, errors)
- Statistical hypothesis testing
- Multi-model comparison

### 🚀 `main.py`
- Complete end-to-end pipeline
- Orchestrates all modules
- Command-line interface
- Configurable parameters
- Automated workflow execution

## Quick Navigation

| I want to...                                      | Go to...                                           |
|---------------------------------------------------|---------------------------------------------------|
| Get started immediately                           | `QUICKSTART.md`                                   |
| Run a working demo                                | `notebooks/00_demo_with_synthetic_data.ipynb`     |
| Use my own data                                   | `notebooks/01_exploratory_analysis.ipynb`         |
| Understand the methodology                        | `USAGE_GUIDE.md`                                  |
| Use command line                                  | Run `python src/main.py --help`                   |
| Understand a specific module                      | Read docstrings in `src/*.py` files               |
| See example outputs                               | Check `results/` after running demo               |
| Customize preprocessing                           | Edit parameters in `data_processing.py`           |
| Try different interaction types                   | Modify `interaction_engineering.py` calls         |
| Add new models                                    | Extend `model_training.py`                        |

## Data Flow

```
1. Raw Data (CSV, Excel, etc.)
   ↓
2. Data Processing
   ↓ (cleaned data)
3. Correlation Analysis
   ↓ (interaction candidates)
4. Interaction Engineering
   ↓ (enhanced dataset)
5. Model Training
   ↓ (baseline + enhanced models)
6. Model Evaluation
   ↓ (metrics + visualizations)
7. Results & Saved Model
```

## Output Files

After running the pipeline, expect these outputs:

### In `results/`:
- `correlation_heatmap.png` - Feature correlation matrix visualization
- `target_correlations.png` - Top features correlated with target
- `interaction_importance.png` - Interaction terms ranked by value
- `Enhanced_Random_Forest_predictions.png` - Predicted vs actual plot
- `Enhanced_Random_Forest_residuals.png` - 4-panel residual analysis
- `Enhanced_Random_Forest_errors.png` - Error distribution plots
- `feature_importance.csv` - All features ranked by importance
- `model_comparison.csv` - Performance metrics for all models

### In `data/processed/`:
- `enhanced_data.csv` - Original data + beneficial interaction terms

### In `models/`:
- `best_model.joblib` - Trained model (use joblib.load() to reload)

## Key Design Principles

1. **Modularity**: Each module can be used independently
2. **Reproducibility**: Random seeds, cross-validation, consistent splits
3. **Interpretability**: Tidy outputs, clear visualizations, statistical rigor
4. **Flexibility**: Configurable parameters, multiple options
5. **Best Practices**: Inspired by tidymodels (R) and scikit-learn (Python)

## Workflow Options

### Option 1: Interactive (Jupyter) - Recommended for Exploration
```bash
jupyter notebook notebooks/00_demo_with_synthetic_data.ipynb
```

### Option 2: Command Line - For Production/Automation
```bash
python src/main.py --data data/raw/housing.csv --target price --interactions 10
```

### Option 3: Python Script - For Custom Workflows
```python
from src import MLOptimizationPipeline

pipeline = MLOptimizationPipeline(
    data_path='data/raw/data.csv',
    target_col='target',
    random_state=42
)
results = pipeline.run_full_pipeline()
```

---

**🎯 Recommendation**: Start with the demo notebook (`00_demo_with_synthetic_data.ipynb`) to see everything working, then adapt for your data using `01_exploratory_analysis.ipynb`.
