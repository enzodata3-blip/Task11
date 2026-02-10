# Quick Start Guide - ML Optimization Framework

## ⚡ Get Started in 3 Steps

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

Or with virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2: Run the Demo Notebook

The easiest way to verify everything works is to run the demo with synthetic data:

```bash
jupyter notebook notebooks/00_demo_with_synthetic_data.ipynb
```

Then click "Run All" to execute all cells. This notebook:
- ✅ Generates synthetic housing data with known interaction effects
- ✅ Demonstrates the complete optimization workflow
- ✅ Shows visualizations and results
- ✅ Validates that discovered interactions match true relationships
- ✅ Saves all outputs to `results/` directory

**Expected runtime**: 2-3 minutes

### Step 3: Use Your Own Data

Once you've verified the demo works, adapt it for your data:

1. Place your dataset in `data/raw/your_data.csv`
2. Open `notebooks/01_exploratory_analysis.ipynb`
3. Update these lines:
   ```python
   data_path = '../data/raw/your_data.csv'
   target_col = 'your_target_column'
   ```
4. Run all cells

Alternatively, use the command line:
```bash
python src/main.py --data data/raw/your_data.csv --target your_target --interactions 10
```

---

## 📊 What You'll Get

After running the pipeline, you'll find:

### In `results/` directory:
- `correlation_heatmap.png` - Feature correlation visualization
- `target_correlations.png` - Feature-target relationship plot
- `Enhanced_Random_Forest_predictions.png` - Predicted vs actual values
- `Enhanced_Random_Forest_residuals.png` - Residual diagnostic plots
- `Enhanced_Random_Forest_errors.png` - Error distribution analysis
- `feature_importance.csv` - Ranked feature importance (with interactions)
- `model_comparison.csv` - Performance comparison table

### In `data/processed/` directory:
- `enhanced_data.csv` - Your data with beneficial interaction terms added

### In `models/` directory:
- `best_model.joblib` - Trained model ready for predictions

---

## 🎯 Key Concepts

### What are Interaction Terms?

Interaction terms capture non-linear relationships between features:
- **Multiplicative**: `income × education` (high income + high education = exponential value)
- **Ratio**: `price / area` (price per unit area)
- **Polynomial**: `age²` (non-linear age effects)

### Why Correlation Analysis?

The framework uses correlation matrices to identify promising interactions:
1. Features correlated with target are valuable
2. Features with moderate inter-correlation may interact
3. Systematic search finds interactions you might miss manually

### The "Human Element"

Machine learning models learn patterns but don't understand **why** features interact. This framework:
- Uses statistical analysis to **guide** feature engineering
- Evaluates each interaction's **actual impact** on performance
- Keeps only **beneficial** interactions (avoids overfitting)
- Produces **interpretable** results you can explain

---

## 🔧 Troubleshooting

### Notebook won't run?
```bash
# Install Jupyter if missing
pip install jupyter ipykernel

# Register kernel
python -m ipykernel install --user --name=ml-opt
```

### Import errors?
```bash
# Make sure you're in the correct directory
cd /path/to/model_a

# Install all dependencies
pip install -r requirements.txt
```

### Module not found errors in notebook?
The notebooks add `../src` to path automatically. If it fails:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / 'src'))
```

---

## 📚 Next Steps

1. **Run the demo** (`00_demo_with_synthetic_data.ipynb`) to see it work
2. **Try your data** (`01_exploratory_analysis.ipynb`)
3. **Read detailed docs** (`USAGE_GUIDE.md`) for advanced usage
4. **Experiment** with different parameters and interaction types

---

## 💡 Tips for Best Results

✅ **Start simple**: Begin with top 10 interaction candidates
✅ **Use cross-validation**: Don't trust single train/test split
✅ **Check residuals**: Ensure model assumptions are met
✅ **Domain knowledge**: Combine stats with your expertise
✅ **Iterate**: Feature engineering is a process, not one-shot

---

**Need help?** Check the full `USAGE_GUIDE.md` or inspect the module docstrings.

**Ready to optimize?** Open that demo notebook and let's go! 🚀
