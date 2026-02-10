"""
ML Optimization Framework
Machine learning model optimization through human-guided interaction term engineering.
"""

__version__ = "1.0.0"
__author__ = "Enzo Rodriguez"
__task_id__ = "TASK_11251"

from .data_processing import DataProcessor
from .correlation_analysis import CorrelationAnalyzer
from .interaction_engineering import InteractionEngineer
from .model_training import ModelTrainer
from .evaluation import ModelEvaluator, compare_multiple_models
from .main import MLOptimizationPipeline

__all__ = [
    'DataProcessor',
    'CorrelationAnalyzer',
    'InteractionEngineer',
    'ModelTrainer',
    'ModelEvaluator',
    'compare_multiple_models',
    'MLOptimizationPipeline'
]
