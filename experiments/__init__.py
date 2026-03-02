"""
Experiments package for running and analyzing feature detection comparisons
"""
from .experiment_runner import ExperimentRunner
from .results_analyzer import ResultsAnalyzer

__all__ = [
    'ExperimentRunner',
    'ResultsAnalyzer'
]