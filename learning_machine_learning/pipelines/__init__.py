"""Pipeline abstrait et implémentations concrètes."""

from learning_machine_learning.pipelines.base import BasePipeline
from learning_machine_learning.pipelines.eurusd import EurUsdPipeline

__all__ = ["BasePipeline", "EurUsdPipeline"]
