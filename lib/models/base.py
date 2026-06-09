from abc import ABC, abstractmethod
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report as skl_report
)
import numpy as np


class AbstractModel(ABC):

    def __init__(self):
        self.model = None
        self.best_params = None

    @abstractmethod
    def train(self, X_train, y_train, **kwargs):
        """Addestra il modello. Deve impostare self.model e self.best_params."""

    def predict(self, X):
        if self.model is None:
            raise ValueError("Eseguire prima train().")
        return self.model.predict(X)

    def evaluate(self, X_test, y_test, average='weighted') -> dict:
        if self.model is None:
            raise ValueError("Eseguire prima train().")
        predictions = self.predict(X_test)
        n_classes = len(np.unique(y_test))
        avg = 'binary' if n_classes == 2 else average
        return {
            "accuracy":  accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, average=avg, zero_division=0),
            "recall":    recall_score(y_test, predictions, average=avg, zero_division=0),
            "f1_score":  f1_score(y_test, predictions, average=avg, zero_division=0),
        }

    def classification_report(self, X_test, y_test) -> str:
        if self.model is None:
            raise ValueError("Eseguire prima train().")
        return skl_report(y_test, self.predict(X_test), zero_division=0)
