from abc import ABC, abstractmethod
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report as skl_report,
)
from sklearn.model_selection import cross_val_predict
import numpy as np


class AbstractModel(ABC):

    def __init__(self):
        self.model      = None
        self.best_params = None
        self.best_score  = None
        self.threshold   = 0.5      # soglia decisionale, modificabile via tune_threshold()

    @abstractmethod
    def train(self, X_train, y_train, **kwargs):
        """Addestra il modello. Deve impostare self.model e self.best_params."""

    def predict(self, X):
        if self.model is None:
            raise ValueError("Eseguire prima train().")
        # Se il modello supporta predict_proba e la soglia è stata tuned, usala
        if self.threshold != 0.5 and hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(X)[:, 1]
            return (proba >= self.threshold).astype(int)
        return self.model.predict(X)

    def tune_threshold(self, X_train, y_train, cv=5, scoring='f1', thresholds=None):
        """
        Trova la soglia decisionale ottimale usando probabilità out-of-fold.

        Usa cross_val_predict per ottenere probabilità honest (non sul training),
        poi scansiona le soglie e tiene quella che massimizza `scoring`.

        Parametri
        ---------
        scoring : 'f1' | 'f1_weighted' | 'accuracy' | 'recall' | 'precision'
        thresholds : lista di soglie da testare (default: 0.30 → 0.70, step 0.01)

        Ritorna
        -------
        dict con best_threshold e scores per soglia
        """
        if self.model is None:
            raise ValueError("Eseguire prima train().")
        if not hasattr(self.model, 'predict_proba'):
            print(f"  [threshold] Il modello non supporta predict_proba — soglia rimane {self.threshold}")
            return {}

        if thresholds is None:
            thresholds = np.arange(0.30, 0.71, 0.01)

        # Probabilità OOF — honest, non vede i propri dati di training
        import joblib
        with joblib.parallel_backend('threading'):
            proba_oof = cross_val_predict(
                self.model, X_train, y_train,
                cv=cv, method='predict_proba', n_jobs=-1,
            )[:, 1]

        _metric_fn = {
            'f1':           lambda y, p: f1_score(y, p, average='binary', zero_division=0),
            'f1_weighted':  lambda y, p: f1_score(y, p, average='weighted', zero_division=0),
            'accuracy':     lambda y, p: accuracy_score(y, p),
            'recall':       lambda y, p: recall_score(y, p, average='binary', zero_division=0),
            'precision':    lambda y, p: precision_score(y, p, average='binary', zero_division=0),
        }
        metric_fn = _metric_fn.get(scoring, _metric_fn['f1'])

        scores = {}
        for t in thresholds:
            preds = (proba_oof >= t).astype(int)
            scores[round(float(t), 2)] = round(metric_fn(y_train, preds), 4)

        best_t = max(scores, key=scores.__getitem__)
        self.threshold = best_t

        return {'best_threshold': best_t, 'best_score': scores[best_t], 'scores': scores}

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

    def oof_report(self, X_train, y_train, cv: int = 5) -> str:
        """Classification report su predizioni out-of-fold (onesto, non sul train set)."""
        if self.model is None:
            raise ValueError("Eseguire prima train().")
        import joblib
        with joblib.parallel_backend('threading'):
            oof_preds = cross_val_predict(self.model, X_train, y_train, cv=cv, n_jobs=-1)
        return skl_report(y_train, oof_preds, zero_division=0)
