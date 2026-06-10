import numpy as np
from lib.models.base import AbstractModel


class GenericEnsemble(AbstractModel):
    """
    Ensemble pesato di modelli già addestrati.

    voting='soft'  → media pesata delle probabilità (predict_proba), poi soglia
    voting='hard'  → voto a maggioranza pesata sulle classi predette

    weights può essere:
      - 'auto': usa il cv_score di ciascun modello come peso
      - dict {nome: peso}: pesi manuali (vengono normalizzati automaticamente)
    """

    def __init__(self, members: list[tuple[str, AbstractModel, float]], voting: str = 'soft'):
        """
        Parameters
        ----------
        members : lista di (name, model, weight)
        voting  : 'soft' | 'hard'
        """
        super().__init__()
        self.members = members   # [(name, model, weight), ...]
        self.voting  = voting
        self.model   = self      # soddisfa il check "model is None" nella base class
        self.best_params = {name: m.best_params for name, m, _ in members}
        self.best_score  = None  # impostato dopo da main.py

    def train(self, X_train, y_train, **kwargs):
        # I modelli membri sono già addestrati; l'ensemble non ha nulla da imparare.
        return self

    def predict(self, X):
        total_weight = sum(w for _, _, w in self.members)

        if self.voting == 'soft':
            proba = np.zeros(len(X))
            for _, model, weight in self.members:
                if not hasattr(model.model, 'predict_proba'):
                    raise ValueError(
                        f"Il modello '{type(model).__name__}' non supporta predict_proba. "
                        "Usa voting='hard' oppure rimuovilo dall'ensemble."
                    )
                p = model.model.predict_proba(X)[:, 1]
                proba += (weight / total_weight) * p
            return (proba >= self.threshold).astype(int)

        else:  # hard voting
            votes = np.zeros(len(X))
            for _, model, weight in self.members:
                votes += (weight / total_weight) * model.predict(X)
            return (votes >= 0.5).astype(int)
