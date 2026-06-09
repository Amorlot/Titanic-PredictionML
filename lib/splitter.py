import pandas as pd
from sklearn.model_selection import train_test_split


class GenericSplitter:
    """
    Train/test split configurabile.
    stratify=True abilita lo split stratificato sul target (consigliato per classificazione).
    """

    def __init__(self, test_size: float = 0.2, random_state: int = 42, stratify: bool = True):
        self.test_size = test_size
        self.random_state = random_state
        self.stratify = stratify

    def split(self, X: pd.DataFrame, y):
        strat = y if self.stratify else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=strat,
        )
        return X_train, X_test, y_train, y_test
