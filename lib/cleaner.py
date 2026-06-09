import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


class GenericCleaner:
    """
    Pulizia valori mancanti con pattern fit/transform.

    Strategie numeriche : 'median' | 'mean' | 'most_frequent'
    Strategie categoriali: 'unknown'       → costante 'unknown'
                           'most_frequent' → valore più frequente del train
    """

    def __init__(self):
        self._num_strategy = None
        self._cat_strategy = None
        self._explicit_num_cols = None
        self._explicit_cat_cols = None

        self._num_cols = []
        self._cat_cols = []
        self._num_imputer = None
        self._cat_imputer = None

    def configure(
        self,
        num_strategy: str = 'median',
        cat_strategy: str = 'unknown',
        num_cols: list = None,
        cat_cols: list = None,
    ) -> "GenericCleaner":
        """
        num_strategy : strategia per i numerici con NaN
        cat_strategy : strategia per i categoriali con NaN
        num_cols     : se None, auto-rileva dal train
        cat_cols     : se None, auto-rileva dal train
        """
        self._num_strategy = num_strategy
        self._cat_strategy = cat_strategy
        self._explicit_num_cols = num_cols
        self._explicit_cat_cols = cat_cols
        return self

    def fit(self, df: pd.DataFrame) -> "GenericCleaner":
        if self._num_strategy is not None:
            if self._explicit_num_cols is not None:
                self._num_cols = self._explicit_num_cols
            else:
                num = df.select_dtypes(include=[np.number]).columns.tolist()
                self._num_cols = [c for c in num if df[c].isnull().any()]

            if self._num_cols:
                self._num_imputer = SimpleImputer(strategy=self._num_strategy)
                self._num_imputer.fit(df[self._num_cols])

        if self._cat_strategy is not None:
            if self._explicit_cat_cols is not None:
                self._cat_cols = self._explicit_cat_cols
            else:
                cat = df.select_dtypes(exclude=[np.number]).columns.tolist()
                self._cat_cols = [c for c in cat if df[c].isnull().any()]

            if self._cat_cols and self._cat_strategy == 'most_frequent':
                self._cat_imputer = SimpleImputer(strategy='most_frequent')
                self._cat_imputer.fit(df[self._cat_cols])

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if self._num_imputer and self._num_cols:
            df[self._num_cols] = self._num_imputer.transform(df[self._num_cols])

        if self._cat_cols:
            if self._cat_strategy == 'unknown':
                for col in self._cat_cols:
                    if col in df.columns:
                        df[col] = df[col].fillna('unknown')
            elif self._cat_strategy == 'most_frequent' and self._cat_imputer:
                df[self._cat_cols] = self._cat_imputer.transform(df[self._cat_cols])

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def report(self, df: pd.DataFrame) -> dict:
        total = len(df)
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        return {
            col: {"count": int(count), "pct": round(count / total * 100, 2)}
            for col, count in missing.items()
        }
