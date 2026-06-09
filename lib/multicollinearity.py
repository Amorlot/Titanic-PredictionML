import numpy as np
import pandas as pd


class MulticollinearityHandler:
    """
    Rilevamento e trattamento della multicollinearità tramite matrice di correlazione.

    fit()      : individua le coppie con |r| > threshold e decide quali colonne droppare
    transform(): droppa le colonne identificate (se drop=True)
    report()   : restituisce le coppie correlate trovate
    """

    def __init__(self):
        self._threshold = 0.85
        self._drop = True
        self._correlated_pairs = []
        self._cols_to_drop = []

    def configure(self, threshold: float = 0.85, drop: bool = True) -> "MulticollinearityHandler":
        self._threshold = threshold
        self._drop = drop
        return self

    def fit(self, df: pd.DataFrame) -> "MulticollinearityHandler":
        num_df = df.select_dtypes(include=[np.number])
        corr = num_df.corr().abs()

        self._correlated_pairs = []
        seen = set()

        for i, col_a in enumerate(corr.columns):
            for col_b in corr.columns[i + 1:]:
                r = corr.loc[col_a, col_b]
                if r > self._threshold:
                    self._correlated_pairs.append((col_a, col_b, round(float(r), 4)))
                    seen.add(col_a)
                    seen.add(col_b)

        # Droppa la colonna con la correlazione media più alta con tutte le altre
        self._cols_to_drop = []
        dropped = set()
        for col_a, col_b, _ in self._correlated_pairs:
            if col_a in dropped or col_b in dropped:
                continue
            mean_a = corr[col_a].drop(index=col_a).mean()
            mean_b = corr[col_b].drop(index=col_b).mean()
            to_drop = col_a if mean_a >= mean_b else col_b
            self._cols_to_drop.append(to_drop)
            dropped.add(to_drop)

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._drop and self._cols_to_drop:
            return df.drop(columns=[c for c in self._cols_to_drop if c in df.columns])
        return df.copy()

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def report(self) -> dict:
        return {
            'pairs': [
                {'col_a': a, 'col_b': b, 'r': r}
                for a, b, r in self._correlated_pairs
            ],
            'cols_to_drop': self._cols_to_drop,
        }
