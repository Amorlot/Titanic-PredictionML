import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OrdinalEncoder


class GenericCleaner:
    """
    Pulizia valori mancanti con pattern fit/transform.

    Strategie numeriche : 'median' | 'mean' | 'most_frequent' | 'knn'
    Strategie categoriali: 'unknown'       → costante 'unknown'
                           'most_frequent' → valore più frequente del train
                           'knn'           → KNN su encoding ordinale temporaneo
    """

    def __init__(self):
        self._num_strategy = None
        self._cat_strategy = None
        self._knn_neighbors = 5
        self._explicit_num_cols = None
        self._explicit_cat_cols = None

        self._num_cols = []
        self._cat_cols = []
        self._num_imputer = None
        self._cat_imputer = None
        self._knn_imputer = None
        self._knn_ord_encoder = None
        self._all_cols_order = None

    def configure(
        self,
        num_strategy: str = 'median',
        cat_strategy: str = 'unknown',
        knn_neighbors: int = 5,
        num_cols: list = None,
        cat_cols: list = None,
    ) -> "GenericCleaner":
        self._num_strategy = num_strategy
        self._cat_strategy = cat_strategy
        self._knn_neighbors = knn_neighbors
        self._explicit_num_cols = num_cols
        self._explicit_cat_cols = cat_cols
        return self

    def fit(self, df: pd.DataFrame) -> "GenericCleaner":
        num_all = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_all = df.select_dtypes(exclude=[np.number]).columns.tolist()

        self._num_cols = self._explicit_num_cols if self._explicit_num_cols is not None \
            else [c for c in num_all if df[c].isnull().any()]
        self._cat_cols = self._explicit_cat_cols if self._explicit_cat_cols is not None \
            else [c for c in cat_all if df[c].isnull().any()]

        use_knn = self._num_strategy == 'knn' or self._cat_strategy == 'knn'

        if use_knn:
            # Ordinal-encode i categoriali su tutte le colonne cat (non solo quelle con NaN)
            # così KNNImputer lavora su numeri
            self._knn_ord_encoder = OrdinalEncoder(
                handle_unknown='use_encoded_value', unknown_value=-1
            )
            self._knn_ord_encoder.fit(df[cat_all])

            df_num = df[num_all].copy()
            df_cat_enc = pd.DataFrame(
                self._knn_ord_encoder.transform(df[cat_all]),
                columns=cat_all, index=df.index
            )
            df_knn = pd.concat([df_num, df_cat_enc], axis=1)
            self._all_cols_order = num_all + cat_all

            self._knn_imputer = KNNImputer(n_neighbors=self._knn_neighbors)
            self._knn_imputer.fit(df_knn[self._all_cols_order])

        else:
            if self._num_cols and self._num_strategy in ('median', 'mean', 'most_frequent'):
                self._num_imputer = SimpleImputer(strategy=self._num_strategy)
                self._num_imputer.fit(df[self._num_cols])

            if self._cat_cols and self._cat_strategy == 'most_frequent':
                self._cat_imputer = SimpleImputer(strategy='most_frequent')
                self._cat_imputer.fit(df[self._cat_cols])

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        use_knn = self._num_strategy == 'knn' or self._cat_strategy == 'knn'

        if use_knn:
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

            df_cat_enc = pd.DataFrame(
                self._knn_ord_encoder.transform(df[cat_cols]),
                columns=cat_cols, index=df.index
            )
            df_knn = pd.concat([df[num_cols], df_cat_enc], axis=1)
            imputed = self._knn_imputer.transform(df_knn[self._all_cols_order])
            df_imputed = pd.DataFrame(imputed, columns=self._all_cols_order, index=df.index)

            # Ripristina i categoriali come stringhe via inverse_transform
            cat_imputed = np.round(df_imputed[cat_cols].values).astype(int).clip(0)
            df[cat_cols] = self._knn_ord_encoder.inverse_transform(cat_imputed)
            df[num_cols] = df_imputed[num_cols].values

        else:
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
