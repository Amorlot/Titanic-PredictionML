import pandas as pd
from sklearn.decomposition import PCA


class PCAReducer:
    """
    Riduzione dimensionalità con PCA con pattern fit/transform.

    n_components: int   → numero fisso di componenti
                  float → varianza spiegata da mantenere (es. 0.95)
    """

    def __init__(self):
        self._n_components = 0.95
        self._random_state = 42
        self._pca = None

    def configure(self, n_components=0.95, random_state=42) -> "PCAReducer":
        self._n_components = n_components
        self._random_state = random_state
        return self

    def fit(self, df: pd.DataFrame) -> "PCAReducer":
        self._pca = PCA(n_components=self._n_components, random_state=self._random_state)
        self._pca.fit(df)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        transformed = self._pca.transform(df)
        cols = [f"PC{i+1}" for i in range(transformed.shape[1])]
        return pd.DataFrame(transformed, columns=cols, index=df.index)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def report(self) -> dict:
        if self._pca is None:
            return {}
        return {
            'n_components_in':  int(self._pca.n_features_in_),
            'n_components_out': int(self._pca.n_components_),
            'variance_explained': round(float(self._pca.explained_variance_ratio_.sum()), 4),
        }
