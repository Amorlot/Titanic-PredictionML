import numpy as np
import pandas as pd


class GenericOutlierHandler:
    """
    Rilevamento e trattamento outlier con pattern fit/transform.

    Strategie di rilevamento : 'iqr' | 'zscore' | ['iqr', 'zscore']
    Strategie di trattamento : 'winsorization'
    """

    def __init__(self):
        self._detection = None
        self._treatment = None
        self._iqr_threshold = 1.5
        self._zscore_threshold = 3.0
        self._cols = None
        self._bounds = {}

    def configure(
        self,
        detection,
        treatment: str = 'winsorization',
        iqr_threshold: float = 1.5,
        zscore_threshold: float = 3.0,
        cols: list = None,
    ) -> "GenericOutlierHandler":
        self._detection = [detection] if isinstance(detection, str) else detection
        self._treatment = treatment
        self._iqr_threshold = iqr_threshold
        self._zscore_threshold = zscore_threshold
        self._cols = cols
        return self

    def fit(self, df: pd.DataFrame) -> "GenericOutlierHandler":
        num_cols = self._cols if self._cols else df.select_dtypes(include=[np.number]).columns.tolist()
        self._bounds = {}

        for col in num_cols:
            s = df[col].dropna()
            lower, upper = -np.inf, np.inf

            if 'iqr' in self._detection:
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                iqr = q3 - q1
                lower = max(lower, q1 - self._iqr_threshold * iqr)
                upper = min(upper, q3 + self._iqr_threshold * iqr)

            if 'zscore' in self._detection:
                mean, std = s.mean(), s.std()
                lower = max(lower, mean - self._zscore_threshold * std)
                upper = min(upper, mean + self._zscore_threshold * std)

            self._bounds[col] = (lower, upper)

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if self._treatment == 'winsorization':
            for col, (lower, upper) in self._bounds.items():
                if col in df.columns:
                    df[col] = df[col].clip(lower=lower, upper=upper)
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def report(self, df: pd.DataFrame) -> dict:
        result = {}
        for col, (lower, upper) in self._bounds.items():
            if col in df.columns:
                n_outliers = int(((df[col] < lower) | (df[col] > upper)).sum())
                if n_outliers:
                    result[col] = {
                        'outliers': n_outliers,
                        'pct': round(n_outliers / len(df) * 100, 2),
                        'lower_bound': round(lower, 4),
                        'upper_bound': round(upper, 4),
                    }
        return result
