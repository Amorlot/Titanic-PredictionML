import numpy as np
import pandas as pd

_MISSING_MARKERS = r'^\s*(\?|nan|NaN|N/A|NA|none|None|--)\s*$'


class GenericLoader:
    """
    Carica un dataset da UCI (per id) o da file CSV.
    Supporta qualsiasi combinazione di feature numeriche e categoriali.
    """

    def __init__(
        self,
        target_col: str,
        dataset_id: int = None,
        csv_path: str = None,
        drop_cols: list = None,
        drop_missing_thresh: float = None,
    ):
        if dataset_id is None and csv_path is None:
            raise ValueError("Specificare dataset_id oppure csv_path.")

        self.target_col = target_col
        self.dataset_id = dataset_id
        self.csv_path = csv_path
        self.drop_cols = drop_cols or []
        self.drop_missing_thresh = drop_missing_thresh

        self.df = None

    def load(self) -> pd.DataFrame:
        if self.dataset_id is not None:
            from ucimlrepo import fetch_ucirepo
            dataset = fetch_ucirepo(id=self.dataset_id)
            self.df = pd.concat([
                dataset.data.features.copy(),
                dataset.data.targets.copy(),
            ], axis=1)
        else:
            self.df = pd.read_csv(self.csv_path)

        self.df.replace(to_replace=_MISSING_MARKERS, value=np.nan, regex=True, inplace=True)

        if self.drop_cols:
            cols = [c for c in self.drop_cols if c in self.df.columns]
            self.df.drop(columns=cols, inplace=True)

        if self.drop_missing_thresh is not None:
            thresh = int((1 - self.drop_missing_thresh) * len(self.df))
            before = set(self.df.columns)
            self.df.dropna(thresh=thresh, axis=1, inplace=True)
            dropped = before - set(self.df.columns)
            if dropped:
                print(f"[Loader] Colonne droppate (>{self.drop_missing_thresh*100:.0f}% NaN): {dropped}")

        return self.df

    def info(self) -> dict:
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]
        return {
            "rows": len(self.df),
            "columns": len(self.df.columns),
            "features": len(X.columns),
            "numerical": X.select_dtypes(include="number").columns.tolist(),
            "categorical": X.select_dtypes(exclude="number").columns.tolist(),
            "target": self.target_col,
            "target_distribution": y.value_counts().to_dict(),
        }

    def missing_report(self) -> dict:
        total = len(self.df)
        missing = self.df.isnull().sum()
        missing = missing[missing > 0]
        return {
            col: {"count": int(count), "pct": round(count / total * 100, 2)}
            for col, count in missing.items()
        }
