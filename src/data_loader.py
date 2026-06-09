import numpy as np
import pandas as pd

_MISSING_MARKERS = r'^\s*(\?|nan|NaN|N/A|NA|none|None|--)\s*$'


class DataLoader:
    def __init__(self, csv_path: str, target_col: str, drop_cols: list = None, drop_missing_thresh: float = None):
        self.csv_path = csv_path
        self.target_col = target_col
        self.drop_cols = drop_cols or []
        self.drop_missing_thresh = drop_missing_thresh
        self.df = None

    def load(self) -> pd.DataFrame:
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
            col: {"count": int(cnt), "pct": round(cnt / total * 100, 2)}
            for col, cnt in missing.items()
        }
