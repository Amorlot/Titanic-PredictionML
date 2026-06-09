import pandas as pd
import numpy as np


class GenericEda:
    """
    Analisi esplorativa generica per dataset misti (numerici + categoriali).
    Tutti i metodi restituiscono dizionari JSON-serializzabili.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def summary(self) -> dict:
        """Statistiche descrittive per colonne numeriche e categoriali."""
        result = {}

        num_df = self.df.select_dtypes(include='number')
        if not num_df.empty:
            desc = num_df.describe().round(4)
            result['numerical'] = {
                col: {k: (None if np.isnan(v) else v) for k, v in desc[col].items()}
                for col in desc.columns
            }

        cat_df = self.df.select_dtypes(exclude='number')
        if not cat_df.empty:
            result['categorical'] = {
                col: {
                    'unique': int(cat_df[col].nunique()),
                    'top_5': cat_df[col].value_counts().head(5).to_dict(),
                }
                for col in cat_df.columns
            }

        return result

    def missing_report(self) -> dict:
        """Valori mancanti per colonna."""
        total = len(self.df)
        missing = self.df.isnull().sum()
        missing = missing[missing > 0]
        return {
            col: {"count": int(count), "pct": round(count / total * 100, 2)}
            for col, count in missing.items()
        }

    def correlation(self) -> dict:
        """Matrice di correlazione (solo colonne numeriche)."""
        num_df = self.df.select_dtypes(include='number')
        if num_df.shape[1] < 2:
            return {}
        corr = num_df.corr().round(4)
        return {
            col: {k: (None if np.isnan(v) else v) for k, v in corr[col].items()}
            for col in corr.columns
        }

    def class_balance(self, y) -> dict:
        """Distribuzione del target."""
        counts = pd.Series(y).value_counts()
        total = len(y)
        return {
            str(cls): {"count": int(cnt), "pct": round(cnt / total * 100, 2)}
            for cls, cnt in counts.items()
        }
