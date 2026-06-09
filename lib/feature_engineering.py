import numpy as np
import pandas as pd


_TITLE_MAP = {
    'Mr':       'Mr',
    'Miss':     'Miss',
    'Mrs':      'Mrs',
    'Master':   'Master',
    'Dr':       'Officer',
    'Rev':      'Officer',
    'Capt':     'Officer',
    'Col':      'Officer',
    'Major':    'Officer',
    'Sir':      'Noble',
    'Lady':     'Noble',
    'Don':      'Noble',
    'Dona':     'Noble',
    'Jonkheer': 'Noble',
    'Countess': 'Noble',
    'Mme':      'Mrs',
    'Mlle':     'Miss',
    'Ms':       'Miss',
}

_AGE_BINS   = [0, 12, 17, 35, 60, 120]
_AGE_LABELS = ['Child', 'Teen', 'Adult', 'Middle', 'Senior']


class FeatureEngineer:
    """
    Feature engineering per il dataset Titanic.

    - Estrae TitleGroup da Name
    - Calcola FarePerPerson = Fare / (SibSp + Parch + 1)
    - Crea AgeGroup da Age (Child/Teen/Adult/Middle/Senior)
    - Crea SocialClass (1-9) combinando Pclass e FarePerPerson tier relativo alla classe
    """

    def __init__(self):
        self._fare_quantiles = {}
        self._drop_raw_fare_for = []
        self._age_medians = {}      # mediana età per TitleGroup (calcolata sul train)
        self._age_fallback = None   # mediana globale come fallback

    def configure(self, drop_raw_fare_for: list = None) -> "FeatureEngineer":
        self._drop_raw_fare_for = drop_raw_fare_for or []
        return self

    def fit(self, df: pd.DataFrame) -> "FeatureEngineer":
        # Mediana età per TitleGroup — usata per imputare Age prima del cleaner
        if 'Name' in df.columns and 'Age' in df.columns:
            titles = df['Name'].str.extract(r',\s*([^\.]+)\.')[0].str.strip()
            title_groups = titles.map(_TITLE_MAP).fillna('Rare')
            temp_age = df.assign(TitleGroup=title_groups)
            self._age_medians  = temp_age.groupby('TitleGroup')['Age'].median().to_dict()
            self._age_fallback = df['Age'].median()

        # Quantili FarePerPerson per SocialClass
        fare_pp = df['Fare'] / (df['SibSp'] + df['Parch'] + 1)
        temp = df.assign(FarePerPerson=fare_pp)
        for pclass, group in temp.groupby('Pclass'):
            fpp = group['FarePerPerson'].dropna()
            self._fare_quantiles[pclass] = (
                fpp.quantile(0.33),
                fpp.quantile(0.66),
            )
        return self

    def transform(self, df: pd.DataFrame, model_name: str = None) -> pd.DataFrame:
        df = df.copy()

        # ── TitleGroup ───────────────────────────────────────────
        if 'Name' in df.columns:
            titles = df['Name'].str.extract(r',\s*([^\.]+)\.')[0].str.strip()
            df['TitleGroup'] = titles.map(_TITLE_MAP).fillna('Rare')
            df.drop(columns=['Name'], inplace=True)

        # ── Imputazione Age per TitleGroup ───────────────────────
        # Master ~5, Miss ~22, Mrs ~36, Mr ~30 — molto più precisa della mediana globale
        if 'Age' in df.columns and df['Age'].isnull().any() and self._age_medians:
            df['Age'] = df.apply(
                lambda r: self._age_medians.get(r['TitleGroup'], self._age_fallback)
                          if pd.isna(r['Age']) else r['Age'],
                axis=1,
            )

        # ── Cabin features (prima del drop) ──────────────────────
        if 'Cabin' in df.columns:
            df['CabinKnown'] = df['Cabin'].notna().astype(int)
            df['CabinDeck']  = df['Cabin'].str[0].fillna('U')
            df.drop(columns=['Cabin'], inplace=True)

        # ── FamilySize / IsAlone ──────────────────────────────────
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone']    = (df['FamilySize'] == 1).astype(int)

        # ── FarePerPerson ────────────────────────────────────────
        df['FarePerPerson'] = df['Fare'] / df['FamilySize']

        # ── AgeGroup ─────────────────────────────────────────────
        df['AgeGroup'] = pd.cut(
            df['Age'],
            bins=_AGE_BINS,
            labels=_AGE_LABELS,
            right=True,
            include_lowest=True,
        ).astype(object)

        # ── Fare tier per persona relativo alla classe ───────────
        def fare_tier(row):
            pclass = row['Pclass']
            fpp    = row['FarePerPerson']
            if pd.isna(fpp) or pclass not in self._fare_quantiles:
                return 1
            q33, q66 = self._fare_quantiles[pclass]
            if fpp <= q33:
                return 0
            elif fpp <= q66:
                return 1
            return 2

        df['_fare_tier'] = df.apply(fare_tier, axis=1)

        # ── SocialClass 1-9 (1=più alto, 9=più basso) ────────────
        df['SocialClass'] = (df['Pclass'] - 1) * 3 + (2 - df['_fare_tier']) + 1
        df.drop(columns=['_fare_tier'], inplace=True)

        # ── Drop Fare grezzo per modelli lineari ─────────────────
        if model_name in self._drop_raw_fare_for and 'Fare' in df.columns:
            df.drop(columns=['Fare'], inplace=True)

        return df

    def fit_transform(self, df: pd.DataFrame, model_name: str = None) -> pd.DataFrame:
        return self.fit(df).transform(df, model_name=model_name)

    def report(self, df: pd.DataFrame) -> dict:
        result = {}
        if 'TitleGroup' in df.columns:
            result['TitleGroup'] = df['TitleGroup'].value_counts().to_dict()
        if self._age_medians:
            result['AgeImputationByTitle'] = {k: round(v, 1) for k, v in self._age_medians.items()}
        if 'AgeGroup' in df.columns:
            result['AgeGroup'] = df['AgeGroup'].value_counts().to_dict()
        if 'SocialClass' in df.columns:
            result['SocialClass'] = df['SocialClass'].value_counts().sort_index().to_dict()
        if 'FarePerPerson' in df.columns:
            result['FarePerPerson'] = {
                'mean': round(float(df['FarePerPerson'].mean()), 2),
                'min':  round(float(df['FarePerPerson'].min()), 2),
                'max':  round(float(df['FarePerPerson'].max()), 2),
            }
        if 'FamilySize' in df.columns:
            result['FamilySize'] = df['FamilySize'].value_counts().sort_index().to_dict()
        if 'IsAlone' in df.columns:
            result['IsAlone'] = df['IsAlone'].value_counts().to_dict()
        if 'CabinKnown' in df.columns:
            result['CabinKnown'] = df['CabinKnown'].value_counts().to_dict()
        if 'CabinDeck' in df.columns:
            result['CabinDeck'] = df['CabinDeck'].value_counts().sort_index().to_dict()
        return result
