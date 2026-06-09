import pandas as pd
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler,
    OneHotEncoder, OrdinalEncoder,
    LabelEncoder,
)


class Encoder:
    def __init__(self):
        self._num_strategy = 'standard'
        self._cat_strategy = 'ohe'
        self._num_cols = None
        self._cat_cols = None
        self._num_scaler = None
        self._cat_encoder = None
        self._target_le = LabelEncoder()

    def configure(self, num_strategy: str = 'standard', cat_strategy: str = 'ohe') -> "Encoder":
        self._num_strategy = num_strategy
        self._cat_strategy = cat_strategy
        return self

    def fit_transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        self._num_cols = df.select_dtypes(include='number').columns.tolist()
        self._cat_cols = df.select_dtypes(exclude='number').columns.tolist()
        return pd.concat(self._fit_transform_parts(df), axis=1)

    def transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.concat(self._transform_parts(df), axis=1)

    def encode_target(self, y, fit: bool = False):
        if fit:
            return self._target_le.fit_transform(y)
        return self._target_le.transform(y)

    def _fit_transform_parts(self, df: pd.DataFrame) -> list:
        parts = []
        if self._num_cols:
            if self._num_strategy == 'standard':
                self._num_scaler = StandardScaler()
            elif self._num_strategy == 'minmax':
                self._num_scaler = MinMaxScaler()
            data = self._num_scaler.fit_transform(df[self._num_cols]) if self._num_scaler else df[self._num_cols].values
            parts.append(pd.DataFrame(data, columns=self._num_cols, index=df.index))
        if self._cat_cols:
            if self._cat_strategy == 'ohe':
                self._cat_encoder = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)
                data = self._cat_encoder.fit_transform(df[self._cat_cols])
                names = self._cat_encoder.get_feature_names_out(self._cat_cols)
            else:
                self._cat_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                data = self._cat_encoder.fit_transform(df[self._cat_cols])
                names = self._cat_cols
            parts.append(pd.DataFrame(data, columns=names, index=df.index))
        return parts if parts else [df.copy()]

    def _transform_parts(self, df: pd.DataFrame) -> list:
        parts = []
        if self._num_cols:
            data = self._num_scaler.transform(df[self._num_cols]) if self._num_scaler else df[self._num_cols].values
            parts.append(pd.DataFrame(data, columns=self._num_cols, index=df.index))
        if self._cat_cols:
            if self._cat_strategy == 'ohe':
                data = self._cat_encoder.transform(df[self._cat_cols])
                names = self._cat_encoder.get_feature_names_out(self._cat_cols)
            else:
                data = self._cat_encoder.transform(df[self._cat_cols])
                names = self._cat_cols
            parts.append(pd.DataFrame(data, columns=names, index=df.index))
        return parts if parts else [df.copy()]
