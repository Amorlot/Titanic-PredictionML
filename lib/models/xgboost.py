import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from lib.models.base import AbstractModel

_DEFAULT_PARAM_GRID = {
    'n_estimators':     [100, 200],
    'max_depth':        [3, 5],
    'learning_rate':    [0.05, 0.1],
    'subsample':        [0.8, 1.0],
    'min_child_weight': [1, 3],
}

class GenericXGBoost(AbstractModel):
    def train(self, X_train, y_train, cv=5, scoring='f1_weighted', random_state=42, param_grid=None):
        grid = GridSearchCV(
            estimator=XGBClassifier(random_state=random_state, eval_metric='logloss',
                                    nthread=1, verbosity=0),
            param_grid=param_grid or _DEFAULT_PARAM_GRID,
            cv=cv, scoring=scoring, n_jobs=-1,
        )
        with joblib.parallel_backend('threading'):
            grid.fit(X_train, y_train)
        self.model = grid.best_estimator_
        self.best_params = grid.best_params_
        self.best_score = grid.best_score_
        return self.model
