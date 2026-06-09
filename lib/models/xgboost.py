from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from models.base import AbstractModel

_DEFAULT_PARAM_GRID = {
    'n_estimators':     [100, 200, 500],
    'max_depth':        [3, 5, 7],
    'learning_rate':    [0.01, 0.1, 0.3],
    'subsample':        [0.8, 1.0],
    'min_child_weight': [1, 5],
}


class GenericXGBoost(AbstractModel):

    def train(
        self,
        X_train,
        y_train,
        cv: int = 5,
        scoring: str = 'f1_weighted',
        param_grid: dict = None,
    ):
        grid = GridSearchCV(
            estimator=XGBClassifier(random_state=42, eval_metric='logloss'),
            param_grid=param_grid or _DEFAULT_PARAM_GRID,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=0,
        )
        grid.fit(X_train, y_train)
        self.model = grid.best_estimator_
        self.best_params = grid.best_params_
        return self.model
