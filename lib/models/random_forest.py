from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from models.base import AbstractModel

_DEFAULT_PARAM_GRID = {
    'n_estimators':      [100, 200, 500],
    'max_depth':         [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'max_features':      ['sqrt', 'log2'],
}


class GenericRandomForest(AbstractModel):

    def train(self, X_train, y_train, cv: int = 5, scoring: str = 'f1_weighted', param_grid: dict = None):
        grid = GridSearchCV(
            estimator=RandomForestClassifier(random_state=42),
            param_grid=param_grid or _DEFAULT_PARAM_GRID,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)
        self.model = grid.best_estimator_
        self.best_params = grid.best_params_
        return self.model
