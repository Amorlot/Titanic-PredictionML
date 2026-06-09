from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from models.base import AbstractModel

_DEFAULT_PARAM_GRID = {
    'n_neighbors': [3, 5, 7, 9, 11, 15],
    'weights':     ['uniform', 'distance'],
    'metric':      ['euclidean', 'manhattan', 'minkowski'],
}


class KNNModel(AbstractModel):

    def train(self, X_train, y_train, cv: int = 5, scoring: str = 'f1_weighted', param_grid: dict = None):
        grid = GridSearchCV(
            estimator=KNeighborsClassifier(),
            param_grid=param_grid or _DEFAULT_PARAM_GRID,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)
        self.model = grid.best_estimator_
        self.best_params = grid.best_params_
        return self.model
