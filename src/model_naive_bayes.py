from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from models.base import AbstractModel

_DEFAULT_PARAM_GRID = {
    'var_smoothing': [1e-11, 1e-10, 1e-9, 1e-8, 1e-7],
}


class NaiveBayesModel(AbstractModel):

    def train(self, X_train, y_train, cv: int = 5, scoring: str = 'f1_weighted', param_grid: dict = None):
        grid = GridSearchCV(
            estimator=GaussianNB(),
            param_grid=param_grid or _DEFAULT_PARAM_GRID,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)
        self.model = grid.best_estimator_
        self.best_params = grid.best_params_
        return self.model
