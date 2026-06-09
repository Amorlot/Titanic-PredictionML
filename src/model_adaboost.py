from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from models.base import AbstractModel

_DEFAULT_PARAM_GRID = {
    'n_estimators':  [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.5, 1.0],
    'estimator__max_depth': [1, 2, 3],
}


class AdaBoostModel(AbstractModel):

    def train(self, X_train, y_train, cv: int = 5, scoring: str = 'f1_weighted', param_grid: dict = None):
        grid = GridSearchCV(
            estimator=AdaBoostClassifier(
                estimator=DecisionTreeClassifier(),
                random_state=42,
                algorithm='SAMME',
            ),
            param_grid=param_grid or _DEFAULT_PARAM_GRID,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)
        self.model = grid.best_estimator_
        self.best_params = grid.best_params_
        return self.model
