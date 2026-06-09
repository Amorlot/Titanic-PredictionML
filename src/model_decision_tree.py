from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from src.model_base import AbstractModel

_DEFAULT_PARAM_GRID = {
    'max_depth':         [None, 3, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'criterion':         ['gini', 'entropy'],
    'min_samples_leaf':  [1, 2, 4],
}


class DecisionTreeModel(AbstractModel):

    def train(self, X_train, y_train, cv: int = 5, scoring: str = 'f1_weighted', param_grid: dict = None):
        grid = GridSearchCV(
            estimator=DecisionTreeClassifier(random_state=42),
            param_grid=param_grid or _DEFAULT_PARAM_GRID,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)
        self.model = grid.best_estimator_
        self.best_params = grid.best_params_
        return self.model
