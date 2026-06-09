from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from lib.models.base import AbstractModel

_DEFAULT_PARAM_GRID = {
    'max_depth':         [None, 3, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'criterion':         ['gini', 'entropy'],
    'min_samples_leaf':  [1, 2, 4],
}

class GenericDecisionTree(AbstractModel):
    def train(self, X_train, y_train, cv=5, scoring='f1_weighted', random_state=42, param_grid=None):
        grid = GridSearchCV(
            estimator=DecisionTreeClassifier(random_state=random_state),
            param_grid=param_grid or _DEFAULT_PARAM_GRID,
            cv=cv, scoring=scoring, n_jobs=-1,
        )
        grid.fit(X_train, y_train)
        self.model = grid.best_estimator_
        self.best_params = grid.best_params_
        self.best_score = grid.best_score_
        return self.model
