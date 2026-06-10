from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from lib.models.base import AbstractModel

_DEFAULT_PARAM_GRID = [
    {'penalty': ['l1'], 'C': [0.001, 0.01, 0.1, 1, 10, 100], 'solver': ['liblinear']},
    {'penalty': ['l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100], 'solver': ['lbfgs', 'liblinear']},
    {'penalty': ['elasticnet'], 'C': [0.001, 0.01, 0.1, 1, 10], 'solver': ['saga'], 'l1_ratio': [0.2, 0.5, 0.8]},
]

class GenericLogreg(AbstractModel):
    def train(self, X_train, y_train, cv=5, scoring='f1_weighted', random_state=42, param_grid=None):
        grid = GridSearchCV(
            estimator=LogisticRegression(max_iter=5000, random_state=random_state),
            param_grid=param_grid or _DEFAULT_PARAM_GRID,
            cv=cv, scoring=scoring, n_jobs=-1,
        )
        grid.fit(X_train, y_train)
        self.model = grid.best_estimator_
        self.best_params = grid.best_params_
        self.best_score = grid.best_score_
        return self.model
