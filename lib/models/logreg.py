from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from models.base import AbstractModel

_DEFAULT_PARAM_GRID = [
    {
        'penalty': ['l1'],
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear'],
    },
    {
        'penalty': ['l2'],
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'solver': ['lbfgs', 'liblinear'],
    },
    {
        'penalty': ['elasticnet'],
        'C': [0.001, 0.01, 0.1, 1, 10],
        'solver': ['saga'],
        'l1_ratio': [0.2, 0.5, 0.8],
    },
]


class GenericLogreg(AbstractModel):

    def train(
        self,
        X_train,
        y_train,
        cv: int = 5,
        scoring: str = 'f1_weighted',
        param_grid: list = None,
    ):
        grid = GridSearchCV(
            estimator=LogisticRegression(max_iter=5000, random_state=42),
            param_grid=param_grid or _DEFAULT_PARAM_GRID,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)
        self.model = grid.best_estimator_
        self.best_params = grid.best_params_
        return self.model
