from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from models.base import AbstractModel

_DEFAULT_PARAM_GRID = [
    {'kernel': ['rbf'],    'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto', 0.01, 0.1]},
    {'kernel': ['linear'], 'C': [0.1, 1, 10, 100]},
    {'kernel': ['poly'],   'C': [0.1, 1, 10],       'degree': [2, 3]},
]

class GenericSVM(AbstractModel):
    def train(self, X_train, y_train, cv=5, scoring='f1_weighted', param_grid=None):
        grid = GridSearchCV(
            estimator=SVC(random_state=42, probability=True),
            param_grid=param_grid or _DEFAULT_PARAM_GRID,
            cv=cv, scoring=scoring, n_jobs=-1,
        )
        grid.fit(X_train, y_train)
        self.model = grid.best_estimator_
        self.best_params = grid.best_params_
        self.best_score = grid.best_score_
        return self.model
