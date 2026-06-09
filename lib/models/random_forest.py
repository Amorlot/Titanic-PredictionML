from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from models.base import AbstractModel

_DEFAULT_PARAM_GRID = {
    'n_estimators':      [100, 200],
    'max_depth':         [5, 10],
    'min_samples_split': [2, 5],
    'max_features':      ['sqrt', 'log2'],
}

class GenericRandomForest(AbstractModel):
    def train(self, X_train, y_train, cv=5, scoring='f1_weighted', param_grid=None):
        grid = GridSearchCV(
            estimator=RandomForestClassifier(random_state=42, n_jobs=1),
            param_grid=param_grid or _DEFAULT_PARAM_GRID,
            cv=cv, scoring=scoring, n_jobs=-1,
        )
        grid.fit(X_train, y_train)
        self.model = grid.best_estimator_
        self.best_params = grid.best_params_
        self.best_score = grid.best_score_
        return self.model
