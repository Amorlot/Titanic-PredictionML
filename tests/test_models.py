import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from src import (
    Logreg, ModelXGBoost,
    DecisionTreeModel, SVMModel,
)

X, y = make_classification(n_samples=400, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

_MODELS = {
    "LogisticRegression": Logreg(),
    "XGBoost":            ModelXGBoost(),
    "DecisionTree":       DecisionTreeModel(),
    "SVM":                SVMModel(),
}


def _test_model(name, model):
    model.train(X_train, y_train)
    assert model.model is not None, f"{name}: model is None dopo train"
    assert model.best_params is not None, f"{name}: best_params is None"

    metrics = model.evaluate(X_test, y_test)
    assert set(metrics.keys()) == {'accuracy', 'precision', 'recall', 'f1_score'}
    assert all(0.0 <= v <= 1.0 for v in metrics.values())

    report = model.classification_report(X_test, y_test)
    assert isinstance(report, str) and 'precision' in report
    print(f"  {name}: OK  f1={metrics['f1_score']:.4f}")


def test_all_models():
    print()
    for name, model in _MODELS.items():
        _test_model(name, model)


if __name__ == "__main__":
    test_all_models()
