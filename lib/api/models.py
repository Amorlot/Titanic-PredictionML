from flask import Blueprint, request, jsonify
import pandas as pd

from lib.models.logreg import GenericLogreg
from lib.models.xgboost import GenericXGBoost
from lib.models.decision_tree import GenericDecisionTree
from lib.models.random_forest import GenericRandomForest
from lib.models.svm import GenericSVM

models_bp = Blueprint('models', __name__, url_prefix='/models')

_AVAILABLE = {
    'logreg':        GenericLogreg,
    'xgboost':       GenericXGBoost,
    'decision_tree': GenericDecisionTree,
    'random_forest': GenericRandomForest,
    'svm':           GenericSVM,
}

_trained: dict = {}


@models_bp.get('/')
def list_models():
    """Elenca i modelli disponibili e quali sono già stati addestrati."""
    return jsonify({
        'available': list(_AVAILABLE.keys()),
        'trained':   list(_trained.keys()),
    }), 200


@models_bp.post('/train')
def train():
    """
    Body JSON:
      { "model": "xgboost",
        "X_train": [ {col: val, ...}, ... ],
        "y_train": [0, 1, ...],
        "cv": 5,
        "scoring": "f1_weighted" }
    """
    body = request.get_json(force=True)
    model_name = body.get('model')

    if model_name not in _AVAILABLE:
        return jsonify({'error': f'Modello non disponibile. Scegli tra: {list(_AVAILABLE.keys())}'}), 400
    if 'X_train' not in body or 'y_train' not in body:
        return jsonify({'error': 'X_train e y_train sono obbligatori'}), 400

    X_train = pd.DataFrame(body['X_train'])
    y_train = body['y_train']

    model = _AVAILABLE[model_name]()
    model.train(X_train, y_train,
                cv=body.get('cv', 5),
                scoring=body.get('scoring', 'f1_weighted'))

    _trained[model_name] = model
    return jsonify({
        'model':       model_name,
        'best_params': model.best_params,
        'best_score':  round(model.best_score, 4),
        'status':      'addestrato',
    }), 200


@models_bp.post('/evaluate')
def evaluate():
    """
    Body JSON:
      { "model": "xgboost",
        "X_test": [ {col: val, ...}, ... ],
        "y_test": [0, 1, ...] }
    """
    body = request.get_json(force=True)
    model_name = body.get('model')

    if model_name not in _trained:
        return jsonify({'error': f'Modello "{model_name}" non addestrato. Chiama prima POST /models/train'}), 400

    X_test = pd.DataFrame(body['X_test'])
    y_test = body['y_test']

    metrics = _trained[model_name].evaluate(X_test, y_test)
    return jsonify({'model': model_name, 'metrics': metrics}), 200


@models_bp.post('/predict')
def predict():
    """
    Body JSON:
      { "model": "xgboost",
        "data": [ {col: val, ...}, ... ] }
    Ritorna: { "predictions": [0, 1, ...] }
    """
    body = request.get_json(force=True)
    model_name = body.get('model')

    if model_name not in _trained:
        return jsonify({'error': f'Modello "{model_name}" non addestrato. Chiama prima POST /models/train'}), 400

    X = pd.DataFrame(body['data'])
    preds = _trained[model_name].predict(X).tolist()
    return jsonify({'model': model_name, 'predictions': preds}), 200
