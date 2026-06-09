from flask import Blueprint, jsonify
from lib.api.loader import _loader
from lib.eda import GenericEda

eda_bp = Blueprint('eda', __name__, url_prefix='/eda')


def _get_eda() -> GenericEda:
    from lib.api import loader as loader_module
    if loader_module._loader is None:
        return None
    return GenericEda(loader_module._loader.df)


@eda_bp.get('/summary')
def summary():
    eda = _get_eda()
    if eda is None:
        return jsonify({'error': 'Dataset non caricato. Chiama prima POST /loader/load'}), 400
    return jsonify(eda.summary()), 200


@eda_bp.get('/missing')
def missing():
    eda = _get_eda()
    if eda is None:
        return jsonify({'error': 'Dataset non caricato. Chiama prima POST /loader/load'}), 400
    return jsonify(eda.missing_report()), 200


@eda_bp.get('/correlation')
def correlation():
    eda = _get_eda()
    if eda is None:
        return jsonify({'error': 'Dataset non caricato. Chiama prima POST /loader/load'}), 400
    return jsonify(eda.correlation()), 200


@eda_bp.get('/balance')
def balance():
    eda = _get_eda()
    if eda is None:
        return jsonify({'error': 'Dataset non caricato. Chiama prima POST /loader/load'}), 400
    from lib.api import loader as loader_module
    target_col = loader_module._loader.target_col
    y = loader_module._loader.df[target_col]
    return jsonify(eda.class_balance(y)), 200
