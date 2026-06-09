from flask import Blueprint, request, jsonify
from lib.loader import GenericLoader

loader_bp = Blueprint('loader', __name__, url_prefix='/loader')

_loader: GenericLoader = None


@loader_bp.post('/load')
def load():
    """
    Body JSON:
      { "csv_path": "data/train.csv", "target_col": "Survived",
        "drop_cols": [...], "drop_missing_thresh": 0.6 }
    """
    global _loader
    body = request.get_json(force=True)

    csv_path    = body.get('csv_path')
    target_col  = body.get('target_col')
    drop_cols   = body.get('drop_cols', [])
    thresh      = body.get('drop_missing_thresh')

    if not csv_path or not target_col:
        return jsonify({'error': 'csv_path e target_col sono obbligatori'}), 400

    _loader = GenericLoader(
        target_col=target_col,
        csv_path=csv_path,
        drop_cols=drop_cols,
        drop_missing_thresh=thresh,
    )
    _loader.load()
    return jsonify(_loader.info()), 200


@loader_bp.get('/info')
def info():
    if _loader is None:
        return jsonify({'error': 'Dataset non caricato. Chiama prima POST /loader/load'}), 400
    return jsonify(_loader.info()), 200


@loader_bp.get('/missing')
def missing():
    if _loader is None:
        return jsonify({'error': 'Dataset non caricato. Chiama prima POST /loader/load'}), 400
    return jsonify(_loader.missing_report()), 200
