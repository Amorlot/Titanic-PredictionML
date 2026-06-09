from flask import Blueprint, request, jsonify
from lib.cleaner import GenericCleaner

cleaner_bp = Blueprint('cleaner', __name__, url_prefix='/cleaner')

_cleaner: GenericCleaner = None


@cleaner_bp.post('/configure')
def configure():
    """
    Body JSON:
      { "num_strategy": "median", "cat_strategy": "most_frequent" }
    """
    global _cleaner
    body = request.get_json(force=True)
    _cleaner = GenericCleaner()
    _cleaner.configure(
        num_strategy=body.get('num_strategy', 'median'),
        cat_strategy=body.get('cat_strategy', 'most_frequent'),
    )
    return jsonify({'status': 'cleaner configurato'}), 200


@cleaner_bp.post('/fit-transform')
def fit_transform():
    """
    Body JSON: { "data": [ {col: val, ...}, ... ] }
    Ritorna i dati puliti.
    """
    global _cleaner
    import pandas as pd
    body = request.get_json(force=True)
    if _cleaner is None:
        _cleaner = GenericCleaner()
        _cleaner.configure()
    df = pd.DataFrame(body['data'])
    df_clean = _cleaner.fit_transform(df)
    return jsonify(df_clean.to_dict(orient='records')), 200


@cleaner_bp.post('/transform')
def transform():
    """
    Body JSON: { "data": [ {col: val, ...}, ... ] }
    Applica il cleaner già fittato.
    """
    import pandas as pd
    body = request.get_json(force=True)
    if _cleaner is None:
        return jsonify({'error': 'Cleaner non ancora fittato. Chiama prima POST /cleaner/fit-transform'}), 400
    df = pd.DataFrame(body['data'])
    df_clean = _cleaner.transform(df)
    return jsonify(df_clean.to_dict(orient='records')), 200
