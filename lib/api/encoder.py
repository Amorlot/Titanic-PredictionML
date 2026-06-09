from flask import Blueprint, request, jsonify
from lib.encoder import GenericEncoder

encoder_bp = Blueprint('encoder', __name__, url_prefix='/encoder')

_encoder: GenericEncoder = None


@encoder_bp.post('/configure')
def configure():
    """
    Body JSON:
      { "num_strategy": "standard", "cat_strategy": "ohe" }
    """
    global _encoder
    body = request.get_json(force=True)
    _encoder = GenericEncoder()
    _encoder.configure(
        num_strategy=body.get('num_strategy', 'standard'),
        cat_strategy=body.get('cat_strategy', 'ohe'),
    )
    return jsonify({'status': 'encoder configurato'}), 200


@encoder_bp.post('/fit-transform')
def fit_transform():
    """
    Body JSON: { "data": [ {col: val, ...}, ... ] }
    Ritorna le feature encoded + l'encoder fittato per trasformazioni future.
    """
    global _encoder
    import pandas as pd
    body = request.get_json(force=True)
    if _encoder is None:
        _encoder = GenericEncoder()
        _encoder.configure()
    df = pd.DataFrame(body['data'])
    df_enc = _encoder.fit_transform_features(df)
    return jsonify(df_enc.to_dict(orient='records')), 200


@encoder_bp.post('/transform')
def transform():
    """
    Body JSON: { "data": [ {col: val, ...}, ... ] }
    """
    import pandas as pd
    body = request.get_json(force=True)
    if _encoder is None:
        return jsonify({'error': 'Encoder non ancora fittato. Chiama prima POST /encoder/fit-transform'}), 400
    df = pd.DataFrame(body['data'])
    df_enc = _encoder.transform_features(df)
    return jsonify(df_enc.to_dict(orient='records')), 200
