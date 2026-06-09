from flask import Blueprint, request, jsonify
from lib.splitter import GenericSplitter

splitter_bp = Blueprint('splitter', __name__, url_prefix='/splitter')


@splitter_bp.post('/split')
def split():
    """
    Body JSON:
      { "data": [ {col: val, ...}, ... ],
        "target_col": "Survived",
        "test_size": 0.2,
        "stratify": true }
    Ritorna: { "train": [...], "test": [...], "y_train": [...], "y_test": [...] }
    """
    import pandas as pd
    body = request.get_json(force=True)

    target_col = body.get('target_col')
    if not target_col:
        return jsonify({'error': 'target_col è obbligatorio'}), 400

    df = pd.DataFrame(body['data'])
    X = df.drop(columns=[target_col])
    y = df[target_col]

    splitter = GenericSplitter(
        test_size=body.get('test_size', 0.2),
        random_state=body.get('random_state', 42),
        stratify=body.get('stratify', True),
    )
    X_train, X_test, y_train, y_test = splitter.split(X, y)

    return jsonify({
        'train':   X_train.to_dict(orient='records'),
        'test':    X_test.to_dict(orient='records'),
        'y_train': y_train.tolist(),
        'y_test':  y_test.tolist(),
        'train_size': len(X_train),
        'test_size':  len(X_test),
    }), 200
