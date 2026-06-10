import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app import app as flask_app
import lib.api.loader as loader_module
import lib.api.cleaner as cleaner_module
import lib.api.encoder as encoder_module
import lib.api.models as models_module

TRAIN_CSV = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'train.csv'))

DIRTY_DATA = [
    {'Age': 25.0, 'Fare': 10.0, 'Sex': 'male',   'Pclass': 1},
    {'Age': None, 'Fare': 20.0, 'Sex': 'female',  'Pclass': 2},
    {'Age': 35.0, 'Fare': None, 'Sex': 'male',    'Pclass': 3},
    {'Age': 30.0, 'Fare': 25.0, 'Sex': None,      'Pclass': 1},
    {'Age': 28.0, 'Fare': 15.0, 'Sex': 'female',  'Pclass': 2},
]

CLEAN_DATA = [{'f0': float(i), 'f1': float(i * 2)} for i in range(1, 11)]

SPLIT_DATA = [
    {'Age': float(i), 'Fare': float(i * 2), 'Survived': i % 2}
    for i in range(1, 51)
]

_rng = np.random.default_rng(42)
_X_raw = _rng.random((100, 3))
MODEL_X = [{'f0': float(r[0]), 'f1': float(r[1]), 'f2': float(r[2])} for r in _X_raw]
MODEL_Y = [int(_rng.integers(0, 2)) for _ in range(100)]


@pytest.fixture
def client():
    flask_app.config['TESTING'] = True
    with flask_app.test_client() as c:
        yield c


@pytest.fixture(autouse=True)
def reset_state():
    loader_module._loader = None
    cleaner_module._cleaner = None
    encoder_module._encoder = None
    models_module._trained.clear()
    yield
    loader_module._loader = None
    cleaner_module._cleaner = None
    encoder_module._encoder = None
    models_module._trained.clear()


@pytest.fixture
def loaded_client(client):
    client.post('/loader/load', json={
        'csv_path': TRAIN_CSV,
        'target_col': 'Survived',
        'drop_cols': ['PassengerId', 'Name', 'Ticket', 'Cabin'],
    })
    return client


@pytest.fixture
def trained_client(client):
    client.post('/models/train', json={
        'model': 'logreg',
        'X_train': MODEL_X,
        'y_train': MODEL_Y,
        'cv': 2,
    })
    return client


# ──────────────────────────────── /loader ────────────────────────────────

class TestLoader:
    def test_load_returns_row_count(self, client):
        res = client.post('/loader/load', json={
            'csv_path': TRAIN_CSV,
            'target_col': 'Survived',
        })
        assert res.status_code == 200
        assert res.get_json()['rows'] == 891

    def test_load_missing_csv_path_is_400(self, client):
        res = client.post('/loader/load', json={'target_col': 'Survived'})
        assert res.status_code == 400

    def test_load_missing_target_col_is_400(self, client):
        res = client.post('/loader/load', json={'csv_path': TRAIN_CSV})
        assert res.status_code == 400

    def test_info_before_load_is_400(self, client):
        assert client.get('/loader/info').status_code == 400

    def test_info_after_load(self, loaded_client):
        res = loaded_client.get('/loader/info')
        assert res.status_code == 200
        assert 'rows' in res.get_json()

    def test_missing_before_load_is_400(self, client):
        assert client.get('/loader/missing').status_code == 400

    def test_missing_after_load_is_dict(self, loaded_client):
        res = loaded_client.get('/loader/missing')
        assert res.status_code == 200
        assert isinstance(res.get_json(), dict)


# ──────────────────────────────── /eda ────────────────────────────────

class TestEda:
    def test_summary_before_load_is_400(self, client):
        assert client.get('/eda/summary').status_code == 400

    def test_summary_after_load(self, loaded_client):
        assert loaded_client.get('/eda/summary').status_code == 200

    def test_missing_after_load(self, loaded_client):
        assert loaded_client.get('/eda/missing').status_code == 200

    def test_correlation_returns_dict(self, loaded_client):
        res = loaded_client.get('/eda/correlation')
        assert res.status_code == 200
        assert isinstance(res.get_json(), dict)

    def test_balance_returns_dict(self, loaded_client):
        res = loaded_client.get('/eda/balance')
        assert res.status_code == 200
        assert isinstance(res.get_json(), dict)


# ──────────────────────────────── /cleaner ────────────────────────────────

class TestCleaner:
    def test_configure(self, client):
        res = client.post('/cleaner/configure', json={
            'num_strategy': 'median',
            'cat_strategy': 'most_frequent',
        })
        assert res.status_code == 200

    def test_fit_transform_fills_nulls(self, client):
        res = client.post('/cleaner/fit-transform', json={'data': DIRTY_DATA})
        assert res.status_code == 200
        result = res.get_json()
        assert len(result) == len(DIRTY_DATA)
        for row in result:
            assert all(v is not None for v in row.values())

    def test_transform_before_fit_is_400(self, client):
        assert client.post('/cleaner/transform', json={'data': DIRTY_DATA}).status_code == 400

    def test_transform_after_fit(self, client):
        client.post('/cleaner/fit-transform', json={'data': DIRTY_DATA})
        res = client.post('/cleaner/transform', json={'data': DIRTY_DATA})
        assert res.status_code == 200
        assert len(res.get_json()) == len(DIRTY_DATA)


# ──────────────────────────────── /encoder ────────────────────────────────

class TestEncoder:
    def test_configure(self, client):
        res = client.post('/encoder/configure', json={
            'num_strategy': 'standard',
            'cat_strategy': 'ohe',
        })
        assert res.status_code == 200

    def test_fit_transform(self, client):
        res = client.post('/encoder/fit-transform', json={'data': CLEAN_DATA})
        assert res.status_code == 200
        assert len(res.get_json()) == len(CLEAN_DATA)

    def test_transform_before_fit_is_400(self, client):
        assert client.post('/encoder/transform', json={'data': CLEAN_DATA}).status_code == 400

    def test_transform_after_fit(self, client):
        client.post('/encoder/fit-transform', json={'data': CLEAN_DATA})
        res = client.post('/encoder/transform', json={'data': CLEAN_DATA})
        assert res.status_code == 200
        assert len(res.get_json()) == len(CLEAN_DATA)


# ──────────────────────────────── /splitter ────────────────────────────────

class TestSplitter:
    def test_split_returns_correct_sizes(self, client):
        res = client.post('/splitter/split', json={
            'data': SPLIT_DATA,
            'target_col': 'Survived',
            'test_size': 0.2,
        })
        assert res.status_code == 200
        data = res.get_json()
        assert data['train_size'] + data['test_size'] == len(SPLIT_DATA)
        assert len(data['train']) == data['train_size']
        assert len(data['y_train']) == data['train_size']

    def test_split_missing_target_is_400(self, client):
        assert client.post('/splitter/split', json={'data': SPLIT_DATA}).status_code == 400


# ──────────────────────────────── /models ────────────────────────────────

class TestModels:
    def test_list_available_models(self, client):
        res = client.get('/models/')
        assert res.status_code == 200
        data = res.get_json()
        assert set(data['available']) == {'logreg', 'xgboost', 'decision_tree', 'random_forest', 'svm'}
        assert data['trained'] == []

    def test_train_invalid_model_is_400(self, client):
        res = client.post('/models/train', json={
            'model': 'nonexistent',
            'X_train': MODEL_X,
            'y_train': MODEL_Y,
        })
        assert res.status_code == 400

    def test_train_missing_data_is_400(self, client):
        assert client.post('/models/train', json={'model': 'logreg'}).status_code == 400

    def test_train_logreg_ok(self, client):
        res = client.post('/models/train', json={
            'model': 'logreg',
            'X_train': MODEL_X,
            'y_train': MODEL_Y,
            'cv': 2,
        })
        assert res.status_code == 200
        data = res.get_json()
        assert data['model'] == 'logreg'
        assert 0.0 <= data['best_score'] <= 1.0

    def test_evaluate_before_train_is_400(self, client):
        res = client.post('/models/evaluate', json={
            'model': 'logreg',
            'X_test': MODEL_X[:20],
            'y_test': MODEL_Y[:20],
        })
        assert res.status_code == 400

    def test_evaluate_after_train(self, trained_client):
        res = trained_client.post('/models/evaluate', json={
            'model': 'logreg',
            'X_test': MODEL_X[:20],
            'y_test': MODEL_Y[:20],
        })
        assert res.status_code == 200
        data = res.get_json()
        assert 'metrics' in data
        assert 'f1_score' in data['metrics']

    def test_predict_before_train_is_400(self, client):
        res = client.post('/models/predict', json={
            'model': 'logreg',
            'data': MODEL_X[:5],
        })
        assert res.status_code == 400

    def test_predict_after_train(self, trained_client):
        res = trained_client.post('/models/predict', json={
            'model': 'logreg',
            'data': MODEL_X[:5],
        })
        assert res.status_code == 200
        assert len(res.get_json()['predictions']) == 5

    def test_trained_list_updated_after_train(self, trained_client):
        res = trained_client.get('/models/')
        assert 'logreg' in res.get_json()['trained']
