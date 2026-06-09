import numpy as np
import pandas as pd
import pytest
from lib.cleaner import GenericCleaner
from lib.outliers import GenericOutlierHandler
from lib.multicollinearity import MulticollinearityHandler
from lib.pca_reducer import PCAReducer

# ──────────────────────────────────────────────
# GenericCleaner – KNN imputation
# ──────────────────────────────────────────────

def make_dirty_df():
    return pd.DataFrame({
        'Age':    [25.0, None, 35.0, 30.0, None],
        'Fare':   [10.0, 20.0, None, 40.0, 50.0],
        'Sex':    ['male', None, 'female', 'male', 'female'],
        'Pclass': [1, 2, 3, 1, 2],
    })


def test_knn_imputation_no_missing():
    df = make_dirty_df()
    cleaner = GenericCleaner()
    cleaner.configure(num_strategy='knn', cat_strategy='knn', knn_neighbors=2)
    result = cleaner.fit_transform(df)
    assert result.isnull().sum().sum() == 0


def test_median_imputation():
    df = make_dirty_df()
    cleaner = GenericCleaner()
    cleaner.configure(num_strategy='median', cat_strategy='most_frequent')
    result = cleaner.fit_transform(df)
    assert result.isnull().sum().sum() == 0


def test_cleaner_transform_after_fit():
    df = make_dirty_df()
    cleaner = GenericCleaner()
    cleaner.configure(num_strategy='knn', cat_strategy='knn', knn_neighbors=2)
    cleaner.fit_transform(df)
    df2 = make_dirty_df()
    result = cleaner.transform(df2)
    assert result.isnull().sum().sum() == 0


# ──────────────────────────────────────────────
# GenericOutlierHandler
# ──────────────────────────────────────────────

def make_outlier_df():
    rng = np.random.default_rng(42)
    normal = rng.normal(loc=10, scale=1, size=100)
    normal[0] = 100.0   # extreme outlier
    normal[1] = -80.0   # extreme outlier
    return pd.DataFrame({'x': normal, 'y': rng.normal(0, 1, 100)})


def test_outlier_fit_and_transform():
    df = make_outlier_df()
    handler = GenericOutlierHandler()
    handler.configure(detection='iqr', treatment='winsorization', iqr_threshold=1.5)
    handler.fit(df)
    result = handler.transform(df)
    assert result['x'].max() < 100.0
    assert result['x'].min() > -80.0


def test_outlier_report_after_fit():
    df = make_outlier_df()
    handler = GenericOutlierHandler()
    handler.configure(detection='iqr', treatment='winsorization', iqr_threshold=1.5, cols=['x'])
    handler.fit(df)
    report = handler.report(df)
    assert 'x' in report
    assert report['x']['outliers'] >= 1


def test_outlier_combined_iqr_zscore():
    df = make_outlier_df()
    handler = GenericOutlierHandler()
    handler.configure(detection=['iqr', 'zscore'], treatment='winsorization',
                      iqr_threshold=1.5, zscore_threshold=3.0)
    handler.fit(df)
    result = handler.transform(df)
    assert result.isnull().sum().sum() == 0


# ──────────────────────────────────────────────
# MulticollinearityHandler
# ──────────────────────────────────────────────

def make_corr_df():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    return pd.DataFrame({
        'x':       x,
        'x_clone': x + rng.normal(0, 0.01, 200),  # nearly identical → |r| ≈ 1
        'z':       rng.normal(size=200),
    })


def test_multicollinearity_detects_high_corr():
    df = make_corr_df()
    handler = MulticollinearityHandler()
    handler.configure(threshold=0.85, drop=True)
    handler.fit(df)
    report = handler.report()
    assert len(report['pairs']) >= 1


def test_multicollinearity_drops_column():
    df = make_corr_df()
    handler = MulticollinearityHandler()
    handler.configure(threshold=0.85, drop=True)
    handler.fit(df)
    result = handler.transform(df)
    assert result.shape[1] < df.shape[1]


def test_multicollinearity_report_only():
    df = make_corr_df()
    handler = MulticollinearityHandler()
    handler.configure(threshold=0.85, drop=False)
    handler.fit(df)
    result = handler.transform(df)
    assert result.shape[1] == df.shape[1]


# ──────────────────────────────────────────────
# PCAReducer
# ──────────────────────────────────────────────

def make_pca_df():
    rng = np.random.default_rng(7)
    return pd.DataFrame(rng.normal(size=(100, 8)), columns=[f'f{i}' for i in range(8)])


def test_pca_reduces_dimensions():
    df = make_pca_df()
    reducer = PCAReducer()
    reducer.configure(n_components=0.95)
    reducer.fit(df)
    result = reducer.transform(df)
    assert result.shape[1] < df.shape[1]
    assert result.shape[0] == df.shape[0]
    assert all(c.startswith('PC') for c in result.columns)


def test_pca_fixed_n_components():
    df = make_pca_df()
    reducer = PCAReducer()
    reducer.configure(n_components=3)
    reducer.fit(df)
    result = reducer.transform(df)
    assert result.shape[1] == 3


def test_pca_report():
    df = make_pca_df()
    reducer = PCAReducer()
    reducer.configure(n_components=0.90)
    reducer.fit(df)
    report = reducer.report()
    assert 'n_components_in' in report
    assert 'n_components_out' in report
    assert 'variance_explained' in report
    assert 0.0 <= report['variance_explained'] <= 1.0
