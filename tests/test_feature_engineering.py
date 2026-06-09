import pandas as pd
import pytest
from lib.feature_engineering import FeatureEngineer

SAMPLE = pd.DataFrame({
    'Name':   ['Doe, Mr. John', 'Smith, Mrs. Jane', 'Brown, Miss. Alice',
               'Jones, Master. Tom', 'White, Dr. Bob'],
    'Pclass': [1, 2, 3, 3, 1],
    'Fare':   [100.0, 30.0, 8.0, 7.5, 90.0],
    'SibSp':  [0, 1, 0, 2, 0],
    'Parch':  [0, 0, 1, 1, 0],
    'Age':    [35.0, 25.0, 10.0, 5.0, 45.0],
})


def make_fe(drop_fare_for=None):
    fe = FeatureEngineer()
    fe.configure(drop_raw_fare_for=drop_fare_for or [])
    fe.fit(SAMPLE)
    return fe


def test_title_group_extraction():
    fe = make_fe()
    out = fe.transform(SAMPLE.copy())
    assert 'TitleGroup' in out.columns
    assert 'Name' not in out.columns
    assert out.loc[0, 'TitleGroup'] == 'Mr'
    assert out.loc[2, 'TitleGroup'] == 'Miss'
    assert out.loc[4, 'TitleGroup'] == 'Officer'


def test_fare_per_person():
    fe = make_fe()
    out = fe.transform(SAMPLE.copy())
    assert 'FarePerPerson' in out.columns
    # Row 0: Fare=100, SibSp=0, Parch=0 → FarePerPerson = 100/1 = 100
    assert abs(out.loc[0, 'FarePerPerson'] - 100.0) < 1e-6
    # Row 1: Fare=30, SibSp=1, Parch=0 → FarePerPerson = 30/2 = 15
    assert abs(out.loc[1, 'FarePerPerson'] - 15.0) < 1e-6


def test_age_group_bins():
    fe = make_fe()
    out = fe.transform(SAMPLE.copy())
    assert 'AgeGroup' in out.columns
    expected = {35.0: 'Adult', 25.0: 'Adult', 10.0: 'Child', 5.0: 'Child', 45.0: 'Middle'}
    for idx, row in out.iterrows():
        age = SAMPLE.loc[idx, 'Age']
        assert row['AgeGroup'] == expected[age], f"Age {age} → {row['AgeGroup']} (expected {expected[age]})"


def test_social_class_range():
    fe = make_fe()
    out = fe.transform(SAMPLE.copy())
    assert 'SocialClass' in out.columns
    assert out['SocialClass'].between(1, 9).all()


def test_drop_fare_for_model():
    fe = make_fe(drop_fare_for=['logreg'])
    out_logreg = fe.transform(SAMPLE.copy(), model_name='logreg')
    out_xgb    = fe.transform(SAMPLE.copy(), model_name='xgboost')
    assert 'Fare' not in out_logreg.columns
    assert 'Fare' in out_xgb.columns


def test_fare_kept_when_not_in_drop_list():
    fe = make_fe(drop_fare_for=[])
    out = fe.transform(SAMPLE.copy(), model_name='logreg')
    assert 'Fare' in out.columns


def test_report_keys():
    fe = make_fe()
    out = fe.transform(SAMPLE.copy())
    report = fe.report(out)
    assert 'TitleGroup' in report
    assert 'AgeGroup' in report
    assert 'SocialClass' in report
    assert 'FarePerPerson' in report
