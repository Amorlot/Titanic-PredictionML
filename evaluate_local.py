"""
Valutazione locale dei modelli senza caricare su Kaggle.

Strategia: StratifiedKFold(5) su train.csv — stesso approccio del notebook di riferimento.
  1. Preprocessing su tutti i 891 campioni di train
  2. GridSearchCV → seleziona best_estimator_
  3. cross_val_score(best_estimator, X_enc, y_enc, cv=5, scoring='accuracy')

Kaggle valuta con accuracy — riportata come metrica principale.

Uso:
    python evaluate_local.py          # usa config.yaml di default
    python evaluate_local.py --folds 10   # più fold per stima più stabile
"""

import argparse
import yaml
import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score

from lib.feature_engineering import FeatureEngineer
from lib.cleaner import GenericCleaner
from lib.outliers import GenericOutlierHandler
from lib.multicollinearity import MulticollinearityHandler
from lib.pca_reducer import PCAReducer
from lib.encoder import GenericEncoder
from lib.models.logreg import GenericLogreg
from lib.models.xgboost import GenericXGBoost
from lib.models.decision_tree import GenericDecisionTree
from lib.models.random_forest import GenericRandomForest
from lib.models.svm import GenericSVM

SEP = "=" * 65

_MODEL_REGISTRY = {
    'logreg':        GenericLogreg,
    'xgboost':       GenericXGBoost,
    'decision_tree': GenericDecisionTree,
    'random_forest': GenericRandomForest,
    'svm':           GenericSVM,
}


def load_config(path='config.yaml'):
    with open(path) as f:
        return yaml.safe_load(f)


def build_pipeline(cfg, df_train):
    """Fitta tutti gli step del preprocessing sul train split e ritorna i transformer."""
    target = cfg['loader']['target_col']
    fe_cfg = cfg.get('feature_engineering', {})
    drop_fare_for = fe_cfg.get('drop_raw_fare_for', [])

    fe = None
    if fe_cfg.get('enabled', False):
        fe = FeatureEngineer()
        fe.configure(drop_raw_fare_for=drop_fare_for)
        fe.fit(df_train)
        df_train = fe.transform(df_train)

    X = df_train.drop(columns=[target])
    y = df_train[target]

    cleaner = GenericCleaner()
    cleaner.configure(
        num_strategy=cfg['cleaner']['num_strategy'],
        cat_strategy=cfg['cleaner']['cat_strategy'],
        knn_neighbors=cfg['cleaner'].get('knn_neighbors', 5),
    )
    X = cleaner.fit_transform(X)

    outlier_cfg = cfg.get('outliers', {})
    outlier_models = set(outlier_cfg.get('apply_to', []))
    outlier_handler = GenericOutlierHandler()
    outlier_handler.configure(
        detection=outlier_cfg.get('detection', 'iqr'),
        treatment=outlier_cfg.get('treatment', 'winsorization'),
        iqr_threshold=outlier_cfg.get('iqr_threshold', 1.5),
        zscore_threshold=outlier_cfg.get('zscore_threshold', 3.0),
        cols=outlier_cfg.get('cols') or None,
    )
    outlier_handler.fit(X)
    X_win = outlier_handler.transform(X)

    mc_cfg = cfg.get('multicollinearity', {})
    mc_models = set(mc_cfg.get('apply_to', []))
    mc_handler = MulticollinearityHandler()
    mc_handler.configure(threshold=mc_cfg.get('threshold', 0.85), drop=mc_cfg.get('drop', True))
    mc_handler.fit(X_win if mc_models & outlier_models else X)
    X_win_mc = mc_handler.transform(X_win)

    enc_cfg = cfg['encoder']
    encoder = GenericEncoder()
    encoder.configure(num_strategy=enc_cfg['num_strategy'], cat_strategy=enc_cfg['cat_strategy'])
    X_enc     = encoder.fit_transform_features(X)
    X_enc_win = encoder.transform_features(X_win)
    y_enc     = encoder.encode_target(y, fit=True)

    encoder_mc = GenericEncoder()
    encoder_mc.configure(num_strategy=enc_cfg['num_strategy'], cat_strategy=enc_cfg['cat_strategy'])
    X_enc_win_mc = encoder_mc.fit_transform_features(X_win_mc)

    pca_cfg    = cfg.get('pca', {})
    pca_models = set(pca_cfg.get('apply_to', []))
    pca_reducers = {}

    drop_fare_for_set = set(drop_fare_for)

    def base_dataset(name):
        use_win = name in outlier_models
        use_mc  = name in mc_models
        if use_win and use_mc:
            base, enc_used, label = X_enc_win_mc, encoder_mc, 'win+no-mc'
        elif use_win:
            base, enc_used, label = X_enc_win, encoder, 'win'
        else:
            base, enc_used, label = X_enc, encoder, 'orig'
        if name in drop_fare_for_set and 'Fare' in base.columns:
            base = base.drop(columns=['Fare'])
            label += ' (no Fare)'
        return base, enc_used, label

    _rs = cfg['models'].get('random_state', 42)
    for name in pca_models:
        if name in cfg['models']['active']:
            X_b, _, _ = base_dataset(name)
            r = PCAReducer()
            r.configure(n_components=pca_cfg.get('n_components', 0.95), random_state=_rs)
            r.fit(X_b)
            pca_reducers[name] = r

    return dict(
        fe=fe, cleaner=cleaner,
        outlier_handler=outlier_handler, outlier_models=outlier_models,
        mc_handler=mc_handler, mc_models=mc_models,
        encoder=encoder, encoder_mc=encoder_mc,
        pca_reducers=pca_reducers, pca_models=pca_models,
        y_enc=y_enc, drop_fare_for=drop_fare_for_set,
        base_dataset_fn=base_dataset,
    )


def transform_test(df_test, model_name, pipe, cfg):
    target = cfg['loader']['target_col']
    fe = pipe['fe']
    if fe is not None:
        df_test = fe.transform(df_test)
    df_test = pipe['cleaner'].transform(df_test.drop(columns=[target], errors='ignore'))
    if model_name in pipe['outlier_models']:
        df_test = pipe['outlier_handler'].transform(df_test)
    if model_name in pipe['mc_models']:
        df_test = pipe['mc_handler'].transform(df_test)

    enc = pipe['encoder_mc'] if model_name in pipe['mc_models'] else pipe['encoder']
    X_test = enc.transform_features(df_test)

    if model_name in pipe['drop_fare_for'] and 'Fare' in X_test.columns:
        X_test = X_test.drop(columns=['Fare'])
    if model_name in pipe['pca_models']:
        X_test = pipe['pca_reducers'][model_name].transform(X_test)
    return X_test


def evaluate_all(cfg, n_folds):
    df = pd.read_csv(cfg['loader']['train_path'])
    target = cfg['loader']['target_col']
    drop_cols = [c for c in cfg['loader'].get('drop_cols', []) if c != 'PassengerId']
    df = df.drop(columns=['PassengerId'] + drop_cols, errors='ignore')

    grid_scoring  = cfg['models'].get('scoring', 'f1_weighted')
    grid_cv       = cfg['models'].get('cv', 5)
    random_state  = cfg['models'].get('random_state', 42)

    # preprocessing su tutti i 891 campioni (come fa il notebook)
    pipe = build_pipeline(cfg, df.copy())
    y_enc = pipe['y_enc']
    cv_strategy = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    results = {}
    trained_models = {}
    for name in cfg['models']['active']:
        X_base, _, label = pipe['base_dataset_fn'](name)

        print(f"  TRAINING {name} ...", flush=True)
        model = _MODEL_REGISTRY[name]()
        model.train(X_base, y_enc, cv=grid_cv, scoring=grid_scoring, random_state=random_state)
        trained_models[name] = model

        scores = cross_val_score(
            model.model, X_base, y_enc,
            cv=cv_strategy, scoring='accuracy', n_jobs=-1,
        )
        results[name] = {'scores': scores, 'label': label}

        for i, s in enumerate(scores, 1):
            bar = '█' * int(s * 30)
            print(f"    Fold {i}: {s:.4f}  {bar}", flush=True)
        print(f"    Media: {scores.mean():.4f}  Std: {scores.std():.4f}\n", flush=True)

        # variante PCA
        if name in pipe['pca_models']:
            X_pca = pipe['pca_reducers'][name].transform(X_base)
            print(f"  TRAINING {name}+PCA ...", flush=True)
            model_pca = _MODEL_REGISTRY[name]()
            model_pca.train(X_pca, y_enc, cv=grid_cv, scoring=grid_scoring, random_state=random_state)
            trained_models[f'{name}_pca'] = model_pca

            scores_pca = cross_val_score(
                model_pca.model, X_pca, y_enc,
                cv=cv_strategy, scoring='accuracy', n_jobs=-1,
            )
            results[f'{name}_pca'] = {'scores': scores_pca, 'label': label + '+PCA'}
            for i, s in enumerate(scores_pca, 1):
                bar = '█' * int(s * 30)
                print(f"    Fold {i}: {s:.4f}  {bar}", flush=True)
            print(f"    Media: {scores_pca.mean():.4f}  Std: {scores_pca.std():.4f}\n", flush=True)

    return results, trained_models, pipe, random_state


def _get_model_X(pipe, name):
    """Restituisce X per un modello, applicando PCA se necessario."""
    X_base, _, label = pipe['base_dataset_fn'](name)
    if name in pipe['pca_models']:
        X_base = pipe['pca_reducers'][name].transform(X_base)
        label += '+PCA'
    return X_base, label


def evaluate_ensemble(cfg, pipe, trained_models, n_folds, random_state):
    ens_cfg = cfg.get('ensemble', {})
    if not ens_cfg.get('enabled', False):
        return None

    members_names = [m for m in ens_cfg.get('members', []) if m in trained_models]
    if len(members_names) < 2:
        print("  [WARN] Ensemble: meno di 2 membri disponibili — saltato\n")
        return None

    weights_cfg = ens_cfg.get('weights', 'auto')
    voting      = ens_cfg.get('voting', 'soft')

    if weights_cfg == 'auto' or weights_cfg is None:
        weights = {m: trained_models[m].best_score for m in members_names}
    elif isinstance(weights_cfg, dict):
        weights = {m: float(weights_cfg.get(m, 1.0)) for m in members_names}
    else:
        weights = {m: 1.0 for m in members_names}

    total_w  = sum(weights[m] for m in members_names)
    y_enc    = pipe['y_enc']
    cv_strat = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    X_ref, _ = _get_model_X(pipe, members_names[0])

    print(f"  ENSEMBLE ({voting}) — membri: {members_names} ...", flush=True)
    fold_scores = []
    for fold_idx, (train_idx, test_idx) in enumerate(cv_strat.split(X_ref, y_enc)):
        y_train_fold = y_enc[train_idx]
        y_test_fold  = y_enc[test_idx]

        if voting == 'soft':
            proba_sum = np.zeros(len(test_idx))
            for mname in members_names:
                X_base, _ = _get_model_X(pipe, mname)
                est = clone(trained_models[mname].model)
                est.fit(X_base.iloc[train_idx], y_train_fold)
                proba_sum += (weights[mname] / total_w) * est.predict_proba(X_base.iloc[test_idx])[:, 1]
            preds = (proba_sum >= 0.5).astype(int)
        else:
            votes = np.zeros(len(test_idx))
            for mname in members_names:
                X_base, _ = _get_model_X(pipe, mname)
                est = clone(trained_models[mname].model)
                est.fit(X_base.iloc[train_idx], y_train_fold)
                votes += (weights[mname] / total_w) * est.predict(X_base.iloc[test_idx])
            preds = (votes >= 0.5).astype(int)

        acc = (preds == y_test_fold).mean()
        fold_scores.append(acc)
        bar = '█' * int(acc * 30)
        print(f"    Fold {fold_idx + 1}: {acc:.4f}  {bar}", flush=True)

    scores = np.array(fold_scores)
    print(f"    Media: {scores.mean():.4f}  Std: {scores.std():.4f}\n", flush=True)
    return {'scores': scores, 'label': f"ensemble({'+'.join(members_names)})"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--folds', type=int, default=5)
    args = parser.parse_args()

    cfg = load_config(args.config)

    print(f"\n{SEP}")
    print(f"  VALUTAZIONE LOCALE — StratifiedKFold({args.folds}) su train.csv")
    print(f"  Kaggle usa Accuracy — stessa metrica riportata qui")
    print(SEP + "\n")

    results, trained_models, pipe, random_state = evaluate_all(cfg, args.folds)

    ens_result = evaluate_ensemble(cfg, pipe, trained_models, args.folds, random_state)
    if ens_result:
        results['ensemble'] = ens_result

    print(f"\n{SEP}")
    print(f"  {'Modello':<22} {'Dataset':<22} {'Accuracy':>10}  {'± Std':>8}")
    print(f"  {'-'*66}")

    sorted_models = sorted(
        results.items(),
        key=lambda x: x[1]['scores'].mean(),
        reverse=True
    )
    for name, r in sorted_models:
        acc = r['scores'].mean()
        std = r['scores'].std()
        print(f"  {name:<22} {r['label']:<22} {acc:>10.4f}  ±{std:.4f}")

    best_name, best_r = sorted_models[0]
    best_acc = best_r['scores'].mean()
    print(f"\n  Miglior modello: {best_name}  →  Accuracy={best_acc:.4f}  "
          f"(stima Kaggle: ~{best_acc:.1%})")
    print(f"\n{SEP}\n")


if __name__ == '__main__':
    main()
