import hashlib
import json
import os

import joblib
import yaml
import pandas as pd

from lib.loader import GenericLoader
from lib.cleaner import GenericCleaner
from lib.outliers import GenericOutlierHandler
from lib.multicollinearity import MulticollinearityHandler
from lib.pca_reducer import PCAReducer
from lib.feature_engineering import FeatureEngineer
from lib.encoder import GenericEncoder
from lib.eda import GenericEda
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


def section(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def load_config(path: str = 'config.yaml') -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _config_hash(cfg: dict) -> str:
    """MD5 dei parametri che, se cambiano, richiedono il retraining."""
    relevant = {k: cfg[k] for k in
                ('models', 'cleaner', 'outliers', 'multicollinearity',
                 'pca', 'encoder', 'feature_engineering')
                if k in cfg}
    return hashlib.md5(json.dumps(relevant, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _cache_path(cache_dir: str, name: str, cfg_hash: str) -> str:
    return os.path.join(cache_dir, f"{name}_{cfg_hash}.pkl")


def _load_or_train(name, model_cls, X, y, cv, scoring, cache_enabled, cache_dir, cfg_hash):
    path = _cache_path(cache_dir, name, cfg_hash)
    if cache_enabled and os.path.exists(path):
        model = joblib.load(path)
        print(f"  [cache] Caricato da {path}")
        return model, True
    model = model_cls()
    model.train(X, y, cv=cv, scoring=scoring)
    if cache_enabled:
        os.makedirs(cache_dir, exist_ok=True)
        joblib.dump(model, path)
        print(f"  [cache] Salvato in {path}")
    return model, False


def main():
    cfg = load_config()

    # resolve cache settings early
    cache_cfg     = cfg.get('model_cache', {})
    cache_enabled = cache_cfg.get('enabled', False)
    cache_dir     = cache_cfg.get('path', 'models/')
    cfg_hash      = _config_hash(cfg)

    active            = cfg['models']['active']
    cv                = cfg['models'].get('cv', 5)
    scoring           = cfg['models'].get('scoring', 'f1_weighted')
    tune_threshold    = cfg['models'].get('tune_threshold', False)
    threshold_scoring = cfg['models'].get('threshold_scoring', 'f1')

    unknown = [m for m in active if m not in _MODEL_REGISTRY]
    if unknown:
        raise ValueError(f"Modelli non riconosciuti in config.yaml: {unknown}")

    # ── CARICAMENTO TRAIN ────────────────────────────────────────
    section("CARICAMENTO TRAIN")
    loader = GenericLoader(
        target_col=cfg['loader']['target_col'],
        csv_path=cfg['loader']['train_path'],
        drop_cols=cfg['loader'].get('drop_cols', []),
        drop_missing_thresh=cfg['loader'].get('drop_missing_thresh'),
    )
    loader.load()
    info = loader.info()
    print(f"  Righe:       {info['rows']}")
    print(f"  Feature num: {info['numerical']}")
    print(f"  Feature cat: {info['categorical']}")
    print(f"  Target dist: {info['target_distribution']}")

    miss = loader.missing_report()
    if miss:
        print("  Valori mancanti:")
        for col, v in miss.items():
            print(f"    {col}: {v['count']} ({v['pct']}%)")

    # ── FEATURE ENGINEERING ──────────────────────────────────────
    fe_cfg  = cfg.get('feature_engineering', {})
    fe      = None
    drop_fare_for = fe_cfg.get('drop_raw_fare_for', [])

    if fe_cfg.get('enabled', False):
        section("FEATURE ENGINEERING")
        fe = FeatureEngineer()
        fe.configure(drop_raw_fare_for=drop_fare_for)
        fe.fit(loader.df)
        loader.df = fe.transform(loader.df)          # keeps Fare for everyone at this stage
        fe_report = fe.report(loader.df)
        print("  TitleGroup:")
        for title, cnt in fe_report.get('TitleGroup', {}).items():
            print(f"    {title:<10} {cnt}")
        age_imp = fe_report.get('AgeImputationByTitle', {})
        if age_imp:
            print("  Imputazione Age per titolo:")
            for title, med in sorted(age_imp.items()):
                print(f"    {title:<10} mediana={med}")
        print("  AgeGroup:")
        for ag, cnt in fe_report.get('AgeGroup', {}).items():
            print(f"    {ag:<10} {cnt}")
        print("  SocialClass (1=top, 9=bottom):")
        for sc, cnt in fe_report.get('SocialClass', {}).items():
            print(f"    classe {sc}: {cnt} passeggeri")
        fpp = fe_report.get('FarePerPerson', {})
        if fpp:
            print(f"  FarePerPerson: min={fpp['min']}  mean={fpp['mean']}  max={fpp['max']}")

    # ── EDA ──────────────────────────────────────────────────────
    section("EDA")
    eda    = GenericEda(loader.df)
    target = cfg['loader']['target_col']

    bal = eda.class_balance(loader.df[target])
    print("  Bilanciamento target:")
    for cls, v in bal.items():
        print(f"    {cls}: {v['count']} ({v['pct']}%)")

    corr = eda.correlation()
    if corr:
        survived_corr = sorted(
            ((k, v) for k, v in corr.get(target, {}).items() if k != target),
            key=lambda x: abs(x[1] or 0), reverse=True
        )
        print(f"\n  Correlazione con {target}:")
        for feat, val in survived_corr:
            print(f"    {feat:<15} {val:.4f}")

    # ── PULIZIA ──────────────────────────────────────────────────
    section("PULIZIA DATI")
    X = loader.df.drop(columns=[target])
    y = loader.df[target]

    cleaner = GenericCleaner()
    cleaner.configure(
        num_strategy=cfg['cleaner']['num_strategy'],
        cat_strategy=cfg['cleaner']['cat_strategy'],
        knn_neighbors=cfg['cleaner'].get('knn_neighbors', 5),
    )
    X = cleaner.fit_transform(X)
    print(f"  Shape dopo pulizia: {X.shape}")

    # ── OUTLIERS ─────────────────────────────────────────────────
    section("OUTLIERS")
    outlier_cfg = cfg.get('outliers', {})
    outlier_handler = GenericOutlierHandler()
    outlier_handler.configure(
        detection=outlier_cfg.get('detection', 'iqr'),
        treatment=outlier_cfg.get('treatment', 'winsorization'),
        iqr_threshold=outlier_cfg.get('iqr_threshold', 1.5),
        zscore_threshold=outlier_cfg.get('zscore_threshold', 3.0),
        cols=outlier_cfg.get('cols') or None,
    )
    outlier_models = set(outlier_cfg.get('apply_to', []))

    outlier_handler.fit(X)
    X_winsorized = outlier_handler.transform(X)

    out_report = outlier_handler.report(X)
    if out_report:
        print("  Colonna          Rilevati    Modificati   Bounds")
        print("  " + "-" * 55)
        for col, v in out_report.items():
            modified = int((X[col] != X_winsorized[col]).sum())
            print(f"  {col:<16} {v['outliers']:>5} ({v['pct']:>5}%)  "
                  f"{modified:>5}          [{v['lower_bound']}, {v['upper_bound']}]")
    else:
        print("  Nessun outlier rilevato.")
    print(f"  Winsorization applicata a: {sorted(outlier_models)}")

    # ── MULTICOLLINEARITÀ ────────────────────────────────────────
    section("MULTICOLLINEARITÀ")
    mc_cfg    = cfg.get('multicollinearity', {})
    mc_models = set(mc_cfg.get('apply_to', []))
    mc_handler = MulticollinearityHandler()
    mc_handler.configure(
        threshold=mc_cfg.get('threshold', 0.85),
        drop=mc_cfg.get('drop', True),
    )
    mc_handler.fit(X_winsorized if mc_models & outlier_models else X)
    mc_report = mc_handler.report()

    if mc_report['pairs']:
        print("  Coppie correlate trovate:")
        for p in mc_report['pairs']:
            print(f"    {p['col_a']} ↔ {p['col_b']}  |r|={p['r']}")
        if mc_report['cols_to_drop']:
            print(f"  Colonne droppate per multicollinearità: {mc_report['cols_to_drop']}")
    else:
        print(f"  Nessuna coppia con |r| > {mc_cfg.get('threshold', 0.85)}")
    print(f"  Trattamento applicato a: {sorted(mc_models)}")

    # ── ENCODING ─────────────────────────────────────────────────
    section("ENCODING")
    enc_cfg = cfg['encoder']

    X_winsorized_mc = mc_handler.transform(X_winsorized)

    # Encoder base: fit su X, usato da tutti i modelli che NON passano per mc
    encoder = GenericEncoder()
    encoder.configure(num_strategy=enc_cfg['num_strategy'], cat_strategy=enc_cfg['cat_strategy'])
    X_enc     = encoder.fit_transform_features(X)
    X_enc_win = encoder.transform_features(X_winsorized)   # stesse colonne di X
    y_enc     = encoder.encode_target(y, fit=True)

    # Encoder mc: fit su X_winsorized_mc (colonne diverse dopo mc removal)
    encoder_mc = GenericEncoder()
    encoder_mc.configure(num_strategy=enc_cfg['num_strategy'], cat_strategy=enc_cfg['cat_strategy'])
    X_enc_win_mc = encoder_mc.fit_transform_features(X_winsorized_mc)

    print(f"  Shape originale:            {X_enc.shape}")
    print(f"  Shape winsorizzato:         {X_enc_win.shape}")
    print(f"  Shape winsorizzato + no-mc: {X_enc_win_mc.shape}")

    # ── PCA ──────────────────────────────────────────────────────
    section("PCA")
    pca_cfg    = cfg.get('pca', {})
    pca_models = set(pca_cfg.get('apply_to', []))

    def base_dataset(name):
        """Restituisce (X_encoded, label, encoder_used) per il modello name."""
        use_win = name in outlier_models
        use_mc  = name in mc_models
        if use_win and use_mc:
            base = X_enc_win_mc
            label = 'winsorized+no-mc'
            enc_used = encoder_mc
        elif use_win:
            base = X_enc_win
            label = 'winsorized'
            enc_used = encoder
        else:
            base = X_enc
            label = 'originale'
            enc_used = encoder

        # Drop Fare grezzo per modelli lineari
        if name in drop_fare_for and 'Fare' in base.columns:
            base = base.drop(columns=['Fare'])
            label += ' (no Fare)'
        return base, label, enc_used

    # PCA separata per ogni modello: logreg e svm hanno colonne diverse
    pca_reducers: dict = {}
    if pca_models:
        for pca_name in [m for m in active if m in pca_models]:
            X_base_for_pca, _, _ = base_dataset(pca_name)
            r = PCAReducer()
            r.configure(n_components=pca_cfg.get('n_components', 0.95))
            r.fit(X_base_for_pca)
            pca_reducers[pca_name] = r
            rep = r.report()
            print(f"  [{pca_name}] in={rep['n_components_in']}  out={rep['n_components_out']}  "
                  f"var={rep['variance_explained']*100:.1f}%")
        print(f"  PCA applicata a: {sorted(pca_models)}")

    # ── TRAINING ─────────────────────────────────────────────────
    results = {}
    for name in active:
        X_base, base_label, enc_used = base_dataset(name)

        # Versione senza PCA
        section(f"TRAINING  {name}  [no PCA]")
        print(f"  Dataset: {base_label}")
        model, from_cache = _load_or_train(
            name, _MODEL_REGISTRY[name],
            X_base, y_enc, cv, scoring,
            cache_enabled, cache_dir, cfg_hash,
        )
        if not from_cache:
            print(f"  Migliori params: {model.best_params}")
        print(f"  CV best score ({scoring}): {model.best_score:.4f}")
        if tune_threshold and not from_cache:
            thr = model.tune_threshold(X_base, y_enc, cv=cv, scoring=threshold_scoring)
            if thr:
                print(f"  Soglia ottimale ({threshold_scoring}): {thr['best_threshold']}  "
                      f"score={thr['best_score']:.4f}  (default=0.5)")
        results[name] = {
            'model': model, 'cv_score': model.best_score,
            'dataset': base_label,
            'use_win': name in outlier_models,
            'use_mc':  name in mc_models,
            'use_pca': False,
            'base_name': name,
            'encoder': enc_used,
        }
        print(f"\n{model.classification_report(X_base, y_enc)}")

        # Versione con PCA
        if name in pca_models:
            section(f"TRAINING  {name}  [PCA]")
            X_pca = pca_reducers[name].transform(X_base)
            pca_label = f"{base_label}+PCA ({X_pca.shape[1]} comp)"
            print(f"  Dataset: {pca_label}")
            pca_name = f"{name}_pca"
            model_pca, from_cache_pca = _load_or_train(
                pca_name, _MODEL_REGISTRY[name],
                X_pca, y_enc, cv, scoring,
                cache_enabled, cache_dir, cfg_hash,
            )
            if not from_cache_pca:
                print(f"  Migliori params: {model_pca.best_params}")
            print(f"  CV best score ({scoring}): {model_pca.best_score:.4f}")
            if tune_threshold and not from_cache_pca:
                thr = model_pca.tune_threshold(X_pca, y_enc, cv=cv, scoring=threshold_scoring)
                if thr:
                    print(f"  Soglia ottimale ({threshold_scoring}): {thr['best_threshold']}  "
                          f"score={thr['best_score']:.4f}  (default=0.5)")
            results[pca_name] = {
                'model': model_pca, 'cv_score': model_pca.best_score,
                'dataset': pca_label,
                'use_win': name in outlier_models,
                'use_mc':  name in mc_models,
                'use_pca': True,
                'base_name': name,
                'encoder': enc_used,
                'pca_reducer': pca_reducers[name],
            }
            print(f"\n{model_pca.classification_report(X_pca, y_enc)}")

    # ── RIEPILOGO ────────────────────────────────────────────────
    section("RIEPILOGO COMPARATIVO  (CV best score — honest out-of-fold)")
    print(f"  {'Modello':<26} {'Dataset':<30} {'CV Score':>8}")
    print("  " + "-" * 70)
    sorted_results = sorted(results.items(), key=lambda x: x[1]['cv_score'], reverse=True)
    for name, r in sorted_results:
        print(f"  {name:<26} {r['dataset']:<30} {r['cv_score']:>8.4f}")

    best_name, best_r = sorted_results[0]
    print(f"\n  Miglior modello ({scoring}): {best_name}  →  CV={best_r['cv_score']:.4f}")

    # ── PREDIZIONE SU TEST ───────────────────────────────────────
    section("PREDIZIONE SU TEST")
    test_raw = pd.read_csv(cfg['loader']['test_path'])
    passenger_ids = test_raw['PassengerId']
    drop = [c for c in cfg['loader'].get('drop_cols', []) if c != 'PassengerId']

    sub_dir = cfg['output'].get('submission_dir', 'data/')
    os.makedirs(sub_dir, exist_ok=True)

    # Per ogni modello base usa la variante con CV score più alto (pca vs no-pca)
    best_per_model: dict = {}
    for key, r in results.items():
        base = r['base_name']
        if base not in best_per_model or r['cv_score'] > best_per_model[base][1]['cv_score']:
            best_per_model[base] = (key, r)

    for base_name, (variant_key, r) in best_per_model.items():
        test_df = test_raw.drop(columns=['PassengerId'] + drop, errors='ignore')

        if fe is not None:
            test_df = fe.transform(test_df)   # Fare rimosso dopo l'encoding, non qui
        test_df = cleaner.transform(test_df)

        if r['use_win']:
            test_df = outlier_handler.transform(test_df)
        if r['use_mc']:
            test_df = mc_handler.transform(test_df)

        test_enc = r['encoder'].transform_features(test_df)

        if base_name in drop_fare_for and 'Fare' in test_enc.columns:
            test_enc = test_enc.drop(columns=['Fare'])

        if r['use_pca']:
            test_enc = r['pca_reducer'].transform(test_enc)

        preds = encoder.decode_target(r['model'].predict(test_enc))

        path = os.path.join(sub_dir, f"submission_{base_name}.csv")
        pd.DataFrame({'PassengerId': passenger_ids, 'Survived': preds}).to_csv(path, index=False)
        print(f"  {path:<45} variante={variant_key:<18} CV={r['cv_score']:.4f}")

    print(f"\n  Miglior modello ({scoring}): {best_name}  →  CV={best_r['cv_score']:.4f}")
    print(f"\n{SEP}\n  FINE\n{SEP}\n")


if __name__ == "__main__":
    main()
