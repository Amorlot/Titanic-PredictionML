import hashlib
import json
import os

import joblib
import numpy as np
import yaml
import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report as skl_report, f1_score

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
from lib.models.ensemble import GenericEnsemble

SEP = "=" * 65

_MODEL_REGISTRY = {
    'logreg':        GenericLogreg,
    'xgboost':       GenericXGBoost,
    'decision_tree': GenericDecisionTree,
    'random_forest': GenericRandomForest,
    'svm':           GenericSVM,
}


def section(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")


def load_config(path: str = 'config.yaml') -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _config_hash(cfg: dict) -> str:
    relevant = {k: cfg[k] for k in
                ('models', 'cleaner', 'outliers', 'multicollinearity',
                 'pca', 'encoder', 'feature_engineering')
                if k in cfg}
    return hashlib.md5(json.dumps(relevant, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _cache_path(cache_dir: str, name: str, cfg_hash: str) -> str:
    return os.path.join(cache_dir, f"{name}_{cfg_hash}.pkl")


def _load_or_train(name, model_cls, X, y, cv, scoring, random_state, cache_enabled, cache_dir, cfg_hash):
    path = _cache_path(cache_dir, name, cfg_hash)
    if cache_enabled and os.path.exists(path):
        model = joblib.load(path)
        print(f"  [cache] Caricato da {path}")
        return model, True
    model = model_cls()
    model.train(X, y, cv=cv, scoring=scoring, random_state=random_state)
    if cache_enabled:
        os.makedirs(cache_dir, exist_ok=True)
        joblib.dump(model, path)
        print(f"  [cache] Salvato in {path}")
    return model, False


def _f1_average(scoring: str) -> str:
    return scoring[len('f1_'):] if scoring.startswith('f1_') else 'weighted'


def main():
    cfg = load_config()

    cache_cfg     = cfg.get('model_cache', {})
    cache_enabled = cache_cfg.get('enabled', False)
    cache_dir     = cache_cfg.get('path', 'models/')
    cfg_hash      = _config_hash(cfg)

    active            = cfg['models']['active']
    cv                = cfg['models'].get('cv', 5)
    scoring           = cfg['models'].get('scoring', 'f1_weighted')
    random_state      = cfg['models'].get('random_state', 42)
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
    fe_cfg        = cfg.get('feature_engineering', {})
    fe            = None
    drop_fare_for = fe_cfg.get('drop_raw_fare_for', [])

    if fe_cfg.get('enabled', False):
        section("FEATURE ENGINEERING")
        fe = FeatureEngineer()
        fe.configure(drop_raw_fare_for=drop_fare_for)
        fe.fit(loader.df)
        loader.df  = fe.transform(loader.df)
        fe_report  = fe.report(loader.df)
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
            key=lambda x: abs(x[1] or 0), reverse=True,
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
    outlier_cfg     = cfg.get('outliers', {})
    outlier_models  = set(outlier_cfg.get('apply_to', []))
    outlier_handler = GenericOutlierHandler()
    outlier_handler.configure(
        detection=outlier_cfg.get('detection', 'iqr'),
        treatment=outlier_cfg.get('treatment', 'winsorization'),
        iqr_threshold=outlier_cfg.get('iqr_threshold', 1.5),
        zscore_threshold=outlier_cfg.get('zscore_threshold', 3.0),
        cols=outlier_cfg.get('cols') or None,
    )
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
    enc_cfg         = cfg['encoder']
    X_winsorized_mc = mc_handler.transform(X_winsorized)

    encoder = GenericEncoder()
    encoder.configure(num_strategy=enc_cfg['num_strategy'], cat_strategy=enc_cfg['cat_strategy'])
    X_enc     = encoder.fit_transform_features(X)
    X_enc_win = encoder.transform_features(X_winsorized)
    y_enc     = encoder.encode_target(y, fit=True)

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
        """Restituisce (X_encoded, label, encoder_used) in base al preprocessing del modello."""
        use_win = name in outlier_models
        use_mc  = name in mc_models
        if use_win and use_mc:
            base, label, enc_used = X_enc_win_mc, 'winsorized+no-mc', encoder_mc
        elif use_win:
            base, label, enc_used = X_enc_win, 'winsorized', encoder
        else:
            base, label, enc_used = X_enc, 'originale', encoder
        if name in drop_fare_for and 'Fare' in base.columns:
            base  = base.drop(columns=['Fare'])
            label = label + ' (no Fare)'
        return base, label, enc_used

    pca_reducers: dict = {}
    if pca_models:
        for pca_name in [m for m in active if m in pca_models]:
            X_base_for_pca, _, _ = base_dataset(pca_name)
            r = PCAReducer()
            r.configure(n_components=pca_cfg.get('n_components', 0.95), random_state=random_state)
            r.fit(X_base_for_pca)
            pca_reducers[pca_name] = r
            rep = r.report()
            print(f"  [{pca_name}] in={rep['n_components_in']}  out={rep['n_components_out']}  "
                  f"var={rep['variance_explained']*100:.1f}%")
        print(f"  PCA applicata a: {sorted(pca_models)}")
    else:
        print("  PCA non abilitata.")

    # ── TRAINING ─────────────────────────────────────────────────
    results = {}
    for name in active:
        X_base, base_label, enc_used = base_dataset(name)
        has_pca = name in pca_models

        section(f"TRAINING  {name}" + ("  [no PCA]" if has_pca else ""))
        print(f"  Dataset: {base_label}")
        model, from_cache = _load_or_train(
            name, _MODEL_REGISTRY[name],
            X_base, y_enc, cv, scoring, random_state,
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
        print(f"\n{model.oof_report(X_base, y_enc, cv=cv)}")

        results[name] = {
            'model':     model,
            'cv_score':  model.best_score,
            'dataset':   base_label,
            'use_win':   name in outlier_models,
            'use_mc':    name in mc_models,
            'use_pca':   False,
            'base_name': name,
            'encoder':   enc_used,
        }

        if has_pca:
            section(f"TRAINING  {name}  [PCA]")
            X_pca     = pca_reducers[name].transform(X_base)
            pca_label = f"{base_label}+PCA ({X_pca.shape[1]} comp)"
            pca_key   = f"{name}_pca"
            print(f"  Dataset: {pca_label}")
            model_pca, from_cache_pca = _load_or_train(
                pca_key, _MODEL_REGISTRY[name],
                X_pca, y_enc, cv, scoring, random_state,
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
            print(f"\n{model_pca.oof_report(X_pca, y_enc, cv=cv)}")

            results[pca_key] = {
                'model':       model_pca,
                'cv_score':    model_pca.best_score,
                'dataset':     pca_label,
                'use_win':     name in outlier_models,
                'use_mc':      name in mc_models,
                'use_pca':     True,
                'base_name':   name,
                'encoder':     enc_used,
                'pca_reducer': pca_reducers[name],
            }

    # ── ENSEMBLE ─────────────────────────────────────────────────
    ens_cfg = cfg.get('ensemble', {})
    if ens_cfg.get('enabled', False):
        section("ENSEMBLE")
        ens_members_names = ens_cfg.get('members', list(active))
        ens_voting        = ens_cfg.get('voting', 'soft')
        ens_weights_cfg   = ens_cfg.get('weights', 'auto')

        members = []
        for mname in ens_members_names:
            if mname not in results:
                print(f"  [WARN] '{mname}' non trovato nei risultati — saltato")
                continue
            r = results[mname]
            w = r['cv_score'] if ens_weights_cfg == 'auto' else float(ens_weights_cfg.get(mname, 1.0))
            members.append((mname, r['model'], w))

        if len(members) < 2:
            print("  [WARN] Servono almeno 2 membri — ensemble saltato")
        else:
            total_w = sum(w for _, _, w in members)

            print(f"  voting={ens_voting}  cv={cv}")
            print(f"  {'Membro':<20} {'Peso':>8}  {'CV score':>8}")
            print("  " + "-" * 40)
            for mname, _, w in members:
                print(f"  {mname:<20} {w:>8.4f}  {results[mname]['cv_score']:>8.4f}")

            # Valutazione OOF: ogni membro produce prob OOF sul proprio dataset
            print(f"\n  Calcolo OOF...")
            if ens_voting == 'soft':
                proba_oof = np.zeros(len(y_enc))
                for mname, mmodel, weight in members:
                    mr       = results[mname]
                    X_m, _, _ = base_dataset(mr['base_name'])
                    if mr.get('use_pca'):
                        X_m = mr['pca_reducer'].transform(X_m)
                    p = cross_val_predict(
                        mmodel.model, X_m, y_enc,
                        cv=cv, method='predict_proba', n_jobs=-1,
                    )[:, 1]
                    proba_oof += (weight / total_w) * p
                ens = GenericEnsemble(members=members, voting=ens_voting)
                ens_oof_preds = (proba_oof >= ens.threshold).astype(int)
            else:
                votes_oof = np.zeros(len(y_enc))
                for mname, mmodel, weight in members:
                    mr        = results[mname]
                    X_m, _, _ = base_dataset(mr['base_name'])
                    if mr.get('use_pca'):
                        X_m = mr['pca_reducer'].transform(X_m)
                    v = cross_val_predict(mmodel.model, X_m, y_enc, cv=cv, n_jobs=-1)
                    votes_oof += (weight / total_w) * v
                ens = GenericEnsemble(members=members, voting=ens_voting)
                ens_oof_preds = (votes_oof >= 0.5).astype(int)

            ens_f1 = f1_score(y_enc, ens_oof_preds, average=_f1_average(scoring), zero_division=0)
            ens.best_score = ens_f1

            print(f"  OOF {scoring}: {ens_f1:.4f}\n")
            print(skl_report(y_enc, ens_oof_preds, zero_division=0))

            first_r = results[members[0][0]]
            results['ensemble'] = {
                'model':     ens,
                'cv_score':  ens_f1,
                'dataset':   f"ensemble({', '.join(m for m, _, _ in members)})",
                'use_win':   False,
                'use_mc':    False,
                'use_pca':   False,
                'base_name': 'ensemble',
                'encoder':   first_r['encoder'],
            }

    # ── RIEPILOGO ────────────────────────────────────────────────
    section("RIEPILOGO COMPARATIVO  (OOF score)")
    print(f"  {'Modello':<26} {'Dataset':<35} {scoring:>12}")
    print("  " + "-" * 76)
    for name, r in sorted(results.items(), key=lambda x: x[1]['cv_score'], reverse=True):
        print(f"  {name:<26} {r['dataset']:<35} {r['cv_score']:>12.4f}")

    best_name, best_r = max(results.items(), key=lambda x: x[1]['cv_score'])
    print(f"\n  Miglior modello: {best_name}  →  {scoring}={best_r['cv_score']:.4f}")

    # ── PREDIZIONE SU TEST ───────────────────────────────────────
    section("PREDIZIONE SU TEST")
    test_raw      = pd.read_csv(cfg['loader']['test_path'])
    passenger_ids = test_raw['PassengerId']
    drop          = [c for c in cfg['loader'].get('drop_cols', []) if c != 'PassengerId']
    sub_dir       = cfg['output'].get('submission_dir', 'data/')
    os.makedirs(sub_dir, exist_ok=True)

    def _prepare_test_enc(base_name, r):
        tdf = test_raw.drop(columns=['PassengerId'] + drop, errors='ignore')
        if fe is not None:
            tdf = fe.transform(tdf)
        tdf = cleaner.transform(tdf)
        if r['use_win']:
            tdf = outlier_handler.transform(tdf)
        if r['use_mc']:
            tdf = mc_handler.transform(tdf)
        tenc = r['encoder'].transform_features(tdf)
        if base_name in drop_fare_for and 'Fare' in tenc.columns:
            tenc = tenc.drop(columns=['Fare'])
        if r.get('use_pca'):
            tenc = r['pca_reducer'].transform(tenc)
        return tenc

    # Per ogni modello base prende la variante con CV score più alto (pca vs no-pca)
    best_per_model: dict = {}
    for key, r in results.items():
        base = r['base_name']
        if base not in best_per_model or r['cv_score'] > best_per_model[base][1]['cv_score']:
            best_per_model[base] = (key, r)

    for base_name, (variant_key, r) in best_per_model.items():
        if base_name == 'ensemble':
            ens_obj = r['model']
            total_w = sum(w for _, _, w in ens_obj.members)
            if ens_obj.voting == 'soft':
                proba = np.zeros(len(test_raw))
                for mname, mmodel, weight in ens_obj.members:
                    tenc   = _prepare_test_enc(mname, results[mname])
                    proba += (weight / total_w) * mmodel.model.predict_proba(tenc)[:, 1]
                raw_preds = (proba >= ens_obj.threshold).astype(int)
            else:
                votes = np.zeros(len(test_raw))
                for mname, mmodel, weight in ens_obj.members:
                    tenc   = _prepare_test_enc(mname, results[mname])
                    votes += (weight / total_w) * mmodel.predict(tenc)
                raw_preds = (votes >= 0.5).astype(int)
            preds = encoder.decode_target(raw_preds)
        else:
            test_enc = _prepare_test_enc(base_name, r)
            preds    = encoder.decode_target(r['model'].predict(test_enc))

        path = os.path.join(sub_dir, f"submission_{base_name}.csv")
        pd.DataFrame({'PassengerId': passenger_ids, 'Survived': preds}).to_csv(path, index=False)
        print(f"  {path:<45} variante={variant_key:<20} {scoring}={r['cv_score']:.4f}")

    print(f"\n  Miglior modello: {best_name}  →  {scoring}={best_r['cv_score']:.4f}")
    print(f"\n{SEP}\n  FINE\n{SEP}\n")


if __name__ == "__main__":
    main()
