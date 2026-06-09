import yaml
import pandas as pd
from lib.loader import GenericLoader
from lib.cleaner import GenericCleaner
from lib.encoder import GenericEncoder
from lib.eda import GenericEda
from lib.models.logreg import GenericLogreg
from lib.models.xgboost import GenericXGBoost
from lib.models.decision_tree import GenericDecisionTree
from lib.models.svm import GenericSVM

SEP = "=" * 65

_MODEL_REGISTRY = {
    'logreg':        GenericLogreg,
    'xgboost':       GenericXGBoost,
    'decision_tree': GenericDecisionTree,
    'svm':           GenericSVM,
}


def section(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def load_config(path: str = 'config.yaml') -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()

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

    # ── EDA ──────────────────────────────────────────────────────
    section("EDA")
    eda = GenericEda(loader.df)

    bal = eda.class_balance(loader.df[cfg['loader']['target_col']])
    print("  Bilanciamento target:")
    for cls, v in bal.items():
        print(f"    {cls}: {v['count']} ({v['pct']}%)")

    corr = eda.correlation()
    if corr:
        target = cfg['loader']['target_col']
        survived_corr = sorted(
            ((k, v) for k, v in corr.get(target, {}).items() if k != target),
            key=lambda x: abs(x[1] or 0), reverse=True
        )
        print(f"\n  Correlazione con {target}:")
        for feat, val in survived_corr:
            print(f"    {feat:<15} {val:.4f}")

    # ── PULIZIA ──────────────────────────────────────────────────
    section("PULIZIA DATI")
    X = loader.df.drop(columns=[cfg['loader']['target_col']])
    y = loader.df[cfg['loader']['target_col']]

    cleaner = GenericCleaner()
    cleaner.configure(
        num_strategy=cfg['cleaner']['num_strategy'],
        cat_strategy=cfg['cleaner']['cat_strategy'],
    )
    X = cleaner.fit_transform(X)
    print(f"  Shape dopo pulizia: {X.shape}")

    # ── ENCODING ─────────────────────────────────────────────────
    section("ENCODING")
    encoder = GenericEncoder()
    encoder.configure(
        num_strategy=cfg['encoder']['num_strategy'],
        cat_strategy=cfg['encoder']['cat_strategy'],
    )
    X_enc = encoder.fit_transform_features(X)
    y_enc = encoder.encode_target(y, fit=True)
    print(f"  Shape dopo encoding: {X_enc.shape}")

    # ── TRAINING ─────────────────────────────────────────────────
    active = cfg['models']['active']
    cv      = cfg['models'].get('cv', 5)
    scoring = cfg['models'].get('scoring', 'f1_weighted')

    unknown = [m for m in active if m not in _MODEL_REGISTRY]
    if unknown:
        raise ValueError(f"Modelli non riconosciuti in config.yaml: {unknown}")

    models = {name: _MODEL_REGISTRY[name]() for name in active}

    results = {}
    for name, model in models.items():
        section(f"TRAINING  {name}")
        model.train(X_enc, y_enc, cv=cv, scoring=scoring)
        print(f"  Migliori params: {model.best_params}")
        metrics = model.evaluate(X_enc, y_enc)
        results[name] = {'model': model, 'metrics': metrics}
        print(f"\n{model.classification_report(X_enc, y_enc)}")

    # ── RIEPILOGO ────────────────────────────────────────────────
    section("RIEPILOGO COMPARATIVO")
    print(f"  {'Modello':<22} {'Accuracy':>8}  {'Precision':>9}  {'Recall':>6}  {'F1':>6}")
    print("  " + "-" * 60)
    sorted_results = sorted(results.items(), key=lambda x: x[1]['metrics']['f1_score'], reverse=True)
    for name, r in sorted_results:
        m = r['metrics']
        print(f"  {name:<22} {m['accuracy']:>8.4f}  {m['precision']:>9.4f}  "
              f"{m['recall']:>6.4f}  {m['f1_score']:>6.4f}")

    best_name, best_r = sorted_results[0]
    print(f"\n  Miglior modello (F1): {best_name}  →  F1={best_r['metrics']['f1_score']:.4f}")

    # ── PREDIZIONE SU TEST ───────────────────────────────────────
    section("PREDIZIONE SU TEST")
    test_raw = pd.read_csv(cfg['loader']['test_path'])
    passenger_ids = test_raw['PassengerId']

    drop = [c for c in cfg['loader'].get('drop_cols', []) if c != 'PassengerId']
    test_df = test_raw.drop(columns=['PassengerId'] + drop, errors='ignore')
    test_df = cleaner.transform(test_df)
    test_enc = encoder.transform_features(test_df)

    best_model = best_r['model']
    preds = encoder._target_le.inverse_transform(best_model.predict(test_enc))

    submission = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': preds})
    submission_path = cfg['output']['submission_path']
    submission.to_csv(submission_path, index=False)
    print(f"  {submission_path} salvato ({len(submission)} righe) con {best_name}")
    print(f"\n{SEP}\n  FINE\n{SEP}\n")


if __name__ == "__main__":
    main()
