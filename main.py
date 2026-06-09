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


def section(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def main():
    # ── CARICAMENTO TRAIN ────────────────────────────────────────
    section("CARICAMENTO TRAIN")
    loader = GenericLoader(
        target_col='Survived',
        csv_path='data/train.csv',
        drop_cols=['PassengerId', 'Name', 'Ticket', 'Cabin'],
        drop_missing_thresh=0.6,
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

    bal = eda.class_balance(loader.df['Survived'])
    print("  Bilanciamento target:")
    for cls, v in bal.items():
        print(f"    {cls}: {v['count']} ({v['pct']}%)")

    corr = eda.correlation()
    if corr:
        survived_corr = sorted(
            ((k, v) for k, v in corr.get('Survived', {}).items() if k != 'Survived'),
            key=lambda x: abs(x[1] or 0), reverse=True
        )
        print("\n  Correlazione con Survived:")
        for feat, val in survived_corr:
            print(f"    {feat:<15} {val:.4f}")

    # ── PULIZIA ──────────────────────────────────────────────────
    section("PULIZIA DATI")
    X = loader.df.drop(columns=['Survived'])
    y = loader.df['Survived']

    cleaner = GenericCleaner()
    cleaner.configure(num_strategy='median', cat_strategy='most_frequent')
    X = cleaner.fit_transform(X)
    print(f"  Shape dopo pulizia: {X.shape}")

    # ── ENCODING ─────────────────────────────────────────────────
    section("ENCODING")
    encoder = GenericEncoder()
    encoder.configure(num_strategy='standard', cat_strategy='ohe')
    X_enc = encoder.fit_transform_features(X)
    y_enc = encoder.encode_target(y, fit=True)
    print(f"  Shape dopo encoding: {X_enc.shape}")

    # ── TRAINING (GridSearchCV con CV interna) ───────────────────
    models = {
        "LogisticRegression": GenericLogreg(),
        "XGBoost":            GenericXGBoost(),
        "DecisionTree":       GenericDecisionTree(),
        "SVM":                GenericSVM(),
    }

    results = {}
    for name, model in models.items():
        section(f"TRAINING  {name}")
        model.train(X_enc, y_enc)
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
    test_raw = pd.read_csv('data/test.csv')
    passenger_ids = test_raw['PassengerId']

    test_df = test_raw.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], errors='ignore')
    test_df = cleaner.transform(test_df)
    test_enc = encoder.transform_features(test_df)

    best_model = best_r['model']
    preds = best_model.predict(test_enc)
    preds_decoded = encoder._target_le.inverse_transform(preds)

    submission = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': preds_decoded})
    submission.to_csv('data/submission.csv', index=False)
    print(f"  submission.csv salvato ({len(submission)} righe) con {best_name}")
    print(f"\n{SEP}\n  FINE\n{SEP}\n")


if __name__ == "__main__":
    main()
