from lib.loader import GenericLoader
from lib.cleaner import GenericCleaner
from lib.encoder import GenericEncoder
from lib.eda import GenericEda
from lib.splitter import GenericSplitter
from lib.models.logreg import GenericLogreg
from lib.models.xgboost import GenericXGBoost
from src.model_decision_tree import DecisionTreeModel
from src.model_svm import SVMModel

SEP = "=" * 65


def section(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def main():
    # ── CARICAMENTO ──────────────────────────────────────────────
    section("CARICAMENTO DATASET")
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

    # ── SPLIT ────────────────────────────────────────────────────
    section("SPLIT (80/20 stratificato)")
    X = loader.df.drop(columns=['Survived'])
    y = loader.df['Survived']
    splitter = GenericSplitter(test_size=0.2, random_state=42, stratify=True)
    X_train, X_test, y_train, y_test = splitter.split(X, y)
    print(f"  Train: {X_train.shape[0]} righe   Test: {X_test.shape[0]} righe")

    # ── PULIZIA ──────────────────────────────────────────────────
    section("PULIZIA DATI")
    cleaner = GenericCleaner()
    cleaner.configure(num_strategy='median', cat_strategy='most_frequent')
    X_train = cleaner.fit_transform(X_train)
    X_test  = cleaner.transform(X_test)
    print(f"  Train pulito: {X_train.shape}")
    print(f"  Test pulito:  {X_test.shape}")

    # ── ENCODING ─────────────────────────────────────────────────
    section("ENCODING")
    encoder = GenericEncoder()
    encoder.configure(num_strategy='standard', cat_strategy='ohe')
    X_train = encoder.fit_transform_features(X_train)
    X_test  = encoder.transform_features(X_test)
    y_train = encoder.encode_target(y_train, fit=True)
    y_test  = encoder.encode_target(y_test,  fit=False)
    print(f"  Train encoded: {X_train.shape}")
    print(f"  Test encoded:  {X_test.shape}")

    # ── MODELLI ──────────────────────────────────────────────────
    models = {
        "LogisticRegression": GenericLogreg(),
        "XGBoost":            GenericXGBoost(),
        "DecisionTree":       DecisionTreeModel(),
        "SVM":                SVMModel(),
    }

    results = {}
    for name, model in models.items():
        section(f"TRAINING  {name}")
        model.train(X_train, y_train)
        print(f"  Migliori params: {model.best_params}")
        metrics = model.evaluate(X_test, y_test)
        results[name] = metrics
        print(f"\n{model.classification_report(X_test, y_test)}")

    # ── TABELLA COMPARATIVA ──────────────────────────────────────
    section("RIEPILOGO COMPARATIVO")
    print(f"  {'Modello':<22} {'Accuracy':>8}  {'Precision':>9}  {'Recall':>6}  {'F1':>6}")
    print("  " + "-" * 60)
    for name, m in sorted(results.items(), key=lambda x: x[1]['f1_score'], reverse=True):
        print(f"  {name:<22} {m['accuracy']:>8.4f}  {m['precision']:>9.4f}  "
              f"{m['recall']:>6.4f}  {m['f1_score']:>6.4f}")

    best_name, best_m = max(results.items(), key=lambda x: x[1]['f1_score'])
    print(f"\n  Miglior modello (F1): {best_name}  →  F1={best_m['f1_score']:.4f}")
    print(f"\n{SEP}\n  FINE\n{SEP}\n")


if __name__ == "__main__":
    main()
