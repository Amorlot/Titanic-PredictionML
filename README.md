# Titanic ML Pipeline

Pipeline di machine learning modulare per classificazione binaria, sviluppata sul dataset Titanic (Kaggle). Ogni passo del preprocessing è configurabile via `config.yaml` senza toccare il codice.

---

## Struttura del progetto

```
titanic/
├── config.yaml              # configurazione completa della pipeline
├── main.py                  # entry point: training, valutazione, submission
├── app.py                   # API REST (Flask)
├── Dockerfile               # immagine per eseguire l'API
├── start.sh                 # avvia l'API dentro il container
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── submissions/         # CSV generati da main.py
├── models/                  # modelli serializzati (cache)
├── lib/
│   ├── loader.py
│   ├── cleaner.py
│   ├── outliers.py
│   ├── multicollinearity.py
│   ├── pca_reducer.py
│   ├── feature_engineering.py
│   ├── encoder.py
│   ├── eda.py
│   ├── splitter.py
│   └── models/
│       ├── base.py
│       ├── logreg.py
│       ├── xgboost.py
│       ├── decision_tree.py
│       ├── random_forest.py
│       ├── svm.py
│       └── ensemble.py
└── tests/
```

---

## Come eseguire

### Direttamente (ambiente locale)

```bash
pip install -r requirements.txt
PYTHONPATH=lib python main.py
```

### Con Docker Hub (immagine pre-built)

```bash
docker pull amorlot/titanic:latest
docker run -it --rm amorlot/titanic bash

# Avvia il training
python3 main.py

# Lancia i test
pytest tests/ -v

# Valutazione locale con accuracy e std per ogni modello
python3 evaluate_local.py
```

### Build locale

```bash
docker build -t titanic .
docker run --rm \
  -v $(pwd)/data/submissions:/app/data/submissions \
  titanic python main.py
```

Il volume `-v` è necessario per salvare i CSV di submission sul filesystem host.

---

## Configurazione — `config.yaml`

### `loader`

```yaml
loader:
  train_path: data/train.csv
  test_path: data/test.csv
  target_col: Survived          # colonna target
  drop_cols:                    # colonne da rimuovere al caricamento
    - PassengerId
    - Ticket
  drop_missing_thresh: 0.8      # droppa colonne con più dell'80% di NaN
```

---

### `feature_engineering`

Feature specifiche per il dataset Titanic. Genera:

| Feature | Descrizione |
|---|---|
| `TitleGroup` | Gruppo del titolo estratto da `Name` (Mr, Mrs, Master, Officer, Noble, Rare) |
| `AgeGroup` | Fascia d'età (Child / Teen / Adult / Middle / Senior) |
| `FamilySize` | `SibSp + Parch + 1` |
| `IsAlone` | 1 se viaggia da solo |
| `FarePerPerson` | `Fare / FamilySize` |
| `SocialClass` | Score 1–9 che combina `Pclass` e tier relativo di `FarePerPerson` |
| `CabinKnown` | 1 se la cabina è nota |
| `CabinDeck` | Lettera del ponte (da `Cabin`) |

```yaml
feature_engineering:
  enabled: true
  drop_raw_fare_for:    # rimuove Fare grezzo per i modelli indicati (rimane FarePerPerson)
    - logreg
    - svm
```

---

### `cleaner`

Imputa i valori mancanti. Il fit avviene sul train set, il transform viene applicato anche al test set.

```yaml
cleaner:
  num_strategy: knn       # median | mean | most_frequent | knn
  cat_strategy: knn       # unknown | most_frequent | knn
  knn_neighbors: 5        # usato solo se una delle strategie è knn
```

- `knn`: usa KNNImputer su tutte le feature (numeriche e categoriali encodate ordinalmente)
- `unknown`: riempie i categoriali con la stringa `"unknown"`

---

### `outliers`

Rilevamento e trattamento degli outlier. Si applica selettivamente ai modelli indicati in `apply_to`.

```yaml
outliers:
  detection:
    - iqr                 # IQR × threshold
    - zscore              # media ± threshold × std
  treatment: winsorization
  iqr_threshold: 1.5
  zscore_threshold: 3.0
  cols:                   # colonne da analizzare (vuoto = tutti i numerici)
    - Fare
    - Age
  apply_to:               # solo questi modelli ricevono i dati winsorizzati
    - logreg
```

Se si usano entrambi `iqr` e `zscore`, i bound vengono calcolati con l'intersezione (il più restrittivo).

---

### `multicollinearity`

Rimuove una feature per ogni coppia con correlazione di Pearson `|r| > threshold`. Sceglie quale delle due droppare in base alla correlazione media con tutte le altre feature.

```yaml
multicollinearity:
  threshold: 0.85
  drop: true              # false = solo report, non droppa
  apply_to:
    - logreg
```

---

### `pca`

Riduzione dimensionalità applicata dopo l'encoding. Ogni modello in `apply_to` viene addestrato sia con che senza PCA; nel riepilogo finale viene tenuta la variante con CV score più elevato.

```yaml
pca:
  n_components: 0.95      # float = % varianza da mantenere, int = n componenti fisso
  apply_to:
    - logreg
```

---

### `encoder`

Applicato a tutte le feature dopo il preprocessing. Il target viene encodato separatamente con `LabelEncoder`.

```yaml
encoder:
  num_strategy: standard  # standard (StandardScaler) | minmax (MinMaxScaler) | none
  cat_strategy: ohe       # ohe (OneHotEncoder, drop='first') | ordinal (OrdinalEncoder)
```

---

### `models`

```yaml
models:
  active:                 # modelli da addestrare (in ordine)
    - logreg
    - xgboost
    - decision_tree
    - random_forest
    - svm
  cv: 5                   # fold per GridSearchCV e OOF evaluation
  scoring: f1_weighted    # metrica di ottimizzazione
  tune_threshold: false   # se true, cerca la soglia decisionale ottimale dopo il training
  threshold_scoring: f1   # f1 | f1_weighted | accuracy | recall | precision
```

Modelli disponibili: `logreg`, `xgboost`, `decision_tree`, `random_forest`, `svm`.

Ogni modello viene valutato con **OOF (out-of-fold)** sul training set: le predizioni vengono prodotte su dati che il modello non ha mai visto durante il proprio fold di training.

---

### `model_cache`

Serializza i modelli addestrati su disco. Se il config non cambia (stesso hash MD5), al prossimo run i modelli vengono caricati invece di essere riaddestrati.

```yaml
model_cache:
  enabled: false
  path: models/
```

---

### `ensemble`

Combina le predizioni di più modelli già addestrati. Viene valutato con OOF aggregando le probabilità di ogni membro sul proprio dataset.

```yaml
ensemble:
  enabled: true
  voting: soft            # soft (media pesata delle probabilità) | hard (voto di maggioranza)
  weights: auto           # auto = usa CV score di ogni membro come peso
  # weights:              # oppure pesi manuali (normalizzati automaticamente)
  #   xgboost: 2
  #   logreg: 1
  #   svm: 1
  members:                # devono essere presenti in models.active
    - xgboost
    - logreg
    - svm
```

**Soft voting** è quasi sempre migliore di hard voting perché sfrutta l'incertezza della probabilità invece della sola decisione binaria.

**Consigli sui membri**: l'ensemble guadagna quando i modelli fanno errori su campioni diversi. Combinare modelli strutturalmente diversi (tree boosting + lineare + margine) è più efficace di combinare più alberi.

---

### `output`

```yaml
output:
  submission_dir: data/submissions/
```

---

## Libreria — `lib/`

### `GenericLoader`
Carica il dataset da CSV o da UCI ML Repository (via `ucimlrepo`). Normalizza i valori mancanti comuni (`?`, `N/A`, `none`, ecc.), droppa le colonne configurate e quelle con troppi NaN.

### `GenericCleaner`
Imputa i valori mancanti con pattern `fit/transform` (fit solo sul train, transform su train e test). Supporta median, mean, most_frequent, KNN.

### `GenericOutlierHandler`
Rileva outlier con IQR e/o Z-score e applica winsorization. Pattern `fit/transform`.

### `MulticollinearityHandler`
Trova coppie di feature con `|r| > threshold` e droppa automaticamente quella con correlazione media più alta. Pattern `fit/transform`.

### `PCAReducer`
Wrapper di `sklearn.decomposition.PCA` con pattern `fit/transform`. Può essere configurato con varianza target (es. `0.95`) o numero fisso di componenti.

### `FeatureEngineer`
Feature engineering specifico per Titanic. Crea `TitleGroup`, `AgeGroup`, `FamilySize`, `IsAlone`, `FarePerPerson`, `SocialClass`, `CabinKnown`, `CabinDeck`. Imputa `Age` con la mediana per `TitleGroup` (più precisa della mediana globale).

### `GenericEncoder`
Scala le feature numeriche (StandardScaler o MinMaxScaler) e codifica le categoriali (OHE o Ordinal). Gestisce anche l'encoding e decoding del target.

### `GenericEda`
Analisi esplorativa: bilanciamento classi, matrice di correlazione.

### `AbstractModel` (base)
Classe base astratta. Espone:
- `train()` — da implementare nelle sottoclassi
- `predict()` — usa la soglia decisionale ottimale se impostata
- `evaluate()` — accuracy, precision, recall, f1 sul test set
- `oof_report()` — classification report su predizioni out-of-fold
- `tune_threshold()` — cerca la soglia decisionale ottimale via OOF

### Modelli
| Classe | Algoritmo | Grid Search su |
|---|---|---|
| `GenericLogreg` | Logistic Regression | penalty, C, solver, l1_ratio |
| `GenericXGBoost` | XGBoost | n_estimators, max_depth, learning_rate, subsample, min_child_weight |
| `GenericDecisionTree` | Decision Tree | criterion, max_depth, min_samples_split, min_samples_leaf |
| `GenericRandomForest` | Random Forest | n_estimators, max_depth, max_features, min_samples_split |
| `GenericSVM` | SVM (LIBSVM) | kernel (rbf/linear), C, gamma |
| `GenericEnsemble` | Ensemble pesato | — (nessun training, combina modelli già addestrati) |

---

## Da aggiungere

### Funzionalità mancanti

- **Nuovi modelli**: aggiungere una classe in `lib/models/` che estende `AbstractModel` e implementa `train()`, poi registrarla in `_MODEL_REGISTRY` in `main.py`.
- **Stacking ensemble**: attualmente l'ensemble fa solo voting. Un meta-learner (es. LogReg addestrata sulle probabilità OOF dei membri) potrebbe migliorare le performance.
- **Feature selection automatica**: nessun meccanismo di selezione delle feature oltre alla rimozione di multicollinearità. Si potrebbe aggiungere `SelectFromModel` o RFE configurabili da YAML.
- **Hyperparameter tuning avanzato**: il grid search è a griglia fissa. Sostituirlo con `RandomizedSearchCV` o `Optuna` renderebbe l'esplorazione più efficiente.
- **Dataset generici**: `FeatureEngineer` è hardcoded per Titanic. Per usare la libreria su altri dataset serve una classe di feature engineering generica o un sistema di plugin.
- **Valutazione sul validation set**: attualmente non c'è un vero held-out set locale. Si potrebbe aggiungere uno split train/validation esplicito in `config.yaml` per stimare meglio le performance reali.
- **Report HTML/PDF**: i risultati vengono stampati solo a console. Un report esportabile (con grafici confusion matrix, feature importance, curva ROC) sarebbe utile.
