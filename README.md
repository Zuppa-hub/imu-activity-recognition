# IMU Activity Recognition Project

## Progetto: Introduction to Data Science and Software Engineering

Un progetto completo di machine learning per il riconoscimento di attività umane usando dati di accelerometro e giroscopio raccolti da un cellulare.

### 📊 Dataset

Il dataset contiene dati sensoriali raccolti durante tre attività diverse:

1. **sitting_table**: Cellulare fermo sul tavolo (activity statica)
2. **stairs_pocket**: Salire e scendere le scale con andamento regolare
3. **walking_pocket**: Camminata normale

**Sensori utilizzati:**
- Accelerometro (x, y, z)
- Giroscopio (x, y, z)

**Formato dei dati grezzi:**
```
time,seconds_elapsed,z,y,x
1767620467350470700,0.143471,1.065960,0.185703,0.104458
```

### 🔄 Pipeline

#### 1️⃣ **Data Cleaning** (`data_cleaning.py`)

Trasforma i dati grezzi in dati analizzabili:

- **Caricamento**: Legge i file CSV di accelerometro e giroscopio
- **Allineamento**: Unisce i segnali accelerometro e giroscopio per timestamp più vicino
- **Rimozione outliers**: Utilizza il metodo IQR (Interquartile Range)
  - Calcola Q1, Q3 e IQR
  - Rimuove valori fuori dall'intervallo [Q1 - 1.5×IQR, Q3 + 1.5×IQR]
- **Smoothing**: Applica un filtro rolling mean con finestra di 5 campioni
- **Salvataggio**: Salva i dati puliti in `data/cleaned/<activity>.csv`
- **Visualizzazione**: Crea grafici di confronto Raw vs Cleaned

**Output:**
```
data/cleaned/
├── sitting_table.csv
├── stairs_pocket.csv
├── walking_pocket.csv
├── sitting_table_accel_comparison.png
├── sitting_table_gyro_comparison.png
├── stairs_pocket_accel_comparison.png
├── stairs_pocket_gyro_comparison.png
├── walking_pocket_accel_comparison.png
└── walking_pocket_gyro_comparison.png
```

#### 2️⃣ **Feature Engineering** (`feature_engineering.py`)

Estrae caratteristiche dai dati puliti:

- **Sliding Window**: Estrae finestre temporali di 2 secondi con step di 1 secondo
- **Feature per finestra**:
  - Mean, std, min, max per ogni asse (x, y, z)
  - Signal magnitude: √(x² + y² + z²)
  - Caratteristiche sia da accelerometro che da giroscopio
- **Total features**: 28 feature per campione
- **Etichettamento**: Assegna l'etichetta di attività a ogni feature vector
- **Shuffle**: Mescola il dataset
- **Salvataggio**: Crea `data/features.csv` pronto per il ML

**Output:**
```
data/features.csv
- 125 campioni
- 28 feature + colonna activity
- Distribuiti: 51 sitting_table, 32 stairs_pocket, 42 walking_pocket
```

#### 3️⃣ **Machine Learning** (`model.py`)

Addestra due classificatori per il riconoscimento di attività:

**Pipeline:**
1. Carica il dataset di feature
2. Split 70% training, 30% test
3. Standardizzazione delle feature (StandardScaler)
4. Training e valutazione di due classificatori

**Classificatori:**
- **Random Forest**: 100 estimatori
- **K-Nearest Neighbors**: k=5

**Metriche di valutazione:**
- Accuracy
- Confusion Matrix
- Classification Report (precision, recall, f1-score)

**Output:**
```
results/
├── confusion_matrix_rf.png
└── confusion_matrix_knn.png
```

### 🚀 Come eseguire il progetto

```bash
# 1. Data Cleaning
python3 data_cleaning.py

# 2. Feature Engineering
python3 feature_engineering.py

# 3. Machine Learning
python3 model.py
```

### 📈 Risultati

Entrambi i classificatori raggiungono un'**accuratezza del 100%**:

```
Random Forest Accuracy:      1.0000
K-Nearest Neighbors Accuracy: 1.0000

Classificazione perfetta su tutte e tre le attività:
- sitting_table: precision=1.00, recall=1.00, f1=1.00
- stairs_pocket: precision=1.00, recall=1.00, f1=1.00
- walking_pocket: precision=1.00, recall=1.00, f1=1.00
```

La matrice di confusione mostra **nessun errore di classificazione**.

### 📁 Struttura del progetto

```
IMU_Project/
├── data_cleaning.py           # Step 1: Pulizia dati
├── feature_engineering.py     # Step 2: Estrazione feature
├── model.py                   # Step 3: Training ML
├── README.md                  # Questa documentazione
├── data/
│   ├── raw/
│   │   ├── sitting_table/
│   │   ├── stairs_pocket/
│   │   └── walking_pocket/
│   ├── cleaned/               # Output Step 1
│   └── features.csv           # Output Step 2
├── notebook/
│   └── exploration.ipynb      # Analisi esplorativa
└── results/                   # Output Step 3
    ├── confusion_matrix_rf.png
    └── confusion_matrix_knn.png
```

### 🛠️ Dipendenze

```
pandas
numpy
matplotlib
scikit-learn
seaborn
```

### 💡 Note tecniche

**Funzioni principali:**

- `remove_outliers_iqr()`: Rimuove outliers usando il metodo IQR
- `load_and_clean_activity()`: Carica, allinea e pulisce i dati sensoriali
- `plot_raw_vs_cleaned()`: Visualizza il confronto Raw vs Cleaned
- `sliding_window()`: Estrae finestre temporali dai dati
- `extract_features_from_window()`: Estrae feature da una singola finestra
- `train_and_evaluate_classifier()`: Addestra e valuta i classificatori

### 🎓 Concetti chiave appresi

1. **Data Cleaning**: Gestione di dati grezzi, outlier removal, signal smoothing
2. **Time-Series Feature Engineering**: Sliding windows, feature extraction
3. **Machine Learning**: Classification, model evaluation, hyperparameter tuning
4. **Model Comparison**: Valutazione e confronto di diversi algoritmi
5. **Data Visualization**: Plots comparativi, confusion matrices

---

**Autore**: Andrea Cazzato  
**Data**: 28 Gennaio 2026  
**Corso**: Introduction to Data Science and Software Engineering
