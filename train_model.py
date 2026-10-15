import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import cross_validate, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def train_and_evaluate():
    print("=== [1] Loading Dataset ===")
    dataset_path = 'water_potability.csv'
    df = pd.read_csv(dataset_path)
    
    total_rows, total_cols = df.shape
    features = [c for c in df.columns if c != 'Potability']
    target_col = 'Potability'
    
    # Missing values stats
    missing_stats = df.isnull().sum().to_dict()
    class_distribution = df[target_col].value_counts().to_dict()
    class_percentages = (df[target_col].value_counts(normalize=True) * 100).round(2).to_dict()
    
    print(f"Dataset Shape: {total_rows} baris, {total_cols} kolom")
    print(f"Distribusi Target: {class_distribution} ({class_percentages})")
    
    print("\n=== [2] Preprocessing & Data Splitting ===")
    X = df[features]
    y = df[target_col]
    
    # Split 80% Train, 20% Test with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    majority_class = int(y_train.mode().iloc[0])
    majority_baseline_accuracy = float(
        accuracy_score(y_test, np.full(len(y_test), majority_class))
    )
    
    # Impute missing values with training set median (no data leakage)
    train_medians = X_train.median().to_dict()
    X_train_imp = X_train.fillna(train_medians)
    X_test_imp = X_test.fillna(train_medians)
    
    # Feature Scaling for MLP
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)
    
    # Save preprocessing artifacts
    joblib.dump(scaler, 'scaler.pkl')
    with open('medians.json', 'w') as f:
        json.dump(train_medians, f, indent=2)
        
    print("Train set shape:", X_train_imp.shape)
    print("Test set shape:", X_test_imp.shape)
    
    print("\n=== [3] Training Model Decision Tree ===")
    dt_model = DecisionTreeClassifier(
        criterion='entropy',
        max_depth=6,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    dt_model.fit(X_train_imp, y_train)
    y_pred_dt = dt_model.predict(X_test_imp)
    
    cm_dt = confusion_matrix(y_test, y_pred_dt)
    acc_dt = float(accuracy_score(y_test, y_pred_dt))
    prec_dt = float(precision_score(y_test, y_pred_dt, zero_division=0))
    rec_dt = float(recall_score(y_test, y_pred_dt, zero_division=0))
    f1_dt = float(f1_score(y_test, y_pred_dt, zero_division=0))
    
    print(f"Decision Tree Accuracy: {acc_dt*100:.2f}%")
    print(f"Decision Tree Precision: {prec_dt*100:.2f}%")
    print(f"Decision Tree Recall: {rec_dt*100:.2f}%")
    print(f"Decision Tree F1-Score: {f1_dt*100:.2f}%")
    print("Confusion Matrix DT:\n", cm_dt)
    
    joblib.dump(dt_model, 'model_dt.pkl')

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1_score': 'f1'
    }
    dt_cv_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('classifier', DecisionTreeClassifier(
            criterion='entropy',
            max_depth=6,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        ))
    ])
    dt_cv_scores = cross_validate(
        dt_cv_pipeline, X, y, cv=cv, scoring=scoring, n_jobs=1
    )
    dt_cross_validation = {
        metric: {
            'mean': round(float(np.mean(dt_cv_scores[f'test_{metric}'])) * 100, 2),
            'std': round(float(np.std(dt_cv_scores[f'test_{metric}'])) * 100, 2)
        }
        for metric in scoring
    }
    
    print("\n=== [4] Training Model Multilayer Perceptron (MLP) ===")
    mlp_model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        max_iter=1500,
        random_state=42
    )
    mlp_model.fit(X_train_scaled, y_train)
    y_pred_mlp = mlp_model.predict(X_test_scaled)
    
    cm_mlp = confusion_matrix(y_test, y_pred_mlp)
    acc_mlp = float(accuracy_score(y_test, y_pred_mlp))
    prec_mlp = float(precision_score(y_test, y_pred_mlp, zero_division=0))
    rec_mlp = float(recall_score(y_test, y_pred_mlp, zero_division=0))
    f1_mlp = float(f1_score(y_test, y_pred_mlp, zero_division=0))
    
    print(f"MLP Accuracy: {acc_mlp*100:.2f}%")
    print(f"MLP Precision: {prec_mlp*100:.2f}%")
    print(f"MLP Recall: {rec_mlp*100:.2f}%")
    print(f"MLP F1-Score: {f1_mlp*100:.2f}%")
    print("Confusion Matrix MLP:\n", cm_mlp)
    
    joblib.dump(mlp_model, 'model_mlp.pkl')

    mlp_cv_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            max_iter=1500,
            random_state=42
        ))
    ])
    mlp_cv_scores = cross_validate(
        mlp_cv_pipeline, X, y, cv=cv, scoring=scoring, n_jobs=1
    )
    mlp_cross_validation = {
        metric: {
            'mean': round(float(np.mean(mlp_cv_scores[f'test_{metric}'])) * 100, 2),
            'std': round(float(np.std(mlp_cv_scores[f'test_{metric}'])) * 100, 2)
        }
        for metric in scoring
    }
    
    print("\n=== [5] Generating Visualizations ===")
    os.makedirs('static/img', exist_ok=True)
    
    # 1. Confusion Matrix DT Plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Tidak Layak (0)', 'Layak (1)'],
                yticklabels=['Tidak Layak (0)', 'Layak (1)'],
                annot_kws={"size": 14, "weight": "bold"})
    plt.title('Confusion Matrix - Decision Tree', fontsize=14, pad=12, fontweight='bold', color='#1e293b')
    plt.xlabel('Prediksi Model', fontsize=12, labelpad=8)
    plt.ylabel('Kondisi Aktual', fontsize=12, labelpad=8)
    plt.tight_layout()
    plt.savefig('static/img/confusion_matrix_dt.png', dpi=300)
    plt.close()
    
    # 2. Confusion Matrix MLP Plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_mlp, annot=True, fmt='d', cmap='Greens', cbar=False,
                xticklabels=['Tidak Layak (0)', 'Layak (1)'],
                yticklabels=['Tidak Layak (0)', 'Layak (1)'],
                annot_kws={"size": 14, "weight": "bold"})
    plt.title('Confusion Matrix - Multilayer Perceptron (MLP)', fontsize=14, pad=12, fontweight='bold', color='#1e293b')
    plt.xlabel('Prediksi Model', fontsize=12, labelpad=8)
    plt.ylabel('Kondisi Aktual', fontsize=12, labelpad=8)
    plt.tight_layout()
    plt.savefig('static/img/confusion_matrix_mlp.png', dpi=300)
    plt.close()
    
    # 3. Feature Importance Plot (Decision Tree)
    feature_importances = pd.Series(dt_model.feature_importances_, index=features).sort_values(ascending=True)
    plt.figure(figsize=(8, 5))
    colors = sns.color_palette("mako", len(feature_importances))
    feature_importances.plot(kind='barh', color=colors)
    plt.title('Tingkat Kepentingan Fitur (Decision Tree)', fontsize=13, fontweight='bold', pad=12, color='#1e293b')
    plt.xlabel('Akumulasi Penurunan Impurity (Entropy)', fontsize=11)
    plt.ylabel('Atribut Parameter Kualitas Air', fontsize=11)
    plt.tight_layout()
    plt.savefig('static/img/feature_importance.png', dpi=300)
    plt.close()
    
    print("\n=== [6] Compiling Comparison & Metrics Data ===")
    # Journal baseline (Zaky et al., 2023)
    journal_metrics = {
        "title": "Performance Analysis of the Decision Tree Classification Algorithm on the Water Quality and Potability Dataset",
        "authors": "Umar Zaky, Ahmad Naswin, Sumiyatun, Aris Wahyu Murdiyanto",
        "journal": "Indonesian Journal of Data and Science (IJODAS)",
        "year": 2023,
        "volume": "Vol. 4, No. 3, Hal. 145-150",
        "doi": "https://doi.org/10.56705/ijodas.v4i3.113",
        "dataset_link": "https://www.kaggle.com/datasets/uom190346a/water-quality-and-potability",
        "method": "Decision Tree (5-Fold Cross Validation)",
        "accuracy": 54.33,
        "precision": 55.00,
        "recall": 54.40,
        "f1_score": 54.70,
        "fold_details": [
            {"fold": "K-1", "accuracy": 54.8, "precision": 54.2, "recall": 55.1, "f1": 55.5},
            {"fold": "K-2", "accuracy": 52.5, "precision": 54.1, "recall": 52.5, "f1": 52.3},
            {"fold": "K-3", "accuracy": 57.4, "precision": 57.5, "recall": 56.7, "f1": 56.2},
            {"fold": "K-4", "accuracy": 52.3, "precision": 52.1, "recall": 51.4, "f1": 53.4},
            {"fold": "K-5", "accuracy": 54.5, "precision": 56.8, "recall": 56.0, "f1": 56.6}
        ]
    }
    
    project_metrics = {
        "dataset": {
            "total_rows": total_rows,
            "total_cols": total_cols,
            "features": features,
            "train_count": len(X_train),
            "test_count": len(X_test),
            "missing_stats": missing_stats,
            "class_distribution": {str(k): int(v) for k, v in class_distribution.items()},
            "class_percentages": {str(k): float(v) for k, v in class_percentages.items()}
        },
        "majority_class_baseline": {
            "class": majority_class,
            "accuracy": round(majority_baseline_accuracy * 100, 2)
        },
        "decision_tree": {
            "accuracy": round(acc_dt * 100, 2),
            "precision": round(prec_dt * 100, 2),
            "recall": round(rec_dt * 100, 2),
            "f1_score": round(f1_dt * 100, 2),
            "cross_validation": dt_cross_validation,
            "confusion_matrix": {
                "tn": int(cm_dt[0, 0]),
                "fp": int(cm_dt[0, 1]),
                "fn": int(cm_dt[1, 0]),
                "tp": int(cm_dt[1, 1])
            },
            "feature_importance": {k: round(float(v), 4) for k, v in feature_importances.items()}
        },
        "multilayer_perceptron": {
            "accuracy": round(acc_mlp * 100, 2),
            "precision": round(prec_mlp * 100, 2),
            "recall": round(rec_mlp * 100, 2),
            "f1_score": round(f1_mlp * 100, 2),
            "cross_validation": mlp_cross_validation,
            "confusion_matrix": {
                "tn": int(cm_mlp[0, 0]),
                "fp": int(cm_mlp[0, 1]),
                "fn": int(cm_mlp[1, 0]),
                "tp": int(cm_mlp[1, 1])
            }
        },
        "comparison_analysis": {
            "persamaan": [
                "Menggunakan dataset kualitas dan kelayakan air (Water Quality and Potability Dataset) yang bersumber dari repositori Kaggle.",
                "Menggunakan 9 atribut parameter fisik dan kimiawi air yang sama persis (ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity) dengan target biner Potability (0 = Tidak Layak, 1 = Layak).",
                "Menerapkan algoritma Decision Tree Classifier untuk pemodelan klasifikasi data.",
                "Mengevaluasi kinerja model menggunakan metrik standar Accuracy, Precision, Recall, dan F1-Score/F-Measure melalui kerangka Confusion Matrix."
            ],
            "perbedaan": [
                f"Skema Evaluasi: Hasil utama proyek menggunakan holdout terstratifikasi 80:20 (Decision Tree akurasi {round(acc_dt*100, 2)}%; MLP {round(acc_mlp*100, 2)}%). Sebagai pembanding tambahan dengan protokol 5-fold jurnal, proyek juga menghitung cross-validation terstratifikasi pada seluruh dataset (Decision Tree akurasi rata-rata {dt_cross_validation['accuracy']['mean']}%; MLP {mlp_cross_validation['accuracy']['mean']}%). Perbandingan tetap tidak sepenuhnya setara karena rincian preprocessing dan konfigurasi model jurnal dapat berbeda.",
                "Penanganan Missing Values: Tabel dataset pada artikel mencatat 'Missing values: No', sedangkan berkas CSV proyek berisi nilai kosong pada ph, Sulfate, dan Trihalomethanes. Proyek mengisi nilai kosong dengan median yang dihitung hanya dari bagian training (atau bagian training fold pada cross-validation).",
                "Konfigurasi Model: Decision Tree menggunakan criterion entropy, max_depth=6, min_samples_split=10, dan min_samples_leaf=5. Model pembanding MLP menggunakan dua hidden layer berukuran (64, 32) serta StandardScaler. Kedua model dilatih dengan data dan parameter proyek.",
                "Implementasi: Artikel melaporkan hasil analitis, sedangkan proyek menyediakan form dan endpoint Flask untuk menguji data baru. Status deployment publik harus dibuktikan dengan URL aktif; keberadaan konfigurasi deployment saja belum membuktikan aplikasi sudah online."
            ],
            "penyebab_perbedaan": [
                "Perbedaan strategi validasi, imputasi, dan parameter model dapat mengubah skor; karena eksperimen tidak mengisolasi satu faktor, selisih skor tidak membuktikan bahwa satu preprocessing atau konfigurasi menjadi penyebab peningkatan.",
                "Decision Tree dan MLP memiliki karakteristik berbeda. Pada holdout proyek, akurasi saja tidak cukup: recall kelas Potability=1 juga perlu dilihat agar sampel positif yang terlewat terlihat.",
                "Dataset tidak seimbang (sekitar 61% kelas 0 dan 39% kelas 1), sehingga akurasi perlu ditafsirkan bersama precision, recall, F1-score, dan confusion matrix."
            ]
        },
        "journal_reference": journal_metrics
    }
    
    with open('metrics.json', 'w', encoding='utf-8') as f:
        json.dump(project_metrics, f, indent=2, ensure_ascii=False)
        
    print("\nAll models, preprocessors, visualizations, and metrics.json saved successfully!")

if __name__ == '__main__':
    train_and_evaluate()
