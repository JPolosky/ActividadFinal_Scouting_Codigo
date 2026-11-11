"""
Actividad final - Entrenamiento y exportación del modelo de scouting.

Pasos:
  1. Cargar los jugadores desde MySQL (tabla players) con MySQLConnector
  2. Preprocesamiento y target Élite (percentil 75 del valor de mercado)
  3. Split train/test con stratify e imputación de estatura por mediana de la posición
  4. get_dummies de la posición
  5. Árbol sin poda vs. árbol con poda (overfitting)
  6. Dummy y class_weight='balanced' (paradoja de la exactitud)
  7. Random Forest con GridSearchCV
  8. Curvas ROC, importancia de variables y exportar el paquete con joblib

Ejecutar: python exportar_modelo.py
"""
import json
import os
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             f1_score, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier

from src.database.connection2 import MySQLConnector

plt.switch_backend("Agg")  # para guardar las gráficas sin abrir ventanas

SEED = 42
DIR_FIG = os.path.join("reports", "figures")
COLOR_A, COLOR_B, COLOR_C = "#1f6f8b", "#e07a3f", "#8a8f98"

QUERY = """
    SELECT
        p.player_id,
        p.name AS jugador,
        p.position AS posicion,
        p.height_in_cm AS estatura_cm,
        TIMESTAMPDIFF(YEAR, p.date_of_birth, CURDATE()) AS edad,
        p.market_value_in_eur AS valor_mercado
    FROM players p
    WHERE p.market_value_in_eur IS NOT NULL
      AND p.date_of_birth IS NOT NULL;
"""


# ---------- Funciones de preprocesamiento ----------
def imputar(df, medianas_estatura, mediana_global_estatura, mediana_global_edad):
    # estatura: mediana de la posición y si no hay, la global; edad: mediana global
    df = df.copy()
    df["estatura_cm"] = df["estatura_cm"].fillna(df["posicion"].map(medianas_estatura))
    df["estatura_cm"] = df["estatura_cm"].fillna(mediana_global_estatura)
    df["edad"] = df["edad"].fillna(mediana_global_edad)
    return df


def armar_X(df, columnas=None):
    # edad + estatura + dummies de posición. drop_first solo al entrenar (como en clase);
    # con columnas dadas se usa reindex para no perder ninguna posición
    dummies = pd.get_dummies(df[["posicion"]], drop_first=columnas is None, dtype=int)
    X = pd.concat([df[["edad", "estatura_cm"]], dummies], axis=1)
    if columnas is not None:
        X = X.reindex(columns=columnas, fill_value=0)
    return X


def metricas(modelo, X, y):
    pred = modelo.predict(X)
    proba = modelo.predict_proba(X)[:, 1]
    return {
        "accuracy": round(accuracy_score(y, pred), 4),
        "precision": round(precision_score(y, pred, zero_division=0), 4),
        "recall": round(recall_score(y, pred, zero_division=0), 4),
        "f1": round(f1_score(y, pred, zero_division=0), 4),
        "auc": round(roc_auc_score(y, proba), 4),
    }


def main():
    print("=================================================================")
    print("  ENTRENAMIENTO, EVALUACIÓN Y EXPORTACIÓN DEL MODELO DE SCOUTING")
    print("=================================================================\n")
    os.makedirs("models", exist_ok=True)
    os.makedirs(DIR_FIG, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                         "font.size": 10})
    resumen = {"fecha_entrenamiento": datetime.now().isoformat(timespec="seconds"),
               "sklearn": sklearn.__version__}

    # 1. Cargar datos
    db = MySQLConnector()
    df = db.extract_data(QUERY)
    db.close()

    # 2. Preprocesamiento (igual que en clase)
    df["estatura_cm"] = pd.to_numeric(df["estatura_cm"], errors="coerce")
    df["edad"] = pd.to_numeric(df["edad"], errors="coerce")
    df["valor_mercado"] = pd.to_numeric(df["valor_mercado"], errors="coerce")
    df["posicion"] = df["posicion"].fillna("Sin Posicion")
    resumen["n_registros"] = int(len(df))
    resumen["nulos_originales"] = {c: int(df[c].isna().sum()) for c in ["estatura_cm", "edad", "posicion"]}
    resumen["posiciones"] = {k: int(v) for k, v in df["posicion"].value_counts().items()}
    resumen["edad_describe"] = df["edad"].describe().round(2).to_dict()
    resumen["valor_describe"] = df["valor_mercado"].describe().round(0).to_dict()

    umbral_elite = float(df["valor_mercado"].quantile(0.75))
    df["es_elite"] = (df["valor_mercado"] >= umbral_elite).astype(int)
    resumen["umbral_p75_eur"] = umbral_elite
    resumen["distribucion_target"] = {str(k): int(v) for k, v in df["es_elite"].value_counts().items()}
    print(f"Umbral P75 = {umbral_elite:,.0f} EUR | Élite = {df['es_elite'].mean():.1%}")

    # Perfil descriptivo de los Élite (para el análisis del informe)
    resumen["tasa_elite_posicion"] = df.groupby("posicion")["es_elite"].mean().round(4).to_dict()
    grupos_edad = pd.cut(df["edad"], bins=[0, 20, 23, 26, 29, 32, 35, 100],
                         labels=["≤20", "21-23", "24-26", "27-29", "30-32", "33-35", "≥36"])
    tasa_edad = df.groupby(grupos_edad, observed=False)["es_elite"].mean()
    resumen["tasa_elite_edad"] = {str(k): round(float(v), 4) for k, v in tasa_edad.items()}
    grupos_est = pd.cut(df["estatura_cm"], bins=[0, 175, 180, 185, 190, 250],
                        labels=["<175", "175-179", "180-184", "185-189", "≥190"], right=False)
    tasa_est = df.groupby(grupos_est, observed=False)["es_elite"].mean()
    resumen["tasa_elite_estatura"] = {str(k): round(float(v), 4) for k, v in tasa_est.items()}

    # 3. Split estratificado. Se hace antes de imputar para no usar datos de test
    df_train, df_test = train_test_split(df, test_size=0.2, stratify=df["es_elite"], random_state=SEED)
    resumen["n_train"], resumen["n_test"] = int(len(df_train)), int(len(df_test))
    resumen["proporcion_elite"] = {"train": round(df_train["es_elite"].mean(), 4),
                                   "test": round(df_test["es_elite"].mean(), 4)}

    # 4. Imputación (medianas calculadas solo con train)
    medianas_estatura = df_train.groupby("posicion")["estatura_cm"].median().to_dict()
    mediana_global_estatura = float(df_train["estatura_cm"].median())
    mediana_global_edad = float(df_train["edad"].median())
    resumen["medianas_estatura"] = medianas_estatura
    resumen["mediana_global_estatura"] = mediana_global_estatura
    resumen["mediana_global_edad"] = mediana_global_edad
    df_train = imputar(df_train, medianas_estatura, mediana_global_estatura, mediana_global_edad)
    df_test = imputar(df_test, medianas_estatura, mediana_global_estatura, mediana_global_edad)

    # 5. Variables dummy
    X_train = armar_X(df_train)
    columnas_X = X_train.columns.tolist()
    X_test = armar_X(df_test, columnas_X)
    y_train, y_test = df_train["es_elite"].values, df_test["es_elite"].values
    resumen["columnas_X"] = columnas_X
    print(f"Variables: {columnas_X}\nTrain = {len(X_train)} | Test = {len(X_test)}")

    # ---------- Overfitting: árbol sin poda vs. con poda ----------
    arbol_libre = DecisionTreeClassifier(random_state=SEED).fit(X_train, y_train)
    arbol_poda = DecisionTreeClassifier(max_depth=6, min_samples_leaf=50,
                                        random_state=SEED).fit(X_train, y_train)
    overfit = {}
    for nombre, m in [("Árbol sin restricción", arbol_libre), ("Árbol con poda previa", arbol_poda)]:
        overfit[nombre] = {"train": round(accuracy_score(y_train, m.predict(X_train)), 4),
                           "test": round(accuracy_score(y_test, m.predict(X_test)), 4),
                           "auc_train": round(roc_auc_score(y_train, m.predict_proba(X_train)[:, 1]), 4),
                           "auc_test": round(roc_auc_score(y_test, m.predict_proba(X_test)[:, 1]), 4),
                           "profundidad": int(m.get_depth()), "hojas": int(m.get_n_leaves())}
        overfit[nombre]["brecha"] = round(overfit[nombre]["train"] - overfit[nombre]["test"], 4)
    resumen["overfitting"] = overfit
    print("Overfitting:", json.dumps(overfit, ensure_ascii=False))

    # probar profundidades de 1 a 25 para ver dónde empieza el sobreajuste
    profundidades = list(range(1, 26))
    acc_tr, acc_te = [], []
    for prof in profundidades:
        m = DecisionTreeClassifier(max_depth=prof, random_state=SEED).fit(X_train, y_train)
        acc_tr.append(accuracy_score(y_train, m.predict(X_train)))
        acc_te.append(accuracy_score(y_test, m.predict(X_test)))
    resumen["curva_complejidad"] = {"max_depth": profundidades, "train": [round(a, 4) for a in acc_tr],
                                    "test": [round(a, 4) for a in acc_te]}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    etiquetas = list(overfit.keys())
    x = np.arange(len(etiquetas))
    b1 = axes[0].bar(x - 0.18, [overfit[k]["train"] for k in etiquetas], 0.36, label="Train", color=COLOR_A)
    b2 = axes[0].bar(x + 0.18, [overfit[k]["test"] for k in etiquetas], 0.36, label="Test", color=COLOR_B)
    axes[0].bar_label(b1, fmt="%.3f", fontsize=8)
    axes[0].bar_label(b2, fmt="%.3f", fontsize=8)
    axes[0].set_xticks(x, etiquetas)
    axes[0].set_ylim(0.5, 1.05)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy Train vs. Test")
    axes[0].legend(frameon=False, loc="upper right")
    axes[1].plot(profundidades, acc_tr, "-o", ms=3, color=COLOR_A, label="Train")
    axes[1].plot(profundidades, acc_te, "-o", ms=3, color=COLOR_B, label="Test")
    axes[1].axvline(6, ls="--", color=COLOR_C, lw=1)
    axes[1].text(6.4, min(acc_te) + 0.005, "max_depth = 6\n(poda elegida)", fontsize=8, color="#555")
    axes[1].set_xlabel("max_depth")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Curva de complejidad del árbol")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(DIR_FIG, "01_overfitting_train_vs_test.png"))
    plt.close(fig)

    # ---------- Paradoja de la exactitud y class_weight ----------
    dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    arbol_bal = DecisionTreeClassifier(max_depth=6, min_samples_leaf=50, class_weight="balanced",
                                       random_state=SEED).fit(X_train, y_train)
    resumen["paradoja_exactitud"] = {
        "Dummy (clase mayoritaria)": metricas(dummy, X_test, y_test),
        "Árbol con poda": metricas(arbol_poda, X_test, y_test),
        "Árbol con poda + balanced": metricas(arbol_bal, X_test, y_test),
    }
    print("Paradoja:", json.dumps(resumen["paradoja_exactitud"], ensure_ascii=False))

    # ---------- Random Forest + GridSearchCV ----------
    grid = {
        "n_estimators": [100, 200],
        "max_depth": [4, 8, 12],
        "min_samples_leaf": [1, 20, 50],
        "max_features": ["sqrt", None],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    busqueda = GridSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=1),
        grid, scoring="roc_auc", cv=cv, n_jobs=-1, refit=True,
    ).fit(X_train, y_train)
    rf = busqueda.best_estimator_
    resumen["gridsearch"] = {"mejores_parametros": busqueda.best_params_,
                             "mejor_auc_cv": round(float(busqueda.best_score_), 4),
                             "combinaciones": int(len(busqueda.cv_results_["params"])), "folds": 5}
    resumen["random_forest_train"] = metricas(rf, X_train, y_train)
    resumen["random_forest_test"] = metricas(rf, X_test, y_test)
    print("GridSearch:", busqueda.best_params_, f"AUC CV = {busqueda.best_score_:.4f}")
    print("RF test:", resumen["random_forest_test"])

    # ---------- Evaluación: ROC, importancia y matriz de confusión ----------
    fig, ax = plt.subplots(figsize=(5.8, 5))
    for nombre, m, color in [("Árbol único (poda + balanced)", arbol_bal, COLOR_B),
                             ("Random Forest optimizado", rf, COLOR_A)]:
        proba = m.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        ax.plot(fpr, tpr, lw=2, color=color, label=f"{nombre} (AUC = {roc_auc_score(y_test, proba):.3f})")
    ax.plot([0, 1], [0, 1], "--", color=COLOR_C, lw=1, label="Azar (AUC = 0.5)")
    ax.set_xlabel("Tasa de Falsos Positivos (FPR)")
    ax.set_ylabel("Tasa de Verdaderos Positivos (TPR / Recall)")
    ax.set_title("Curva ROC comparativa (conjunto de Test)")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(DIR_FIG, "02_curva_roc_comparativa.png"))
    plt.close(fig)

    importancias = pd.Series(rf.feature_importances_, index=columnas_X).sort_values()
    resumen["importancia_variables"] = {k: round(float(v), 4) for k, v in importancias[::-1].items()}
    fig, ax = plt.subplots(figsize=(7, 3.8))
    bars = ax.barh(importancias.index, importancias.values, color=COLOR_A)
    ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
    ax.set_xlabel("Importancia de Gini (reducción media de impureza)")
    ax.set_title("Importancia de variables (Random Forest)")
    fig.tight_layout()
    fig.savefig(os.path.join(DIR_FIG, "03_importancia_variables.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.6, 4))
    ConfusionMatrixDisplay.from_estimator(rf, X_test, y_test, display_labels=["Estándar", "Élite"],
                                          cmap="Blues", colorbar=False, ax=ax, values_format="d")
    ax.set_title("Matriz de confusión RF (Test)")
    fig.tight_layout()
    fig.savefig(os.path.join(DIR_FIG, "04_matriz_confusion_rf.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.hist(np.log10(df["valor_mercado"]), bins=40, color=COLOR_C)
    ax.axvline(np.log10(umbral_elite), color=COLOR_B, lw=2, label=f"P75 = {umbral_elite:,.0f} EUR")
    ax.set_xlabel("log10(valor de mercado en EUR)")
    ax.set_ylabel("Jugadores")
    ax.set_title("Distribución del valor de mercado y umbral Élite")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(DIR_FIG, "00_distribucion_target.png"))
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, serie, titulo in [(axes[0], pd.Series(resumen["tasa_elite_posicion"]).sort_values(), "Posición"),
                              (axes[1], tasa_edad, "Edad (años)"), (axes[2], tasa_est, "Estatura (cm)")]:
        b = ax.bar([str(i) for i in serie.index], serie.values * 100, color=COLOR_A)
        ax.bar_label(b, fmt="%.0f%%", fontsize=8)
        ax.axhline(25, ls="--", color=COLOR_B, lw=1)
        ax.set_title(f"% de Élite por {titulo.lower()}")
        ax.tick_params(axis="x", labelsize=8, rotation=20)
    axes[0].set_ylabel("% de jugadores Élite")
    fig.tight_layout()
    fig.savefig(os.path.join(DIR_FIG, "06_tasa_elite_por_variable.png"))
    plt.close(fig)

    resumen["classification_report_rf"] = classification_report(
        y_test, rf.predict(X_test), target_names=["Estándar", "Élite"], output_dict=True)

    # Guardar el modelo junto con lo que necesita la API (mismas claves que en clase)
    paquete_despliegue = {
        "modelo": rf,
        "columnas_X": columnas_X,
        "medianas_estatura": medianas_estatura,
        "mediana_global_estatura": mediana_global_estatura,
        "mediana_global_edad": mediana_global_edad,
        "umbral_elite_eur": umbral_elite,
        "metricas_test": resumen["random_forest_test"],
        "mejores_parametros": busqueda.best_params_,
        "version": "2.0.0",
        "entrenado": resumen["fecha_entrenamiento"],
    }
    ruta_modelo = os.path.join("models", "modelo_scouting_rf.joblib")
    joblib.dump(paquete_despliegue, ruta_modelo, compress=3)
    with open(os.path.join("reports", "metricas.json"), "w", encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nPaquete de modelo exportado exitosamente en: '{ruta_modelo}'")


if __name__ == "__main__":
    main()
