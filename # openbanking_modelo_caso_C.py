import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error, r2_score

PATH = r"C:\Users\Admin\Desktop\Proyecto open banking\casos_uso\caso_C_baja_inclusion.csv"

# =======================================================
# 1. CARGA DEL ARCHIVO
# =======================================================
print("📌 Cargando Caso C...")
df = pd.read_csv(PATH, encoding="latin-1")
print(f"✔ Filas: {df.shape[0]}, Columnas: {df.shape[1]}\n")

# =======================================================
# 2. RECONSTRUCCIÓN DE TOTAL_AHORROS (si es necesario)
# =======================================================
if "TOTAL_AHORROS" not in df.columns:
    print("⚠ Recalculando TOTAL_AHORROS (no estaba presente)")

    # Busca columnas correctas para sumarlas
    possible_a = [c for c in df.columns if "SALDO TOTAL CTA AHORROS" in c]
    possible_e = [c for c in df.columns if "ELECTRONICAS" in c and "SALDO" in c]

    if possible_a and possible_e:
        df["TOTAL_AHORROS"] = df[possible_a[0]].fillna(0) + df[possible_e[0]].fillna(0)
        print(f"  -> Usando columnas: {possible_a[0]} + {possible_e[0]}")
    else:
        # Último recurso: sumar todos los saldos
        saldo_cols = [c for c in df.columns if "SALDO" in c]
        df["TOTAL_AHORROS"] = df[saldo_cols].fillna(0).sum(axis=1)
        print("  -> Usando SUM(saldo_cols) porque faltan columnas clave")

# =======================================================
# 3. DIAGNÓSTICO INICIAL
# =======================================================
print("\n--- Diagnóstico inicial ---")
print("TOTAL_AHORROS: min,max,median: ",
      df["TOTAL_AHORROS"].min(),
      df["TOTAL_AHORROS"].max(),
      df["TOTAL_AHORROS"].median())

print("Total NaNs por columna (top 10):")
print(df.isna().sum().sort_values(ascending=False).head(10))

# =======================================================
# 4. CREAR EL TARGET BINARIO
# =======================================================
df["OBJ_BIN"] = (df["TOTAL_AHORROS"] < df["TOTAL_AHORROS"].median()).astype(int)
print("\nDistribución OBJ_BIN:")
print(df["OBJ_BIN"].value_counts())

# =======================================================
# 5. SELECCIÓN DE VARIABLES NUMÉRICAS ÚTILES
# =======================================================
exclude = [
    "TIPO DE ENTIDAD","CODIGO DE LA ENTIDAD","NOMBRE DE LA ENTIDAD",
    "FECHA DE CORTE","UNIDAD DE CAPTURA","DEPARTAMENTO","MUNICIPIO",
    "RENGLON","TIPO"
]

cols = [c for c in df.columns if c not in exclude]
num_df = df[cols].select_dtypes(include=[np.number]).copy()

print(f"\nVariables numéricas detectadas: {num_df.shape[1]}")
print("Primeras columnas:", list(num_df.columns[:15]))

# Elimina columnas constantes
nunique = num_df.nunique()
const_cols = nunique[nunique <= 1].index.tolist()
print(f"Columnas constantes (n={len(const_cols)}): {const_cols[:8]}")
num_df.drop(columns=const_cols, inplace=True)

# Rellena nulos con mediana
num_df = num_df.fillna(num_df.median())

print("\nResumen X (describe):")
print(num_df.describe().T[['count','mean','std','min','50%','max']].head(15))

# Evita fuga de información
if "TOTAL_AHORROS" in num_df.columns:
    num_df = num_df.drop(columns=["TOTAL_AHORROS"])

# Detecta columnas sin varianza
zero_var_cols = [c for c in num_df.columns if np.isclose(num_df[c].var(), 0)]
print(f"\nColumnas con varianza ~0: {len(zero_var_cols)}")

# =======================================================
# 6. SELECCIÓN DEL TIPO DE MODELADO (CLASIFICACIÓN O REGRESIÓN)
# =======================================================
y = df["OBJ_BIN"]

# Si solo hay una clase → usar cuantiles
if y.nunique() < 2:
    print("\n⚠ OBJ_BIN es monoclase. Se usará qcut para crear grupos por ahorros.")
    df["OBJ_Q2"] = pd.qcut(df["TOTAL_AHORROS"].rank(method="first"), q=2, labels=False)
    print("Distrib OBJ_Q2:", df["OBJ_Q2"].value_counts())
    y = df["OBJ_Q2"]

X = num_df.copy()

if X.shape[1] == 0:
    raise SystemExit("❌ No hay variables útiles tras la limpieza.")

# =======================================================
# 7A. REGRESIÓN (si no hay clases)
# =======================================================
if y.nunique() < 2:
    print("\n⚠ Aún no hay 2 clases → Modelo de Regresión.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, df["TOTAL_AHORROS"], test_size=0.3, random_state=42)

    reg = RandomForestRegressor(n_estimators=300, random_state=42)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    print("\nRegresión - RMSE:", mean_squared_error(y_test, y_pred, squared=False))
    print("Regresión - R2:", r2_score(y_test, y_pred))

    importances = pd.Series(reg.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nTop variables (regresión):")
    print(importances.head(20))

    importances.head(15)[::-1].plot(kind="barh", figsize=(10,8))
    plt.title("Top importancias (regresión)")
    plt.tight_layout()
    plt.show()

# =======================================================
# 7B. CLASIFICACIÓN (RandomForestClassifier)
# =======================================================
else:
    print("\n➡ Entrando en clasificación (RandomForestClassifier)")
    print("Clases y counts:\n", y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)

    clf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("\n===== classification_report =====")
    print(classification_report(y_test, y_pred))

    importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nTop variables (clasificación):")
    print(importances.head(20))

    importances.head(15)[::-1].plot(kind="barh", figsize=(10,8))
    plt.title("Top importancias (clasificación)")
    plt.tight_layout()
    plt.show()

print("\n✅ Fin diagnóstico/modelado.")
