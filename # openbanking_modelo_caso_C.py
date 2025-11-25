
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error, r2_score

PATH = r"C:\Users\Admin\Desktop\Proyecto open banking\casos_uso\caso_C_baja_inclusion.csv"

print("📌 Cargando Caso C...")
df = pd.read_csv(PATH, encoding="latin-1")
print(f"✔ Filas: {df.shape[0]}, Columnas: {df.shape[1]}\n")

# ----- asegurarnos TOTAL_AHORROS existe -----
if "TOTAL_AHORROS" not in df.columns:
    print("⚠ Recalculando TOTAL_AHORROS (no estaba presente)")

    possible_a = [c for c in df.columns if "SALDO TOTAL CTA AHORROS" in c]
    possible_e = [c for c in df.columns if "ELECTRONICAS" in c and "SALDO" in c]
    if possible_a and possible_e:
        df["TOTAL_AHORROS"] = df[possible_a[0]].fillna(0) + df[possible_e[0]].fillna(0)
        print(f"  -> Usando columnas: {possible_a[0]} + {possible_e[0]}")
    else:
        
        saldo_cols = [c for c in df.columns if "SALDO" in c]
        if saldo_cols:
            df["TOTAL_AHORROS"] = df[saldo_cols].fillna(0).sum(axis=1)
            print("  -> Usando SUM(saldo_cols)")
        else:
            raise SystemExit("❌ No se pudo construir TOTAL_AHORROS automát. Verifica columnas.")


print("\n--- Diagnóstico inicial ---")
print("TOTAL_AHORROS: min,max,median: ", df["TOTAL_AHORROS"].min(), df["TOTAL_AHORROS"].max(), df["TOTAL_AHORROS"].median())
print("Total NaNs por columna (top 10):")
print(df.isna().sum().sort_values(ascending=False).head(10))


df["OBJ_BIN"] = (df["TOTAL_AHORROS"] < df["TOTAL_AHORROS"].median()).astype(int)
print("\nDistribución OBJ_BIN:")
print(df["OBJ_BIN"].value_counts())


exclude = ["TIPO DE ENTIDAD","CODIGO DE LA ENTIDAD","NOMBRE DE LA ENTIDAD",
           "FECHA DE CORTE","UNIDAD DE CAPTURA","DEPARTAMENTO","MUNICIPIO",
           "RENGLON","TIPO"]
cols = [c for c in df.columns if c not in exclude]
num_df = df[cols].select_dtypes(include=[np.number]).copy()

print(f"\nVariables numéricas detectadas: {num_df.shape[1]}")
print("Primeras columnas:", list(num_df.columns[:15]))

nunique = num_df.nunique()
const_cols = nunique[nunique <= 1].index.tolist()
low_var = [c for c in num_df.columns if num_df[c].std() == 0]
print(f"Columnas constantes (n={len(const_cols)}): {const_cols[:8]}")
num_df.drop(columns=const_cols, inplace=True)


num_df = num_df.fillna(num_df.median())

print("\nResumen X (describe):")
print(num_df.describe().T[['count','mean','std','min','50%','max']].head(15))

if "TOTAL_AHORROS" in num_df.columns:
    num_df = num_df.drop(columns=["TOTAL_AHORROS"])

zero_var_cols = [c for c in num_df.columns if np.isclose(num_df[c].var(), 0)]
print(f"\nColumnas con varianza ~0: {len(zero_var_cols)} (muestra: {zero_var_cols[:8]})")

y = df["OBJ_BIN"]
if y.nunique() < 2:
    print("\n⚠ OBJ_BIN es monoclase. Creando etiqueta por quantiles (q=2) a partir de TOTAL_AHORROS.")
    df["OBJ_Q2"] = pd.qcut(df["TOTAL_AHORROS"].rank(method="first"), q=2, labels=False)
    print("Distrib OBJ_Q2:", df["OBJ_Q2"].value_counts().to_dict())
    y = df["OBJ_Q2"]


X = num_df.copy()

if X.shape[1] == 0:
    raise SystemExit("❌ No hay variables numéricas útiles para modelar después de limpieza.")


if y.nunique() < 2:
    print("\n⚠ Aun no hay 2 clases. Se intentará Regresión sobre TOTAL_AHORROS.")
   
    X_train, X_test, y_train, y_test = train_test_split(X, df["TOTAL_AHORROS"], test_size=0.3, random_state=42)
    reg = RandomForestRegressor(n_estimators=300, random_state=42)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    print("\nRegresión - RMSE:", mean_squared_error(y_test, y_pred, squared=False))
    print("Regresión - R2:", r2_score(y_test, y_pred))
    importances = pd.Series(reg.feature_importances_, index=X.columns).sort_values(ascending=False).head(20)
    print("\nTop variables (regresión):")
    print(importances)
    # plot
    importances.head(15)[::-1].plot(kind="barh", figsize=(10,8), title="Top importancias (regresión)")
    plt.xlabel("Importancia")
    plt.tight_layout()
    plt.show()
else:
    
    print("\n➡ Entrando en clasificación (RandomForestClassifier)")
    print("Clases y counts:\n", y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("\n===== classification_report =====")
    print(classification_report(y_test, y_pred))

    importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False).head(20)
    print("\nTop variables (clasificación):")
    print(importances)

    
    importances.head(15)[::-1].plot(kind="barh", figsize=(10,8), title="Top importancias (clasificación)")
    plt.xlabel("Importancia")
    plt.tight_layout()
    plt.show()

print("\n✅ Fin diagnóstico/modelado. Copia la salida completa y me la pegas aquí.")

