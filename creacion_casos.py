import pandas as pd
import os
import re

BASE_PATH = r"C:\Users\Admin\Desktop\Proyecto open banking\base_limpia.csv"
OUTPUT_DIR = r"C:\Users\Admin\Desktop\Proyecto open banking\casos_uso"

print("📌 Cargando base limpia...")
df = pd.read_csv(BASE_PATH, encoding="latin-1")
print("✔ Base cargada")

# ============================
# NORMALIZAR NOMBRES DE COLUMNAS
# ============================

# Convierte nombres a mayúsculas, elimina espacios dobles y espacios al inicio/final
def normalizar(col):
    col = col.upper()
    col = re.sub(r"\s+", " ", col)
    col = col.strip()
    return col

df.columns = [normalizar(c) for c in df.columns]

print("\n===== COLUMNAS NORMALIZADAS =====")
for c in df.columns:
    print(c)

# Crear carpeta de salida si no existe
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================
# VARIABLES CLAVE
# ============================

# Identificar columna de saldos electrónicos
col_ahorros_elec = [c for c in df.columns if "ELECTRONICAS" in c and "SALDO" in c]
if not col_ahorros_elec:
    raise ValueError("No encontré la columna de SALDO TOTAL CTA AHORROS ELECTRONICAS.")
col_ahorros_elec = col_ahorros_elec[0]

# Identificar columna de saldos tradicionales
col_ahorros = [c for c in df.columns if "SALDO TOTAL CTA AHORROS" in c and "ELECTRONICAS" not in c]
if not col_ahorros:
    raise ValueError("No encontré la columna de SALDO TOTAL CTA AHORROS.")
col_ahorros = col_ahorros[0]

# Crear TOTAL_AHORROS (tradicional + electrónica)
df["TOTAL_AHORROS"] = df[col_ahorros] + df[col_ahorros_elec]

# Crear brecha de género: diferencia hombres - mujeres
df["BRECHA_AHORROS_GENERO"] = df.get("SALDO CTA AHORRO HOMBRES", 0) - df.get("SALDO CTA AHORRO MUJERES", 0)

# Proporciones hombre/mujer sobre total de cuentas
df["PROP_MUJERES"] = df.get("NRO CTA AHORRO MUJERES", 0) / (df.get("NRO TOTAL CTA AHORROS", 0) + 1)
df["PROP_HOMBRES"] = df.get("NRO CTA AHORRO HOMBRES", 0) / (df.get("NRO TOTAL CTA AHORROS", 0) + 1)

# ============================
# CASO C — Baja inclusión
# ============================

# Filtra municipios con:
# - Menores ahorros (bottom 40%)
# - Menor cantidad de productos
# - Pocos corresponsales activos (< 5)
cond1 = df["TOTAL_AHORROS"] < df["TOTAL_AHORROS"].quantile(0.40)
cond2 = df["NRO PROD DEPOSITO NIVEL NACIONAL"] < df["NRO PROD DEPOSITO NIVEL NACIONAL"].quantile(0.40)
cond3 = df.get("NRO CORRESPONSALES ACTIVOS", 0) < 5

caso_C = df[cond1 & cond2 & cond3].copy()

# Si queda vacío, tomar los 60 municipios más débiles
if caso_C.empty:
    caso_C = df.sort_values("TOTAL_AHORROS").head(60).copy()
    print("⚠ Caso C estaba vacío → Se generó usando los 60 municipios con menor ahorro.")

C_PATH = os.path.join(OUTPUT_DIR, "caso_C_baja_inclusion.csv")
caso_C.to_csv(C_PATH, index=False, encoding="latin-1")
print(f"✔ Archivo creado: {C_PATH}")

# ============================
# CASO D — Brecha de género
# ============================

# Selecciona los municipios donde la brecha absoluta está en el top 25%
caso_D = df[df["BRECHA_AHORROS_GENERO"].abs() > df["BRECHA_AHORROS_GENERO"].quantile(0.75)].copy()

# Crea brecha relativa: brecha / total ahorros
caso_D["BRECHA_RELATIVA"] = caso_D["BRECHA_AHORROS_GENERO"] / (df["TOTAL_AHORROS"] + 1)

D_PATH = os.path.join(OUTPUT_DIR, "caso_D_brecha_genero.csv")
caso_D.to_csv(D_PATH, index=False, encoding="latin-1")
print(f"✔ Archivo creado: {D_PATH}")

# ============================
# CASO E — Oportunidades
# ============================

# Busca municipios con:
# - Muy pocos créditos de consumo (<10)
# - Altos ahorros (mayor a la mediana)
caso_E = df[
    (df.get("NRO CREDITO CONSUMO MUJERES", 0) + df.get("NRO CREDITO CONSUMO HOMBRES", 0) < 10) &
    (df["TOTAL_AHORROS"] > df["TOTAL_AHORROS"].median())
].copy()

# Calcula potencial de crédito como 20% del ahorro
caso_E["POTENCIAL_CREDITO"] = caso_E["TOTAL_AHORROS"] * 0.2

E_PATH = os.path.join(OUTPUT_DIR, "caso_E_oportunidades_producto.csv")
caso_E.to_csv(E_PATH, index=False, encoding="latin-1")
print(f"✔ Archivo creado: {E_PATH}")

print("\n🎉 Segmentación completada con éxito.")
