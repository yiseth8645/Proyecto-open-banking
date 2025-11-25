import pandas as pd   # Librería para manipulación de datos

# Ruta del archivo original proporcionado por la Superfinanciera
PATH = r"C:\Users\Admin\Downloads\Inclusión_Financiera_20251121.csv"

print("📌 Cargando archivo con dtype=str (modo seguro)...")

# Lectura del archivo en "modo seguro":
# - dtype=str → carga todo como texto, evita errores por datos corruptos
# - on_bad_lines="skip" → ignora filas dañadas sin detener el programa
df = pd.read_csv(
    PATH,
    encoding="latin-1",   # Maneja caracteres especiales en español
    dtype=str,            # Previene fallos por tipos incorrectos
    on_bad_lines="skip"   # Salta filas inválidas
)

print("✔ Archivo cargado correctamente (modo seguro)")
print("📌 Columnas detectadas:")
print(df.columns)

print("🧽 Limpiando dataset...")

# Limpieza de encabezados:
# strip() elimina espacios al inicio y al final de los nombres
df.columns = df.columns.str.strip()

# Identificación automática de columnas numéricas:
# Se seleccionan aquellas que contienen palabras clave típicas de variables financieras
cols_numericas = [
    c for c in df.columns
    if ("NRO" in c.upper()) or ("MONTO" in c.upper()) or ("SALDO" in c.upper())
]

print(f"📌 Columnas numéricas detectadas: {len(cols_numericas)}")

# Limpieza y conversión de cada columna numérica detectada
for col in cols_numericas:
    df[col] = (
        df[col]
        .astype(str)                  # Convertir siempre a texto primero
        .str.replace(".", "", regex=False)      # Quita separadores de miles
        .str.replace(",", ".", regex=False)     # Convierte coma decimal a punto
        .str.replace(" ", "", regex=False)      # Quita espacios internos
        .str.replace("-", "0", regex=False)     # Reemplaza '-' por 0
    )

    # Conversión final a número (float), NaN en caso de error
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Rellenar NaN numéricos con 0 (evita errores en cálculos posteriores)
df[cols_numericas] = df[cols_numericas].fillna(0)

# Eliminar filas completamente vacías
df = df.dropna(how="all")

print("✔ Limpieza completada")

# Guardar archivo limpio y estandarizado
OUTPUT = r"C:\Users\Admin\Desktop\Proyecto open banking\base_limpia.csv"
df.to_csv(OUTPUT, index=False, encoding="latin-1")

print("Archivo limpio guardado en:")
print(OUTPUT)
