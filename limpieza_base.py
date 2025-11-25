import pandas as pd



PATH = r"C:\Users\Admin\Downloads\Inclusión_Financiera_20251121.csv"

print("📌 Cargando archivo con dtype=str (modo seguro)...")

df = pd.read_csv(
    PATH,
    encoding="latin-1",
    dtype=str,          
    on_bad_lines="skip" 
)

print("✔ Archivo cargado correctamente (modo seguro)")
print("📌 Columnas detectadas:")
print(df.columns)



print("🧽 Limpiando dataset...")


df.columns = df.columns.str.strip()


cols_numericas = [
    c for c in df.columns
    if ("NRO" in c.upper()) or ("MONTO" in c.upper()) or ("SALDO" in c.upper())
]

print(f"📌 Columnas numéricas detectadas: {len(cols_numericas)}")

for col in cols_numericas:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(".", "", regex=False)      
        .str.replace(",", ".", regex=False)     
        .str.replace(" ", "", regex=False)
        .str.replace("-", "0", regex=False)
    )
    
    df[col] = pd.to_numeric(df[col], errors="coerce")

df[cols_numericas] = df[cols_numericas].fillna(0)

df = df.dropna(how="all")

print("✔ Limpieza completada")


OUTPUT = r"C:\Users\Admin\Desktop\Proyecto open banking\base_limpia.csv"

df.to_csv(OUTPUT, index=False, encoding="latin-1")

print("📁 Archivo limpio guardado en:")
print(OUTPUT)
