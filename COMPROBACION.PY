import pandas as pd

path_C = r"C:\Users\Admin\Desktop\Proyecto open banking\casos_uso\caso_C_baja_inclusion.csv"
path_D = r"C:\Users\Admin\Desktop\Proyecto open banking\casos_uso\caso_D_brecha_genero.csv"
path_E = r"C:\Users\Admin\Desktop\Proyecto open banking\casos_uso\caso_E_oportunidades_producto.csv"

print(r"C:\Users\Admin\Desktop\Proyecto open banking\casos_uso\caso_C_baja_inclusion.csv")
print(pd.read_csv(path_C, encoding="latin-1").head(), "\n")

print(r"C:\Users\Admin\Desktop\Proyecto open banking\casos_uso\caso_D_brecha_genero.csv")
print(pd.read_csv(path_D, encoding="latin-1").head(), "\n")

print(r"C:\Users\Admin\Desktop\Proyecto open banking\casos_uso\caso_E_oportunidades_producto.csv")
print(pd.read_csv(path_E, encoding="latin-1").head(), "\n")
