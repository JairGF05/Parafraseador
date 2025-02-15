import pandas as pd
import os
from transformers import pipeline
from concurrent.futures import ThreadPoolExecutor

# Cargar modelo de parafraseo más rápido
paraphrase_pipeline = pipeline("text2text-generation", model="t5-small")

def parafrasear_texto(texto):
    try:
        if not texto or not isinstance(texto, str):
            return ""
        resultado = paraphrase_pipeline(f"{texto}", max_length=200, min_length=30, do_sample=False)
        return resultado[0]['generated_text']
    except Exception as e:
        print(f"Error al parafrasear: {e}")
        return texto

def procesar_fila(row):
    try:
        if row.isnull().all():
            return None
        if "text" in row:
            texto_original = str(row["text"])
            texto_parafraseado = parafrasear_texto(texto_original)
            return texto_parafraseado
    except Exception as e:
        print(f"Error al procesar la fila: {e}")
    return None

def extraer_y_parafrasear(input_excel, output_file):
    try:
        hojas = pd.read_excel(input_excel, sheet_name=None, header=None)
    except Exception as e:
        print(f"Error al leer el archivo Excel: {e}")
        return
    
    with open(output_file, 'w', encoding='utf-8') as archivo_salida:
        for nombre_hoja, data in hojas.items():
            print(f"Procesando hoja: {nombre_hoja}")
            
            try:
                encabezados = data.iloc[0].tolist()
                data.columns = encabezados
                data = data.iloc[1:]
            except Exception as e:
                print(f"Error al procesar la hoja {nombre_hoja}: {e}")
                continue
            
            archivo_salida.write(f"\n--- Hoja: {nombre_hoja} ---\n")
            
            with ThreadPoolExecutor() as executor:
                resultados = list(executor.map(procesar_fila, [row for _, row in data.iterrows()]))
            
            for resultado in resultados:
                if resultado:
                    archivo_salida.write(f"{resultado}\n")
    
    print(f"Texto parafraseado guardado en: {output_file}")

def main():
    input_excel = "corpus_diezmil.xlsx"
    output_file = "parafraseo.txt"
    extraer_y_parafrasear(input_excel, output_file)

if __name__ == "__main__":
    main()