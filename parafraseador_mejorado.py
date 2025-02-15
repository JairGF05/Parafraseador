import pandas as pd
import os
from transformers import pipeline

# carga rapida del modelo de parafraseo
paraphrase_pipeline = pipeline(
    "text2text-generation", model="t5-small", device=0) 


def parafrasear_texto(textos):
    try:
        if not textos:
            return []

        valid_texts = [t for t in textos if isinstance(t, str) and t.strip()]
        if not valid_texts:
            return ["" for _ in textos]

        # Proceso batch de parafraseo
        results = paraphrase_pipeline(
            valid_texts, max_length=200, min_length=30, do_sample=False)
        return [res["generated_text"] for res in results]

    except Exception as e:
        print(f"Error during paraphrasing: {e}")
        return textos


def procesar_fila(row):
    try:
        texto_original = getattr(row, "text", None)
        if not isinstance(texto_original, str) or not texto_original.strip():
            return ""
        return texto_original
    except Exception as e:
        print(f"Error processing row: {e}")
        return ""


def extraer_y_parafrasear(input_excel, output_file):
    try:
        hojas = pd.read_excel(input_excel, sheet_name=None, header=None)
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return

    with open(output_file, "w", encoding="utf-8") as archivo_salida:
        for nombre_hoja, data in hojas.items():
            print(f"Processing sheet: {nombre_hoja}")

            try:
                encabezados = data.iloc[0].tolist()
                data.columns = encabezados
                data = data.iloc[1:].dropna(
                    subset=["text"]) 
            except Exception as e:
                print(f"Error processing sheet {nombre_hoja}: {e}")
                continue

            archivo_salida.write(f"\n--- Sheet: {nombre_hoja} ---\n")

            # Procesamiento eficiente usando itertuples
            textos = [procesar_fila(row)
                      for row in data.itertuples(index=False)]

            # Parafraseo en batchess
            batch_size = 8  # Ajustable dependiendo de la memoria disponible
            for i in range(0, len(textos), batch_size):
                resultados = parafrasear_texto(textos[i: i + batch_size])
                archivo_salida.writelines(f"{r}\n" for r in resultados if r)

    print(f"Paraphrased text saved to: {output_file}")


def main():
    input_excel = "corpus_diezmil.xlsx"
    output_file = "parafraseo.txt"
    extraer_y_parafrasear(input_excel, output_file)


if __name__ == "__main__":
    main()
