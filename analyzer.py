import os
import numpy as np
import pandas as pd
from collections import Counter
import glob

def analyze_file(filepath):
    """
    Analyzes a single data file and prints the statistics.
    """
    print(f"\n--- Analizando archivo: {os.path.basename(filepath)} ---")

    pedidos_ids = []
    pedidos_quantities = Counter()
    capacidades_ids = []
    capacidades_values = Counter()
    ni_values = []

    with open(filepath, 'r') as f:
        try:
            line_content = f.readline().strip().split()
            if not line_content:
                print("Archivo vacío o con formato incorrecto en la primera línea.")
                return
            n, m, k = map(int, line_content)

            # Pedidos
            for _ in range(n):
                line = list(map(int, f.readline().strip().split()))
                if not line: continue
                ni = line[0]
                ni_values.append(ni)
                items = line[1:]
                for i in range(0, len(items), 2):
                    item_id = items[i]
                    item_cnt = items[i+1]
                    pedidos_ids.append(item_id)
                    pedidos_quantities[item_id] += item_cnt
            
            # Capacidades
            for _ in range(k):
                line = list(map(int, f.readline().strip().split()))
                if not line: continue
                
                # The format is just pairs of id and capacity.
                items = line
                if len(line) > 1 and len(line) % 2 == 0:
                     items = line
                elif len(line) > 1 and len(line) % 2 != 0: # k_i might be the first element, so we skip it
                    items = line[1:]

                for i in range(0, len(items), 2):
                    item_id = items[i]
                    capacidad = items[i+1]
                    capacidades_ids.append(item_id)
                    capacidades_values[item_id] += capacidad
                        
        except (ValueError, IndexError) as e:
            print(f"Error procesando el archivo: {e}. Omitiendo.")
            return
        except StopIteration:
            print(f"Error: El archivo {os.path.basename(filepath)} parece estar incompleto o mal formado.")
            return

    if not pedidos_ids and not capacidades_ids and not ni_values:
        print("No se encontraron datos válidos en este archivo.")
        return

    # Análisis de Pedidos
    if pedidos_ids:
        print("\n1. Análisis de Pedidos:")
        id_counts_pedidos = Counter(pedidos_ids)
        df_pedidos = pd.DataFrame.from_dict(id_counts_pedidos, orient='index', columns=['apariciones'])
        df_pedidos['cantidad_total'] = df_pedidos.index.map(pedidos_quantities)

        print("  a. Estadísticas de apariciones de IDs:")
        if id_counts_pedidos:
            print(f"    - ID más común: {id_counts_pedidos.most_common(1)[0][0]} (aparece {id_counts_pedidos.most_common(1)[0][1]} veces)")
        print(df_pedidos['apariciones'].describe().apply(lambda x: f"{x:,.2f}").to_string())

        print("\n  b. Estadísticas de cantidad total por ID:")
        print(df_pedidos['cantidad_total'].describe().apply(lambda x: f"{x:,.2f}").to_string())
    
    # Análisis de Capacidades
    if capacidades_ids:
        print("\n2. Análisis de Capacidades:")
        id_counts_capacidades = Counter(capacidades_ids)
        df_capacidades = pd.DataFrame.from_dict(id_counts_capacidades, orient='index', columns=['apariciones'])
        df_capacidades['capacidad_total'] = df_capacidades.index.map(capacidades_values)

        print("  a. Estadísticas de apariciones de IDs:")
        if id_counts_capacidades:
            print(f"    - ID más común: {id_counts_capacidades.most_common(1)[0][0]} (aparece {id_counts_capacidades.most_common(1)[0][1]} veces)")
        print(df_capacidades['apariciones'].describe().apply(lambda x: f"{x:,.2f}").to_string())

        print("\n  b. Estadísticas de capacidad total por ID:")
        print(df_capacidades['capacidad_total'].describe().apply(lambda x: f"{x:,.2f}").to_string())

    # Análisis de n_i
    if ni_values:
        print("\n3. Análisis de n_i (items por pedido):")
        df_ni = pd.Series(ni_values)
        print(df_ni.describe().apply(lambda x: f"{x:,.2f}").to_string())

def analyze_directory(directory_path):
    print(f"Analizando el directorio: {directory_path}")
    print("=" * 40)

    all_files = sorted(glob.glob(os.path.join(directory_path, '*.txt')))
    if not all_files:
        print("No se encontraron archivos .txt en el directorio.")
        return

    for filepath in all_files:
        analyze_file(filepath)


if __name__ == "__main__":
    # Path to the specific file of interest
    file_to_analyze = 'datasets/a/instance_0010.txt'
    
    # Check if the file exists before analyzing
    if os.path.exists(file_to_analyze):
        # We reuse the function that analyzes a single file
        analyze_file(file_to_analyze)
    else:
        print(f"Error: El archivo {file_to_analyze} no fue encontrado.") 