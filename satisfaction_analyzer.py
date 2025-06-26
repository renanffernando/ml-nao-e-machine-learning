import os
import glob
from collections import Counter

def analyze_file_satisfaction(filepath):
    """
    For a single file, finds the most frequent order ID and checks
    if its total demand can be satisfied by its total capacity.
    """
    with open(filepath, 'r') as f:
        try:
            # Data reading
            line_content = f.readline().strip().split()
            if not line_content:
                print(f"Archivo: {os.path.basename(filepath)} - Vacío o formato incorrecto.")
                print("-" * 20)
                return
            n, m, k = map(int, line_content)

            pedidos_quantities = Counter()
            capacidades_values = Counter()
            pedidos_id_counts = Counter()

            # Pedidos
            for _ in range(n):
                line = list(map(int, f.readline().strip().split()))
                if not line: continue
                items = line[1:]
                for i in range(0, len(items), 2):
                    item_id = items[i]
                    item_cnt = items[i+1]
                    pedidos_id_counts[item_id] += 1
                    pedidos_quantities[item_id] += item_cnt

            # Capacidades
            for _ in range(k):
                line = list(map(int, f.readline().strip().split()))
                if not line: continue
                
                items = line
                if len(line) > 1 and len(line) % 2 != 0:
                    items = line[1:] # k_i might be the first element, so we skip it

                for i in range(0, len(items), 2):
                    item_id = items[i]
                    capacidad = items[i+1]
                    capacidades_values[item_id] += capacidad
            
            # Analysis
            if not pedidos_id_counts:
                print(f"Archivo: {os.path.basename(filepath)}")
                print("  - No se encontraron pedidos en este archivo.")
                print("-" * 20)
                return

            most_common_id, frequency = pedidos_id_counts.most_common(1)[0]
            total_demand = pedidos_quantities[most_common_id]
            total_capacity = capacidades_values.get(most_common_id, 0) # Use .get for safety
            
            satisfaction_status = "SÍ" if total_capacity >= total_demand else "NO"

            # Print results for the file
            print(f"Archivo: {os.path.basename(filepath)}")
            print(f"  - ID más frecuente en pedidos: {most_common_id} (aparece {frequency} veces)")
            print(f"  - Demanda total para el ID {most_common_id}: {total_demand}")
            print(f"  - Capacidad total para el ID {most_common_id}: {total_capacity}")
            print(f"  - ¿Se puede satisfacer la demanda?: {satisfaction_status}")
            print("-" * 20)

        except (ValueError, IndexError, StopIteration) as e:
            print(f"Error procesando el archivo {os.path.basename(filepath)}: {e}. Saltando archivo.")

def analyze_directory(directory_path):
    print(f"Iniciando análisis de satisfacción para el directorio: {directory_path}")
    print("=" * 40)
    all_files = sorted(glob.glob(os.path.join(directory_path, '*.txt')))
    if not all_files:
        print("No se encontraron archivos .txt en el directorio.")
        return
    for filepath in all_files:
        analyze_file_satisfaction(filepath)

if __name__ == "__main__":
    analyze_directory('datasets/a') 