import os
import glob
from collections import Counter

def analyze_b_file(filepath):
    """
    Analyzes a single file from directory 'b' to find order statistics
    and the top 5 most frequently ordered items with their total demand and capacity.
    """
    print(f"--- Análisis del archivo: {os.path.basename(filepath)} ---")

    with open(filepath, 'r') as f:
        try:
            line_content = f.readline().strip().split()
            if not line_content:
                print("  - Archivo vacío o con formato incorrecto.\n")
                return
            n, m, k = map(int, line_content)

            ni_values = []
            pedidos_id_counts = Counter()
            pedidos_quantities = Counter()
            capacidades_values = Counter()

            if n == 0:
                print("  - El archivo no contiene pedidos (n=0).\n")
                return

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
                    pedidos_id_counts[item_id] += 1
                    pedidos_quantities[item_id] += item_cnt
            
            # Capacidades
            for _ in range(k):
                line = list(map(int, f.readline().strip().split()))
                if not line: continue
                
                items = line
                if len(line) > 1 and len(line) % 2 != 0:
                    items = line[1:]

                for i in range(0, len(items), 2):
                    item_id = items[i]
                    capacidad = items[i+1]
                    capacidades_values[item_id] += capacidad

            # Analysis and Print
            if not ni_values:
                print("  - No se pudieron leer los datos de los pedidos.\n")
                return

            avg_items_per_order = sum(ni_values) / len(ni_values) if ni_values else 0
            max_items_per_order = max(ni_values) if ni_values else 0
            
            print(f"  - Items por pedido: Promedio={avg_items_per_order:.2f}, Máximo={max_items_per_order}")

            if not pedidos_id_counts:
                print("  - No se encontraron IDs de items en los pedidos.")
            else:
                top_5_items = pedidos_id_counts.most_common(5)
                print("  - Top 5 IDs más pedidos (ID, Veces, Demanda Total, Capacidad Total):")
                for item_id, count in top_5_items:
                    demanda = pedidos_quantities[item_id]
                    capacidad = capacidades_values.get(item_id, 0)
                    print(f"    - ID: {item_id:<6} (pedido {count} veces) | Demanda: {demanda:<5} | Capacidad: {capacidad:<5}")
            
            print("")

        except (ValueError, IndexError, StopIteration) as e:
            print(f"  - Error procesando el archivo: {e}.\n")

def analyze_directory(directory_path):
    print(f"Iniciando análisis detallado para el directorio: {directory_path}")
    print("=" * 70)
    all_files = sorted(glob.glob(os.path.join(directory_path, '*.txt')))
    if not all_files:
        print("No se encontraron archivos .txt en el directorio.")
        return
    for filepath in all_files:
        analyze_b_file(filepath)

if __name__ == "__main__":
    analyze_directory('datasets/b') 