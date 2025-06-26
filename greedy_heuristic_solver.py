"""
Implementación de una heurística Greedy para encontrar una solución factible
(Lower Bound) para un problema de optimización fraccional.

La heurística sigue una lógica de dos fases:
1. CONSTRUCCIÓN: Añade iterativamente las órdenes (`x_i`) que ofrecen la mejor
   ganancia neta hasta alcanzar la cota de demanda inferior `L`. La ganancia
   se calcula como el beneficio de la orden menos el coste de abrir nuevos
   pasillos (`y_k`) necesarios para cubrir la demanda de items.
2. AJUSTE: Si en la fase de construcción se excede la cota superior `R`,
   elimina de forma inteligente las órdenes que menos perjudiquen al
   objetivo hasta que la solución vuelva a ser factible.

Uso (PowerShell):
Get-Content .\\datasets\\a\\instance_0010.txt | python greedy_heuristic_solver.py
"""
import sys
import time
import math
from collections import Counter

def parse_input(lines):
    """
    Parsea los datos de entrada. Idéntico a los otros solvers.
    """
    try:
        n_str, m_str, k_str = lines.pop(0).strip().split()
        n, m, k = int(n_str), int(m_str), int(k_str)

        orders = []
        all_item_ids = set()
        for i in range(n):
            parts = list(map(int, lines.pop(0).strip().split()))
            order_items = {parts[j]: parts[j+1] for j in range(1, len(parts), 2)}
            total_quantity = sum(order_items.values())
            orders.append({'id': i, 'items': Counter(order_items), 'total_quantity': total_quantity})
            all_item_ids.update(order_items.keys())

        aisles = []
        for j in range(k):
            parts = list(map(int, lines.pop(0).strip().split()))
            aisle_items = {parts[i]: parts[i+1] for i in range(1, len(parts), 2)}
            aisles.append({'id': j, 'items': Counter(aisle_items)})
        
        l_bound, r_bound = map(int, lines.pop(0).strip().split())
        
        print(f"INFO: n={n}, m={m}, k={k}, L={l_bound}, R={r_bound}")
        print(f"INFO: {len(all_item_ids)} items únicos encontrados.")
        return n, k, orders, aisles, l_bound, r_bound, sorted(list(all_item_ids))
    except (IndexError, ValueError) as e:
        print(f"ERROR: Formato de entrada inválido. {e}")
        return None

def find_heuristic_lower_bound(n, k, orders, aisles, l, r, all_items, lambda_val):
    """
    Implementa la heurística de construcción y ajuste descrita.
    """
    X = set()  # Índices de x_i = 1
    Y = set()  # Índices de y_k = 1

    A = Counter()  # Demanda acumulada por item: A_j = sum_{i in X} a_ij
    B = Counter()  # Capacidad acumulada por item: B_j = sum_{k in Y} b_kj
    C_X = 0        # Beneficio acumulado: C_X = sum_{i in X} c_i
    
    # --- FASE 1: CONSTRUCCIÓN (alcanzar L) ---
    print("\n--- FASE 1: Construcción hasta alcanzar L ---")
    
    while C_X < l:
        candidates_evaluation = []
        
        # Evaluar cada orden candidata i no seleccionada
        for i in range(n):
            if i in X:
                continue

            order = orders[i]
            c_i = order['total_quantity']
            
            # Calcular deficiencias que esta orden crearía
            deficiencies = Counter()
            for item_j, demand_ij in order['items'].items():
                if A[item_j] + demand_ij > B[item_j]:
                    deficiencies[item_j] = (A[item_j] + demand_ij) - B[item_j]

            # Sub-heurística de cobertura para estimar el coste de abrir pasillos
            Y_add = set()
            B_temp = B.copy()
            
            # Mientras haya deficiencias que cubrir...
            while any(d > 0 for d in deficiencies.values()):
                best_k = -1
                max_gain = -1
                
                # Encontrar el mejor pasillo k no abierto
                for k_cand in range(k):
                    if k_cand in Y or k_cand in Y_add:
                        continue
                    
                    # Calcular cuánta deficiencia cubre este pasillo
                    gain_k = sum(min(aisles[k_cand]['items'][item_j], deficiencies[item_j]) for item_j in deficiencies)
                    
                    if gain_k > max_gain:
                        max_gain = gain_k
                        best_k = k_cand
                
                if best_k == -1: # No se pueden cubrir más deficiencias
                    break
                    
                Y_add.add(best_k)
                for item_j, cap_kj in aisles[best_k]['items'].items():
                    B_temp[item_j] += cap_kj
                    if A[item_j] + order['items'][item_j] <= B_temp[item_j]:
                        deficiencies[item_j] = 0

            cost_i = lambda_val * len(Y_add)
            delta_i = c_i - cost_i
            candidates_evaluation.append({'id': i, 'delta': delta_i, 'y_add': Y_add})
        
        if not candidates_evaluation:
            print("AVISO: No quedan más candidatos para añadir.")
            break

        # Elegir el mejor candidato
        positive_candidates = [c for c in candidates_evaluation if c['delta'] > 0]
        if positive_candidates:
            best_candidate = max(positive_candidates, key=lambda c: c['delta'])
        else:
            best_candidate = max(candidates_evaluation, key=lambda c: c['delta'])

        # Aplicar el mejor candidato a la solución
        i_selected = best_candidate['id']
        X.add(i_selected)
        Y.update(best_candidate['y_add'])
        
        # Actualizar contadores
        C_X += orders[i_selected]['total_quantity']
        for item_j, demand_ij in orders[i_selected]['items'].items():
            A[item_j] += demand_ij
        for k_added in best_candidate['y_add']:
            for item_j, cap_kj in aisles[k_added]['items'].items():
                B[item_j] += cap_kj
        
        print(f"  Añadido x_{i_selected}. C_X acumulado: {C_X:.2f}. |Y|: {len(Y)}. Delta: {best_candidate['delta']:.2f}")

    if C_X < l:
        print(f"AVISO: La fase de construcción terminó sin alcanzar L. C_X final: {C_X:.2f}")

    # --- FASE 2: AJUSTE (reducir a R si es necesario) ---
    print("\n--- FASE 2: Ajuste para no exceder R ---")
    
    # Esta fase es compleja y una implementación simple puede ser suficiente
    # Por simplicidad, eliminaremos la orden con menor c_i hasta cumplir R.
    # Una heurística más avanzada evaluaría el impacto en Y.
    while C_X > r:
        if not X:
            break

        # Encontrar la orden en X con la menor cantidad total (c_i)
        i_to_remove = min(X, key=lambda i: orders[i]['total_quantity'])
        
        X.remove(i_to_remove)
        C_X -= orders[i_to_remove]['total_quantity']
        
        print(f"  Ajuste: Eliminado x_{i_to_remove}. Nuevo C_X: {C_X:.2f}")
        
        # Nota: no se eliminan pasillos Y para mantener la factibilidad del resto.
        # Una mejora sería recalcular los Y necesarios.

    print("\n--- Heurística Finalizada ---")
    if not Y: # Evitar división por cero si no se abrió ningún pasillo
        return X, Y, 0
    
    final_objective = C_X / len(Y)
    return X, Y, final_objective

if __name__ == "__main__":
    input_data = sys.stdin.readlines()
    if not input_data:
        print("ERROR: No se proporcionaron datos de entrada.")
        sys.exit(1)

    parsed_data = parse_input(input_data)
    if parsed_data:
        n, k, orders, aisles, l, r, all_items = parsed_data
        
        # Estimar un lambda razonable para guiar la heurística.
        # Un valor alto desincentiva abrir pasillos. Un valor bajo, lo contrario.
        # Usamos un promedio del rango teórico.
        lambda_heuristic = (l / k + r) / 2 if k > 0 else r

        print(f"\nINFO: Iniciando heurística con lambda = {lambda_heuristic:.4f}")
        
        start_time = time.time()
        
        selected_X, selected_Y, objective_val = find_heuristic_lower_bound(
            n, k, orders, aisles, l, r, all_items, lambda_heuristic
        )
        
        duration = time.time() - start_time
        
        print("\n--- RESULTADO DE LA HEURÍSTICA ---")
        print(f"Tiempo de ejecución: {duration:.4f} segundos")
        print(f"Valor del Objetivo (Lower Bound): {objective_val:.4f}")
        print(f"Pedidos seleccionados ({len(selected_X)}): {sorted(list(selected_X))}")
        print(f"Pasillos abiertos ({len(selected_Y)}): {sorted(list(selected_Y))}")
        
        # Auditoría final de la solución generada
        total_quantity = sum(orders[i]['total_quantity'] for i in selected_X)
        print(f"\nAuditoría:")
        print(f"  Cantidad Total: {total_quantity:.2f} (Rango: [{l}, {r}])")
        
        is_feasible = l <= total_quantity <= r
        
        final_demand = Counter()
        for i in selected_X:
            final_demand.update(orders[i]['items'])
            
        final_capacity = Counter()
        for k_idx in selected_Y:
            final_capacity.update(aisles[k_idx]['items'])
        
        violations = 0
        for item, demand in final_demand.items():
            if demand > final_capacity[item]:
                print(f"  VIOLACIÓN en Item {item}: Demanda={demand}, Capacidad={final_capacity[item]}")
                violations += 1
                is_feasible = False
                
        if is_feasible:
            print("  La solución generada es FACTIBLE.")
        else:
            print(f"  La solución generada NO ES FACTIBLE ({violations} violaciones).")

