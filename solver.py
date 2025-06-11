"""
Script para resolver un problema de optimización fraccional entero-mixto
utilizando el algoritmo de Dinkelbach y un método de generación de restricciones.

Uso (PowerShell):
Get-Content .\\datasets\\a\\instance_0010.txt | python solver.py

Dependencias:
pip install docplex
"""
import sys
import random
import time
import math
from docplex.mp.model import Model
from collections import Counter

def parse_input(lines):
    """
    Parsea los datos de entrada. Devuelve las estructuras de datos y
    un conjunto de todos los items que aparecen en los pedidos.
    """
    print("INFO: Parseando datos de entrada...")
    
    try:
        n_str, m_str, k_str = lines.pop(0).strip().split()
        n, m, k = int(n_str), int(m_str), int(k_str)

        # Estructuras de datos para un acceso eficiente
        all_item_ids = set()

        # Leer pedidos (órdenes)
        orders = []
        for i in range(n):
            parts = list(map(int, lines.pop(0).strip().split()))
            order_items = {parts[j]: parts[j+1] for j in range(1, len(parts), 2)}
            total_quantity = sum(order_items.values())
            orders.append({'id': i, 'items': order_items, 'total_quantity': total_quantity})
            all_item_ids.update(order_items.keys())

        # Leer capacidades (pasillos)
        aisles = []
        for j in range(k):
            parts = list(map(int, lines.pop(0).strip().split()))
            # Se asume que el primer elemento es un contador y el resto son pares id-capacidad.
            aisle_items = {parts[i]: parts[i+1] for i in range(1, len(parts), 2)}
            aisles.append({'id': j, 'items': aisle_items})
            # NO actualizamos all_item_ids aquí, como sugirió el usuario.
            # Solo nos importan los items que tienen demanda.

        # Devolvemos todos los items que aparecen en los pedidos para la intersección
        items_in_orders = set()
        for order in orders:
            items_in_orders.update(order['items'].keys())

        # Leer L y R sin modificarlos
        l_bound, r_bound = map(int, lines.pop(0).strip().split())
        
        print(f"INFO: n={n}, m={m}, k={k}, L={l_bound}, R={r_bound}")
        print(f"INFO: {len(items_in_orders)} items únicos encontrados en pedidos.")
        return n, k, orders, aisles, l_bound, r_bound, items_in_orders
    except (IndexError, ValueError) as e:
        print(f"ERROR: Formato de entrada inválido. {e}")
        sys.exit(1)


def audit_solution(n, k, orders, aisles, l_bound, r_bound, all_item_ids, x_sol, y_sol):
    """
    Verifica si una solución dada es factible para TODAS las restricciones.
    """
    print("\n" + "-"*50)
    print("AUDITORÍA DE LA SOLUCIÓN FINAL")
    print("-" * 50)
    
    # 1. Verificar restricción de rango [L, R]
    total_quantity = sum(orders[i]['total_quantity'] * x_sol[i] for i in range(n))
    print(f"Verificando Rango [L, R]: {l_bound} <= {total_quantity:.2f} <= {r_bound}")
    if not (l_bound <= total_quantity <= r_bound):
        print("  --> ERROR: La restricción de rango [L, R] está VIOLADA.")
        return False
    else:
        print("  --> OK: La restricción de rango [L, R] se cumple.")

    # 2. Verificar TODAS las restricciones de capacidad
    print("\nVerificando todas las restricciones de capacidad (Demanda <= Capacidad):")
    all_constraints_ok = True
    for item_j in all_item_ids:
        demand = sum(orders[i]['items'].get(item_j, 0) * x_sol[i] for i in range(n))
        capacity = sum(aisles[j]['items'].get(item_j, 0) * y_sol[j] for j in range(k))
        
        if demand > capacity + 1e-5: # Usamos una pequeña tolerancia
            print(f"  - Item {item_j:<5}: Demanda={demand:.2f}, Capacidad={capacity:.2f} --> VIOLADA")
            all_constraints_ok = False
    
    if all_constraints_ok:
        print("\n  --> OK: Todas las restricciones de capacidad se cumplen.")
        print("La solución es FACTIBLE para el problema original completo.")
        return True
    else:
        print("\n  --> ERROR: La solución NO ES FACTIBLE para el problema original completo.")
        return False

def get_initial_constraints_greedy(orders, relevant_constraint_items):
    """
    Implementa una heurística greedy para seleccionar los 5 items iniciales
    que cubren la mayor cantidad de pedidos.
    """
    print("INFO: Iniciando heurística greedy para seleccionar las 5 restricciones iniciales...")
    
    remaining_orders = [set(order['items'].keys()) for order in orders]
    initial_constraints = set()
    
    item_counts = Counter()
    for order_items in remaining_orders:
        for item_id in order_items:
            if item_id in relevant_constraint_items:
                item_counts[item_id] += 1

    # Bucle para seleccionar hasta 5 items de forma greedy
    for _ in range(5):
        if not any(remaining_orders) or not item_counts:
            break

        # Encontrar el item más frecuente entre los restantes
        most_common_item, _ = item_counts.most_common(1)[0]
        initial_constraints.add(most_common_item)

        # "Cubrir" los pedidos que contienen este item y actualizar contadores
        orders_to_remove_indices = []
        for i, order_items in enumerate(remaining_orders):
            if most_common_item in order_items:
                orders_to_remove_indices.append(i)
                for item_id in order_items:
                    if item_id in item_counts:
                        item_counts[item_id] -= 1
        
        # Marcar los pedidos cubiertos como sets vacíos
        for i in orders_to_remove_indices:
            remaining_orders[i] = set()

        # Limpiar el contador de items con cuenta cero para eficiencia
        item_counts = Counter({item: count for item, count in item_counts.items() if count > 0})

    print(f"INFO: Conjunto inicial de restricciones greedy (hasta 5) ({len(initial_constraints)}): {initial_constraints}")
    return initial_constraints

def solve_model_once(n, k, orders, aisles, l_bound, r_bound, lambda_val, active_constraints_items):
    """
    Crea, resuelve y devuelve la solución para un subproblema con un lambda y
    un conjunto de restricciones de capacidad fijos.
    """
    mdl = Model(name='subproblem')
    # Silenciar la salida de CPLEX para no saturar la consola
    mdl.context.cplex_parameters.mip.display = 0

    x_vars = mdl.binary_var_list(n, name='x')
    y_vars = mdl.binary_var_list(k, name='y')

    # Expresiones para la función objetivo y restricciones
    total_quantity_expr = mdl.sum(orders[i]['total_quantity'] * x_vars[i] for i in range(n))
    total_aisles_expr = mdl.sum(y_vars[j] for j in range(k))

    # Restricciones estáticas (siempre presentes)
    mdl.add_range(l_bound, total_quantity_expr, r_bound)
    mdl.add_constraint(total_aisles_expr >= 1, ctname="force_aisle_opening")

    # Añadir las restricciones de capacidad activas para esta iteración
    for item_j in active_constraints_items:
        demand = mdl.sum(orders[i]['items'].get(item_j, 0) * x_vars[i] for i in range(n))
        capacity = mdl.sum(aisles[j]['items'].get(item_j, 0) * y_vars[j] for j in range(k))
        mdl.add_constraint(demand <= capacity, ctname=f"cap_item_{item_j}")

    # Función objetivo de Dinkelbach
    mdl.set_objective('max', total_quantity_expr - lambda_val * total_aisles_expr)

    solution = mdl.solve()

    if not solution:
        return None, None, None

    x_sol = {i: x_vars[i].solution_value for i in range(n)}
    y_sol = {j: y_vars[j].solution_value for j in range(k)}
    
    return x_sol, y_sol, solution.get_objective_value()

# --- Algoritmo Principal ---
def main():
    lines = sys.stdin.readlines()
    if not lines:
        print("ERROR: No se proporcionaron datos de entrada.")
        return

    n, k, orders, aisles, l_bound, r_bound, items_in_orders = parse_input(lines)

    # --- Pre-procesamiento Lógico ---
    # Determinar qué items necesitan restricciones de capacidad
    available_items_in_aisles = set()
    for aisle in aisles:
        available_items_in_aisles.update(aisle['items'].keys())

    relevant_constraint_items = items_in_orders.intersection(available_items_in_aisles)
    if not relevant_constraint_items:
        print("ERROR: No hay items en común entre pedidos y pasillos.")
        return

    # --- Bucle Externo: Algoritmo de Dinkelbach (Estructura Original) ---
    lambda_val = 0.0
    epsilon = 1e-4
    max_dinkelbach_iter = 50

    # Empezamos con el conjunto de restricciones de la heurística greedy.
    # Este conjunto se irá expandiendo en el bucle interno.
    active_constraints_items = get_initial_constraints_greedy(orders, relevant_constraint_items)
    if not active_constraints_items:
        active_constraints_items = {random.choice(list(relevant_constraint_items))}

    print("\n" + "="*50)
    print("INICIANDO ALGORITMO (Dinkelbach Externo)")
    print("="*50)

    # Variables para guardar la mejor solución factible encontrada
    best_ratio = -1
    best_x_sol, best_y_sol = None, None

    for d_iter in range(max_dinkelbach_iter):
        print(f"\n--- Dinkelbach Iteración {d_iter+1}, Lambda = {lambda_val:.4f} ---")

        # --- Bucle Interno: Generación de Restricciones ---
        # En cada iteración de Dinkelbach, resolvemos el subproblema hasta que sea factible.
        max_constraint_gen_iter = len(relevant_constraint_items) + 5
        
        # Guardamos la última solución del bucle interno para esta iteración de Dinkelbach
        x_sol, y_sol = None, None

        for cg_iter in range(max_constraint_gen_iter):
            print(f"  Constraint-Gen Iter {cg_iter+1}, Restricciones activas: {len(active_constraints_items)}")
            
            # Construir y resolver el modelo con el lambda y las restricciones actuales
            current_x, current_y, sub_obj_val = solve_model_once(n, k, orders, aisles, l_bound, r_bound, lambda_val, active_constraints_items)
            
            if current_x is None:
                print("  ERROR: El problema se volvió infactible.")
                break 
            
            x_sol, y_sol = current_x, current_y

            # --- Lógica de Generación de Restricciones ---
            violations = []
            candidate_items = relevant_constraint_items - active_constraints_items
            for item_j in candidate_items:
                demand_val = sum(orders[i]['items'].get(item_j, 0) * x_sol[i] for i in range(n))
                capacity_val = sum(aisles[j]['items'].get(item_j, 0) * y_sol[j] for j in range(k))
                violation = demand_val - capacity_val
                if violation > epsilon:
                    violations.append((violation, item_j))

            if not violations:
                print("  INFO: No hay más restricciones violadas para este lambda. Subproblema resuelto.")
                break
            
            # --- HEURÍSTICA PORCENTUAL: Greedy-Cover sobre el 50% peor ---
            num_violations = len(violations)
            num_to_consider = math.ceil(num_violations * 0.5)
            print(f"  INFO: {num_violations} violaciones. Analizando el 50% peor ({num_to_consider} items).")
            
            # 1. Seleccionar el 50% de violaciones más grandes
            violations.sort(reverse=True, key=lambda x: x[0])
            top_violations = violations[:int(num_to_consider)]
            top_violated_items = {item for _, item in top_violations}

            # 2. Identificar pedidos seleccionados en la solución actual
            selected_order_indices = {i for i, val in x_sol.items() if val > 0.5}

            # 3. Filtrar para obtener "pedidos problemáticos"
            problematic_orders_to_cover = []
            for i in selected_order_indices:
                items_to_cover_in_order = set(orders[i]['items'].keys()).intersection(top_violated_items)
                if items_to_cover_in_order:
                    problematic_orders_to_cover.append(items_to_cover_in_order)
            
            items_added_this_iter = set()

            if not problematic_orders_to_cover:
                # Fallback: añadir el 5% de las violaciones más grandes directamente.
                num_to_add = max(1, math.ceil(num_violations * 0.05))
                print(f"  INFO: Ningún pedido seleccionado contiene los items más violados. Añadiendo los {int(num_to_add)} con mayor violación directa.")
                for _, item_to_add in violations[:int(num_to_add)]:
                    items_added_this_iter.add(item_to_add)
            else:
                # 4. Heurística Greedy Set-Cover para elegir 5% de los items candidatos
                num_to_add = max(1, math.ceil(len(top_violated_items) * 0.05))
                print(f"  INFO: Realizando cobertura sobre {len(problematic_orders_to_cover)} pedidos para elegir hasta {int(num_to_add)} items.")
                
                candidate_items_for_cover = set(top_violated_items)

                for _ in range(int(num_to_add)):
                    if not problematic_orders_to_cover:
                        break

                    item_counts = Counter()
                    for item_set in problematic_orders_to_cover:
                        valid_items_in_set = item_set.intersection(candidate_items_for_cover)
                        item_counts.update(valid_items_in_set)
                    
                    if not item_counts:
                        break

                    most_common_item, _ = item_counts.most_common(1)[0]
                    items_added_this_iter.add(most_common_item)
                    
                    if most_common_item in candidate_items_for_cover:
                        candidate_items_for_cover.remove(most_common_item)
                        
                    problematic_orders_to_cover = [iset for iset in problematic_orders_to_cover if most_common_item not in iset]
            
            print(f"  INFO: Heurística seleccionó {len(items_added_this_iter)} restricciones para añadir: {items_added_this_iter}")
            active_constraints_items.update(items_added_this_iter)
        
        # Si el subproblema fue infactible, no podemos continuar con este lambda
        if x_sol is None:
            print("ADVERTENCIA: No se pudo encontrar una solución para el lambda actual. El algoritmo termina.")
            break
            
        # --- Chequeo de Convergencia de Dinkelbach ---
        numerator = sum(orders[i]['total_quantity'] * x_sol[i] for i in range(n))
        denominator = sum(y_sol[j] for j in range(k))
        
        final_obj_val = numerator - lambda_val * denominator
        print(f"Valor final de Z(lambda) para esta iteración = {final_obj_val:.4f}")

        # Guardar la solución actual como la mejor hasta ahora
        current_ratio = 0
        if denominator > 1e-6:
            current_ratio = numerator / denominator

        if current_ratio > best_ratio:
            best_ratio = current_ratio
            best_x_sol, best_y_sol = x_sol, y_sol
            print(f"INFO: Nueva mejor solución encontrada con ratio: {best_ratio:.4f} (Num: {numerator:.2f}, Sum(y): {denominator:.0f})")

        if abs(final_obj_val) < epsilon:
            print("\n" + "="*50)
            print(f"CONVERGENCIA GLOBAL ALCANZADA (Ratio Final: {best_ratio:.4f})")
            
            # La solución final es la mejor que hemos encontrado
            is_feasible = audit_solution(n, k, orders, aisles, l_bound, r_bound, relevant_constraint_items, best_x_sol, best_y_sol)
            
            if is_feasible:
                print("\n--- RESULTADOS FINALES ---")
                selected_orders = [i for i, val in best_x_sol.items() if val > 0.5]
                selected_aisles = [j for j, val in best_y_sol.items() if val > 0.5]
                print(f"Pedidos seleccionados ({len(selected_orders)}): {selected_orders}")
                print(f"Pasillos seleccionados (Sum y_k = {len(selected_aisles)}): {selected_aisles}")
                final_numerator = sum(orders[i]['total_quantity'] for i in selected_orders)
                final_denominator = len(selected_aisles)
                print(f"Valor final del Ratio (Cantidad / Pasillos): {final_numerator} / {final_denominator} = {best_ratio:.4f}")

            return

        # Actualizar lambda para la siguiente iteración
        if denominator < 1e-6:
             print("ADVERTENCIA: Denominador es cero. El algoritmo no puede continuar.")
             break
        
        lambda_val = numerator / denominator
        print(f"INFO: Actualizando lambda a {lambda_val:.4f} para la siguiente iteración.")

    print("\nADVERTENCIA: Se alcanzó el número máximo de iteraciones de Dinkelbach.")
    # Si no converge, imprimir la mejor solución encontrada hasta ahora
    if best_x_sol:
        print("\n" + "="*50)
        print("MEJOR SOLUCIÓN ENCONTRADA (NO HUBO CONVERGENCIA FORMAL)")
        print(f"Ratio Final Aproximado: {best_ratio:.4f}")
        print("="*50)
        is_feasible = audit_solution(n, k, orders, aisles, l_bound, r_bound, relevant_constraint_items, best_x_sol, best_y_sol)
        
        if is_feasible:
            print("\n--- RESULTADOS DE LA MEJOR SOLUCIÓN ---")
            selected_orders = [i for i, val in best_x_sol.items() if val > 0.5]
            selected_aisles = [j for j, val in best_y_sol.items() if val > 0.5]
            print(f"Pedidos seleccionados ({len(selected_orders)}): {selected_orders}")
            print(f"Pasillos seleccionados (Sum y_k = {len(selected_aisles)}): {selected_aisles}")
            final_numerator = sum(orders[i]['total_quantity'] for i in selected_orders)
            final_denominator = len(selected_aisles)
            print(f"Valor del Ratio (Cantidad / Pasillos): {final_numerator} / {final_denominator} = {best_ratio:.4f}")

if __name__ == "__main__":
    main() 