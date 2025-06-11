import sys
import time
import math
import random
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

        orders = []
        for i in range(n):
            parts = list(map(int, lines.pop(0).strip().split()))
            order_items = {parts[j]: parts[j+1] for j in range(1, len(parts), 2)}
            total_quantity = sum(order_items.values())
            orders.append({'id': i, 'items': order_items, 'total_quantity': total_quantity})

        aisles = []
        for j in range(k):
            parts = list(map(int, lines.pop(0).strip().split()))
            aisle_items = {parts[i]: parts[i+1] for i in range(1, len(parts), 2)}
            aisles.append({'id': j, 'items': aisle_items})

        items_in_orders = set()
        for order in orders:
            items_in_orders.update(order['items'].keys())

        l_bound, r_bound = map(int, lines.pop(0).strip().split())
        
        print(f"INFO: n={n}, m={m}, k={k}, L={l_bound}, R={r_bound}")
        print(f"INFO: {len(items_in_orders)} items únicos encontrados en pedidos.")
        return n, k, orders, aisles, l_bound, r_bound, items_in_orders
    except (IndexError, ValueError) as e:
        print(f"ERROR: Formato de entrada inválido. {e}")
        sys.exit(1)

def audit_solution(all_items, orders, aisles, l_bound, r_bound, x_sol, y_sol):
    print("\n--- AUDITORÍA DE LA SOLUCIÓN ---")
    selected_orders = [i for i, v in x_sol.items() if v > 0.5]
    if not (l_bound <= sum(orders[i]['total_quantity'] for i in selected_orders) <= r_bound):
        print(f"FALLO: Cantidad total fuera de [{l_bound}, {r_bound}]")
        return False
    violations = 0
    for item in all_items:
        demand = sum(o['items'].get(item, 0) for i, o in enumerate(orders) if i in selected_orders)
        capacity = sum(a['items'].get(item, 0) for i, a in enumerate(aisles) if y_sol.get(i,0) > 0.5)
        if demand > capacity + 1e-5:
            print(f"FALLO: Item {item}, Demanda {demand} > Capacidad {capacity}")
            violations += 1
    if violations == 0:
        print("ÉXITO: Solución factible.")
        return True
    return False

def get_initial_constraints_greedy(orders, all_items):
    """
    Implementa una heurística greedy para seleccionar las 5 items iniciales
    que cubren la mayor cantidad de pedidos.
    """
    print("INFO: Iniciando heurística greedy para seleccionar las 5 restricciones iniciales...")
    
    remaining_orders = [set(order['items'].keys()) for order in orders]
    initial_constraints = set()
    
    item_counts = Counter()
    for order_items in remaining_orders:
        for item_id in order_items:
            if item_id in all_items:
                item_counts[item_id] += 1

    for _ in range(5):
        if not any(remaining_orders) or not item_counts:
            break

        most_common_item, _ = item_counts.most_common(1)[0]
        initial_constraints.add(most_common_item)

        orders_to_remove_indices = []
        for i, order_items in enumerate(remaining_orders):
            if most_common_item in order_items:
                orders_to_remove_indices.append(i)
                for item_id in order_items:
                    if item_id in item_counts:
                        item_counts[item_id] -= 1
        
        for i in orders_to_remove_indices:
            remaining_orders[i] = set()

        item_counts = Counter({item: count for item, count in item_counts.items() if count > 0})

    print(f"INFO: Conjunto inicial de restricciones greedy (hasta 5) ({len(initial_constraints)}): {initial_constraints}")
    return initial_constraints

def main(input_data):
    start_time = time.time()
    n, k, orders, aisles, l_bound, r_bound, all_items = parse_input(input_data)
    
    mdl = Model(name='FractionalSolver')
    mdl.set_time_limit(16000)
    x = mdl.binary_var_list(n, name='x')
    y = mdl.binary_var_list(k, name='y')

    numerator = mdl.sum(orders[i]['total_quantity'] * x[i] for i in range(n))
    denominator = mdl.sum(y[j] for j in range(k))
    mdl.add_constraint(denominator >= 1, ctname="min_aisles")
    mdl.add_range(l_bound, numerator, r_bound)

    if k > 0:
        lambda_low, lambda_high = l_bound / k, r_bound
    else:
        lambda_low, lambda_high = 0.0, 0.0

    best_feasible_lambda = 0.0
    
    active_constraints = get_initial_constraints_greedy(orders, all_items)
    if not active_constraints and all_items:
        active_constraints = {random.choice(list(all_items))}

    for item in active_constraints:
        demand = mdl.sum(orders[i]['items'].get(item, 0) * x[i] for i in range(n))
        capacity = mdl.sum(aisles[j]['items'].get(item, 0) * y[j] for j in range(k))
        mdl.add_constraint(demand <= capacity, ctname=f"cap_{item}")

    print("="*50 + "\nINICIANDO SOLVER (Búsqueda Binaria + CG Persistente)\n" + "="*50)
    
    num_constraints_to_add = max(1, math.floor(n * 0.05))
    print(f"INFO: Se añadirán hasta {num_constraints_to_add} restricciones por iteración de CG.")

    for bs_iter in range(50):
        lambda_mid = (lambda_low + lambda_high) / 2
        if lambda_high - lambda_low < 1e-4: break
        
        if mdl.get_constraint_by_name("lambda_ct") is not None:
            mdl.remove_constraint("lambda_ct")
        mdl.add_constraint(numerator - lambda_mid * denominator >= 0, ctname="lambda_ct")
        
        print(f"\n--- BS Iter {bs_iter+1}, Probando Lambda = {lambda_mid:.4f} ---")
        print(f"  Rango actual: [low: {lambda_low:.4f}, high: {lambda_high:.4f}]")
        
        is_feasible_for_lambda = False

        max_cg_iter = (len(all_items) // num_constraints_to_add) + 5
        for cg_iter in range(max_cg_iter):
            print(f"  CG Iter {cg_iter+1}, Restricciones: {len(active_constraints)}")
            
            solution = mdl.solve()
            
            if not solution:
                lambda_high = lambda_mid
                break

            x_sol = {i: x[i].solution_value for i in range(n)}
            y_sol = {j: y[j].solution_value for j in range(k)}
            
            violations = []
            candidate_items = all_items - active_constraints
            for item in candidate_items:
                demand = sum(o['items'].get(item, 0) for i, o in enumerate(orders) if x_sol.get(i,0)>0.5)
                capacity = sum(a['items'].get(item, 0) for j, a in enumerate(aisles) if y_sol.get(j,0)>0.5)
                if demand > capacity + 1e-5:
                    violations.append({'item': item, 'viol': demand - capacity})

            if not violations:
                is_feasible_for_lambda = True
                break

            violations.sort(key=lambda v: v['viol'], reverse=True)
            newly_violated_items = {v['item'] for v in violations[:num_constraints_to_add]}
            
            for item in newly_violated_items:
                demand = mdl.sum(orders[i]['items'].get(item, 0) * x[i] for i in range(n))
                capacity = mdl.sum(aisles[j]['items'].get(item, 0) * y[j] for j in range(k))
                mdl.add_constraint(demand <= capacity, ctname=f"cap_{item}")
            active_constraints.update(newly_violated_items)
        
        if is_feasible_for_lambda:
            best_feasible_lambda = lambda_mid
            lambda_low = lambda_mid
        else:
            lambda_high = lambda_mid

    final_lambda = best_feasible_lambda
    print(f"\nBúsqueda terminada. Mejor Lambda factible: {final_lambda:.4f}")
    print("Ejecutando una pasada final de CG con el lambda óptimo para garantizar la factibilidad...")

    if mdl.get_constraint_by_name("lambda_ct") is not None:
        mdl.remove_constraint("lambda_ct")
    mdl.add_constraint(numerator - final_lambda * denominator >= 0, ctname="lambda_ct")

    final_sol = None
    final_feasible = False
    
    max_cg_iter_final = (len(all_items) // num_constraints_to_add) + 5
    for cg_iter in range(max_cg_iter_final):
        print(f"  CG Final Iter {cg_iter+1}, Restricciones: {len(active_constraints)}")
        
        solution = mdl.solve()
        if not solution:
            print("ERROR CRÍTICO: El modelo se volvió infactible en la pasada final.")
            break

        x_sol = {i: x[i].solution_value for i in range(n)}
        y_sol = {j: y[j].solution_value for j in range(k)}

        violations = []
        candidate_items = all_items - active_constraints
        for item in candidate_items:
            demand = sum(o['items'].get(item, 0) for i, o in enumerate(orders) if x_sol.get(i,0)>0.5)
            capacity = sum(a['items'].get(item, 0) for j, a in enumerate(aisles) if y_sol.get(j,0)>0.5)
            if demand > capacity + 1e-5:
                violations.append({'item': item, 'viol': demand - capacity})

        if not violations:
            print("Solución final encontrada y es factible para todas las restricciones.")
            final_sol = (x_sol, y_sol)
            final_feasible = True
            break

        violations.sort(key=lambda v: v['viol'], reverse=True)
        newly_violated_items = {v['item'] for v in violations[:num_constraints_to_add]}
        
        print(f"    Añadiendo {len(newly_violated_items)} nuevas restricciones: {newly_violated_items}")
        for item in newly_violated_items:
            demand = mdl.sum(orders[i]['items'].get(item, 0) * x[i] for i in range(n))
            capacity = mdl.sum(aisles[j]['items'].get(item, 0) * y[j] for j in range(k))
            mdl.add_constraint(demand <= capacity, ctname=f"cap_{item}")
        active_constraints.update(newly_violated_items)


    print(f"\nBúsqueda terminada en {time.time()-start_time:.2f}s.")
    if final_feasible:
        print("Auditoría de la mejor solución encontrada...")
        audit_solution(all_items, orders, aisles, l_bound, r_bound, final_sol[0], final_sol[1])
    else:
        print("ERROR: No se pudo generar una solución final completamente factible.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as f:
            input_data = f.readlines()
    else:
        input_data = sys.stdin.readlines()
    if input_data:
        main(input_data)
    else:
        print("ERROR: No se proporcionaron datos.") 