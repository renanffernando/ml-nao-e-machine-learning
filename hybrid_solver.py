import sys
import time
import math
from docplex.mp.model import Model
from collections import Counter

def parse_input(lines):
    n, m, k = map(int, lines[0].split())
    
    orders_data = [list(map(int, line.split())) for line in lines[1:n+1]]
    orders = [{'id': i, 'total_quantity': p[0], 'items': Counter(p[2:])} for i, p in enumerate(orders_data)]
    
    aisles_data = [list(map(int, line.split())) for line in lines[n+1:n+k+1]]
    aisles = [{'id': j, 'items': Counter(p[2:])} for j, p in enumerate(aisles_data)]

    l, r = map(int, lines[n+k+1].split())
    
    return n, k, orders, aisles, l, r

def audit_solution(all_items, orders, aisles, l_bound, r_bound, x_sol, y_sol):
    print("\n--- AUDITORÍA DE LA SOLUCIÓN ---")
    selected_orders = [i for i, v in x_sol.items() if v > 0.5]
    if not (l_bound <= len(selected_orders) <= r_bound):
        print(f"FALLO: Pedidos seleccionados {len(selected_orders)} fuera de [{l_bound}, {r_bound}]")
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

def solve_relaxed_problem(n, k, orders, aisles, l_bound, r_bound, lambda_val, hard_constraints, coeffs_x, coeffs_y):
    mdl = Model(name='RelaxedProblem')
    mdl.set_time_limit(20)
    mdl.context.cplex_parameters.threads = 1
    x = mdl.binary_var_list(n, name='x')
    y = mdl.binary_var_list(k, name='y')

    numerator = mdl.sum(orders[i]['total_quantity'] * x[i] for i in range(n))
    denominator = mdl.sum(y[j] for j in range(k))
    penalty_x = mdl.sum(coeffs_x[i] * x[i] for i in range(n))
    penalty_y = mdl.sum(coeffs_y[j] * y[j] for j in range(k))
    mdl.maximize(numerator - lambda_val * denominator - penalty_x + penalty_y)

    mdl.add_constraint(mdl.sum(x) >= l_bound)
    mdl.add_constraint(mdl.sum(x) <= r_bound)

    for item_j in hard_constraints:
        demand = mdl.sum(orders[i]['items'].get(item_j, 0) * x[i] for i in range(n))
        capacity = mdl.sum(aisles[j]['items'].get(item_j, 0) * y[j] for j in range(k))
        mdl.add_constraint(demand <= capacity)

    solution = mdl.solve()
    if not solution: return None, None
    x_sol = {i: x[i].solution_value for i in range(n)}
    y_sol = {j: y[j].solution_value for j in range(k)}
    return x_sol, y_sol

def main(input_data):
    start_time = time.time()
    n, k, orders, aisles, l_bound, r_bound = parse_input(input_data)
    
    all_items = set().union(*(o['items'].keys() for o in orders), *(a['items'].keys() for a in aisles))
    
    lambda_val, epsilon, max_dinkelbach_iter = 0.0, 1e-4, 30
    best_ratio, best_x_sol, best_y_sol = -1, None, None
    max_promo_iter, max_subgradient_iter = 10, 15

    print("="*50 + f"\nINICIANDO SOLVER HÍBRIDO (n={n}, k={k}, items={len(all_items)})\n" + "="*50)

    for d_iter in range(max_dinkelbach_iter):
        print(f"\n--- Dinkelbach Iter {d_iter+1}, Lambda = {lambda_val:.4f} ---")
        hard_constraints = set()
        relaxed_items = all_items.copy()
        current_x, current_y = None, None
        
        for promo_iter in range(max_promo_iter):
            print(f"  Promo Iter {promo_iter+1}, Duras: {len(hard_constraints)}")
            
            # Para recordar la solución más factible encontrada en esta ronda de subgradiente
            best_sol_in_promo = {'x': None, 'y': None, 'infeasibility': float('inf')}

            u = {item: 0.0 for item in relaxed_items}
            g_prev = {item: 0.0 for item in relaxed_items}
            d_prev = {item: 0.0 for item in relaxed_items}
            
            for sg_iter in range(max_subgradient_iter):
                coeffs_x = {i: sum(u.get(item, 0) * orders[i]['items'].get(item, 0) for item in relaxed_items) for i in range(n)}
                coeffs_y = {j: sum(u.get(item, 0) * aisles[j]['items'].get(item, 0) for item in relaxed_items) for j in range(k)}
                
                x_sol, y_sol = solve_relaxed_problem(n, k, orders, aisles, l_bound, r_bound, lambda_val, hard_constraints, coeffs_x, coeffs_y)
                if x_sol is None: continue # Saltar si el solver falla
                
                # --- Guardar la solución más factible de esta ronda ---
                # Calcular la infactibilidad total de la solución actual
                current_infeasibility = 0
                for item in all_items:
                    demand = sum(o['items'].get(item, 0) for i, o in enumerate(orders) if x_sol.get(i,0) > 0.5)
                    capacity = sum(a['items'].get(item, 0) for i, a in enumerate(aisles) if y_sol.get(i,0) > 0.5)
                    current_infeasibility += max(0, demand - capacity)
                
                # Si es la mejor que hemos visto, la guardamos
                if current_infeasibility < best_sol_in_promo['infeasibility']:
                    best_sol_in_promo['x'] = x_sol
                    best_sol_in_promo['y'] = y_sol
                    best_sol_in_promo['infeasibility'] = current_infeasibility

                subgrads = {item: sum(orders[i]['items'].get(item,0)*x_sol[i] for i in range(n)) - sum(aisles[j]['items'].get(item,0)*y_sol[j] for j in range(k)) for item in relaxed_items}
                
                g_curr = subgrads
                if sg_iter > 0:
                    g_prev_dot = sum(g_prev[item]**2 for item in relaxed_items)
                    beta = max(0, sum(g_curr[item]*(g_curr[item]-g_prev[item]) for item in relaxed_items) / (g_prev_dot+1e-9))
                    d_curr = {item: g_curr[item] + beta * d_prev[item] for item in relaxed_items}
                else:
                    d_curr = g_curr
                
                step = 1.0 / (sg_iter + 1)
                u = {item: max(0, u[item] + step * d_curr[item]) for item in relaxed_items}
                g_prev, d_prev = g_curr, d_curr
                
                # --- VISIBILIDAD DE MULTIPLICADORES ---
                if sg_iter % 3 == 0 and u: # Imprimir cada 3 iteraciones
                    max_u = max(u.values())
                    avg_u = sum(u.values()) / len(u)
                    top_5_u = sorted(u.items(), key=lambda item: item[1], reverse=True)[:5]
                    top_5_str = ", ".join([f"i{item_id}:{val:.2f}" for item_id, val in top_5_u])
                    print(f"    sg_iter {sg_iter+1}: Max u={max_u:.3f}, Avg u={avg_u:.3f} | Top: {top_5_str}")

            # --- CHEQUEO DE FACTIBILIDAD USANDO LA MEJOR SOLUCIÓN ENCONTRADA ---
            current_x = best_sol_in_promo['x']
            current_y = best_sol_in_promo['y']
            
            if current_x is None: 
                print("    ADVERTENCIA: No se encontró ninguna solución en la ronda de subgradiente.")
                break # Salir del bucle de promoción
            
            # --- ÚNICO CHEQUEO DE FACTIBILIDAD (Explícito y Corregido) ---
            violations = []
            for item_id in all_items:
                demand = sum(order_obj['items'].get(item_id, 0) for order_idx, order_obj in enumerate(orders) if current_x.get(order_idx, 0.0) > 0.5)
                capacity = sum(aisle_obj['items'].get(item_id, 0) for aisle_idx, aisle_obj in enumerate(aisles) if current_y.get(aisle_idx, 0.0) > 0.5)
                if demand > capacity + 1e-5:
                    violations.append({'item': item_id, 'viol': demand - capacity})

            if not violations:
                print("  INFO: Solución factible encontrada para este lambda.")
                break
            
            violations.sort(key=lambda x: x['viol'], reverse=True)
            new_hard = {v['item'] for v in violations[:3]}
            print(f"  INFO: Promoviendo {len(new_hard)} restricciones: {new_hard}")
            hard_constraints.update(new_hard)
            relaxed_items.difference_update(new_hard)
            if not relaxed_items: break
        
        if current_x is None: print("ADVERTENCIA: No se encontró solución factible. Terminando."); break
        
        numerator = sum(orders[i]['total_quantity'] * current_x[i] for i in range(n))
        denominator = sum(current_y[j] for j in range(k))
        if denominator < 1e-6: print("ADVERTENCIA: Denominador cero."); break
        
        final_obj_val = numerator - lambda_val * denominator
        print(f"Z(lambda) = {final_obj_val:.4f}")
        current_ratio = numerator / denominator
        
        # --- CORRECCIÓN: SOLO ACTUALIZAR SI LA SOLUCIÓN ES "RENTABLE" ---
        if current_ratio > best_ratio and final_obj_val >= 0:
            best_ratio, best_x_sol, best_y_sol = current_ratio, current_x, current_y
            print(f"INFO: Nueva mejor ratio (rentable) encontrada: {best_ratio:.4f}")

        if abs(final_obj_val) < epsilon: print("\nCONVERGENCIA GLOBAL ALCANZADA"); break
        lambda_val = current_ratio

    print("\n" + "="*50 + f"\nALGORITMO TERMINADO EN {time.time()-start_time:.2f}s (Mejor Ratio: {best_ratio:.4f})\n" + "="*50)
    if best_x_sol:
        audit_solution(all_items, orders, aisles, l_bound, r_bound, best_x_sol, best_y_sol)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as f: input_data = f.readlines()
    else:
        input_data = sys.stdin.readlines()
    if input_data: main(input_data)
    else: print("ERROR: No se proporcionaron datos.") 