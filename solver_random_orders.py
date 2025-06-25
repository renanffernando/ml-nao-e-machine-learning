#!/usr/bin/env python3

import json
import sys
import time
import random
from docplex.mp.model import Model

def parse_input(lines):
    """Parse the input data from lines."""
    n, m, k = map(int, lines[0].split())
    
    orders = []
    for i in range(1, n + 1):
        line = lines[i].strip()
        if not line:
            continue
        parts = line.split()
        order_id = int(parts[0])
        items = {}
        total_quantity = 0
        for j in range(1, len(parts), 2):
            if j + 1 < len(parts):
                item_id = int(parts[j])
                quantity = int(parts[j + 1])
                items[item_id] = quantity
                total_quantity += quantity
        orders.append({'id': order_id, 'items': items, 'total_quantity': total_quantity})
    
    aisles = []
    for i in range(n + 1, n + 1 + k):
        line = lines[i].strip()
        if not line:
            continue
        parts = line.split()
        aisle_id = int(parts[0])
        items = {}
        for j in range(1, len(parts), 2):
            if j + 1 < len(parts):
                item_id = int(parts[j])
                capacity = int(parts[j + 1])
                items[item_id] = capacity
        aisles.append({'id': aisle_id, 'items': items})
    
    l, r = map(int, lines[n + 1 + k].split())
    
    return n, m, k, orders, aisles, l, r

def solve_dinkelbach_orders(orders, aisles, l_bound, r_bound, selected_orders, max_lambda=0.0, show_iterations=False):
    """
    Solver using Dinkelbach algorithm for selected orders.
    """
    if not selected_orders:
        return None, None, 0
    
    if show_iterations:
        print(f"  Solving with {len(selected_orders)} selected orders using Dinkelbach")
    
    n_selected = len(selected_orders)
    k_aisles = len(aisles)
    
    # Obtener todos los items relevantes
    all_items = set()
    for order_idx in selected_orders:
        order = orders[order_idx]
        all_items.update(order['items'].keys())
    for aisle in aisles:
        all_items.update(aisle['items'].keys())
    
    max_iter = 100
    epsilon = 1e-6
    
    for iteration in range(max_iter):
        if show_iterations:
            print(f"    Iter {iteration}: λ_max_usado={max_lambda:.6f}", end="")
        
        # Crear modelo para esta iteración de Dinkelbach
        mdl = Model(name=f'dinkelbach_iter_{iteration}')
        mdl.context.cplex_parameters.mip.display = 0  # Silenciar CPLEX
        mdl.set_time_limit(30)
        
        # Variables de decisión
        x_vars = mdl.binary_var_list(n_selected, name='x')
        y_vars = mdl.binary_var_list(k_aisles, name='y')
        
        # Expresiones para objetivo
        total_quantity_expr = mdl.sum(orders[selected_orders[i]]['total_quantity'] * x_vars[i] for i in range(n_selected))
        total_aisles_expr = mdl.sum(y_vars[j] for j in range(k_aisles))
        
        # Restricción de rango [L, R] - usar cantidad total, no número de pedidos
        mdl.add_constraint(total_quantity_expr >= l_bound)
        mdl.add_constraint(total_quantity_expr <= r_bound)
        
        # Forzar que al menos un pasillo esté abierto
        mdl.add_constraint(total_aisles_expr >= 1)
        
        # Restricciones de capacidad
        for item_id in all_items:
            demand_expr = mdl.sum(
                orders[selected_orders[i]]['items'].get(item_id, 0) * x_vars[i] 
                for i in range(n_selected)
            )
            capacity_expr = mdl.sum(
                aisles[j]['items'].get(item_id, 0) * y_vars[j] 
                for j in range(k_aisles)
            )
            mdl.add_constraint(demand_expr <= capacity_expr)
        
        # Función objetivo de Dinkelbach: max(f(x,y) - lambda * g(x,y))
        mdl.set_objective('max', total_quantity_expr - max_lambda * total_aisles_expr)
        
        # Resolver
        solution = mdl.solve()
        
        if not solution:
            if show_iterations:
                print(f" → No factible")
            return None, None, 0
        
        # Obtener valores de las variables
        x_sol = [int(solution.get_value(x_vars[i])) for i in range(n_selected)]
        y_sol = [int(solution.get_value(y_vars[j])) for j in range(k_aisles)]
        
        # Calcular numerador y denominador
        numerator = sum(orders[selected_orders[i]]['total_quantity'] * x_sol[i] for i in range(n_selected))
        denominator = sum(y_sol[j] for j in range(k_aisles))
        
        # Calcular el nuevo lambda
        if denominator > 0:
            lambda_val = numerator / denominator
        else:
            lambda_val = 0.0
        
        if show_iterations:
            print(f" → lambda_calculado={lambda_val:.6f}, cantidad={numerator}, pasillos={denominator}, ratio={lambda_val:.6f}")
        
        # VERIFICACIÓN DETALLADA DE LA SOLUCIÓN
        if show_iterations and lambda_val > 12.0:  # Solo para ratios sospechosamente altos
            print(f"    *** VERIFICACIÓN DE SOLUCIÓN SOSPECHOSA (ratio={lambda_val:.6f}) ***")
            selected_order_indices = [i for i in range(n_selected) if x_sol[i] == 1]
            selected_aisle_indices = [j for j in range(k_aisles) if y_sol[j] == 1]
            
            print(f"    Pedidos seleccionados: {len(selected_order_indices)}")
            print(f"    Pasillos seleccionados: {len(selected_aisle_indices)}")
            
            # Verificar capacidades por item
            total_demand = {}
            total_capacity = {}
            
            for i in selected_order_indices:
                order = orders[selected_orders[i]]
                for item_id, quantity in order['items'].items():
                    total_demand[item_id] = total_demand.get(item_id, 0) + quantity
            
            for j in selected_aisle_indices:
                aisle = aisles[j]
                for item_id, capacity in aisle['items'].items():
                    total_capacity[item_id] = total_capacity.get(item_id, 0) + capacity
            
            print(f"    Demanda total por item: {total_demand}")
            print(f"    Capacidad total por item: {total_capacity}")
            
            # Verificar si hay violaciones
            violations = []
            for item_id, demand in total_demand.items():
                capacity = total_capacity.get(item_id, 0)
                if demand > capacity:
                    violations.append(f"Item {item_id}: demanda={demand} > capacidad={capacity}")
            
            if violations:
                print(f"    *** VIOLACIONES DETECTADAS: {violations} ***")
            else:
                print(f"    *** VERIFICACIÓN PASADA ***")
        
        # CONDICIÓN DE TERMINACIÓN TEMPRANA: si el nuevo lambda <= max_lambda usado
        if lambda_val <= max_lambda + epsilon:
            if show_iterations:
                print(f"    *** TERMINANDO: lambda_calculado ({lambda_val:.6f}) <= lambda_max_usado ({max_lambda:.6f}) ***")
            return x_sol, y_sol, lambda_val
        
        # Actualizar max_lambda para la siguiente iteración
        if lambda_val > max_lambda:
            max_lambda = lambda_val
        
        # Verificar convergencia estándar
        if abs(lambda_val - max_lambda) < epsilon:
            if show_iterations:
                print(f"    *** CONVERGENCIA ***")
            return x_sol, y_sol, lambda_val
    
    # Si llegamos aquí, no convergió
    if show_iterations:
        print(f"    *** NO CONVERGIÓ ***")
    return x_sol, y_sol, lambda_val if 'lambda_val' in locals() else 0.0

def solve_with_progressive_search(orders, aisles, l_bound, r_bound, time_limit_minutes=10):
    """
    Solve using progressive search strategy, exploring random orders with ALL aisles available.
    """
    n = len(orders)
    k = len(aisles)
    best_ratio = 0
    best_solution = None
    best_orders = None
    best_aisles = None
    
    # Configuración inicial - empezar con 100 pedidos y trials
    initial_orders = max(50, n // 4)  # Empezar con 100 pedidos
    current_trials = 4
    
    time_limit_seconds = time_limit_minutes * 60
    start_time = time.time()
    
    # MAX_LAMBDA GLOBAL para Dinkelbach
    max_lambda = 0.0
    phase = 1
    
    print(f"=== BÚSQUEDA PROGRESIVA DE PEDIDOS ALEATORIOS ===")
    print(f"Total pedidos disponibles: {n}")
    print(f"Total pasillos disponibles: {k} (TODOS disponibles)")
    print(f"Empezando con: {initial_orders} pedidos por trial (máximo 500)")
    print(f"Rango: [{l_bound}, {r_bound}]")
    print(f"Límite de tiempo: {time_limit_minutes} minutos")
    print()
    
    # Probar diferentes números de pedidos - incrementos de 50 en 50, máximo 500
    order_counts = []
    current_orders = initial_orders
    max_orders = min(2000, n)  # No superar 500 pedidos
    while current_orders <= max_orders:
        order_counts.append(current_orders)
        if current_orders >= max_orders:
            break
        # Incremento de 50 en 50
        current_orders = min(current_orders + n // 10, max_orders)
    
    for order_count in order_counts:
        if time.time() - start_time >= time_limit_seconds:
            print(f"Límite de tiempo alcanzado")
            break
            
        print(f"--- FASE {phase}: {order_count} pedidos, {current_trials} trials ---")
        print(f"Tiempo transcurrido: {time.time() - start_time:.1f}s / {time_limit_seconds}s")
        print("-" * 50)
        
        phase_start_time = time.time()
        phase_best_ratio = 0
        
        for trial in range(current_trials):
            if time.time() - start_time >= time_limit_seconds:
                print(f"Límite de tiempo alcanzado")
                break
            
            # Seleccionar PEDIDOS aleatoriamente
            if order_count >= n:
                selected_orders = list(range(n))
            else:
                selected_orders = sorted(random.sample(range(n), min(order_count, n)))
            
            print(f"  Trial {trial + 1}/{current_trials}... Pedidos seleccionados: {len(selected_orders)} de {n}")
            
            # Resolver con Dinkelbach usando TODOS los pasillos
            trial_start = time.time()
            x_sol, y_sol, ratio = solve_dinkelbach_orders(
                orders, aisles, l_bound, r_bound, selected_orders, max_lambda, show_iterations=True
            )
            trial_time = time.time() - trial_start
            
            if x_sol is not None:
                num_orders = sum(x_sol)
                num_aisles = sum(y_sol)
                
                # Convertir índices de pasillos a IDs reales
                selected_aisle_ids = []
                for j in range(len(y_sol)):
                    if y_sol[j] == 1:
                        selected_aisle_ids.append(aisles[j]['id'])
                
                print(f"    Ratio: {ratio:.4f} ({num_orders} pedidos, {num_aisles} pasillos) - Tiempo: {trial_time:.2f}s")
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_solution = (x_sol, y_sol)
                    best_orders = selected_orders.copy()
                    best_aisles = selected_aisle_ids
                    
                    print(f"    *** NUEVA MEJOR SOLUCIÓN GLOBAL: {ratio:.4f} ***")
                    print(f"    *** ACTUALIZANDO λ_global a {ratio:.6f} ***")
                
                if ratio > phase_best_ratio:
                    phase_best_ratio = ratio
                
                # ACTUALIZAR MAX_LAMBDA GLOBALMENTE
                if ratio > max_lambda:
                    max_lambda = ratio
            else:
                print(f"    Sin solución - Tiempo: {trial_time:.2f}s")
        
        phase_time = time.time() - phase_start_time
        print(f"\nFASE {phase} COMPLETADA:")
        print(f"  Mejor ratio de la fase: {phase_best_ratio:.4f}")
        print(f"  Mejor ratio global: {best_ratio:.4f}")
        print(f"  Tiempo de la fase: {phase_time:.2f}s")
        print(f"  λ_global actual: {max_lambda:.6f}")
        print()
        
        phase += 1
        # Reducir trials para fases posteriores
        current_trials = max(current_trials + 1, 5)
    
    total_time = time.time() - start_time
    print(f"======================================================================")
    print(f"BÚSQUEDA PROGRESIVA COMPLETADA")
    print(f"Tiempo total: {total_time:.2f}s ({total_time/60:.1f} minutos)")
    print(f"Mejor λ encontrado: {max_lambda:.6f}")
    print()
    print(f"RESUMEN:")
    print(f"Mejor ratio encontrado: {best_ratio:.4f}")
    print()
    
    if best_solution:
        x_sol, y_sol = best_solution
        num_orders = sum(x_sol)
        num_aisles = sum(y_sol)
        
        # Calcular cantidad total
        total_quantity = 0
        selected_order_ids = []
        for i, selected in enumerate(x_sol):
            if selected == 1:
                order_idx = best_orders[i]
                selected_order_ids.append(orders[order_idx]['id'])
                total_quantity += orders[order_idx]['total_quantity']
        
        print(f"============================================================")
        print(f"MEJOR SOLUCIÓN ENCONTRADA")
        print(f"============================================================")
        print(f"Pedidos seleccionados ({num_orders}): {selected_order_ids}")
        print(f"Pasillos seleccionados ({num_aisles}): {best_aisles}")
        print(f"Cantidad total: {total_quantity}")
        print(f"Número de pasillos: {num_aisles}")
        print(f"Ratio final: {total_quantity}/{num_aisles} = {best_ratio:.6f}")
        print(f"Tiempo total: {total_time:.2f} segundos")
        print()
        print(f"Verificación de rango: {l_bound} <= {total_quantity} <= {r_bound}")
        if l_bound <= total_quantity <= r_bound:
            print("✓ Restricción de rango cumplida")
        else:
            print("✗ Restricción de rango violada")
        
        return {
            'ratio': best_ratio,
            'orders': selected_order_ids,
            'aisles': best_aisles,
            'num_orders': num_orders,
            'num_aisles': num_aisles,
            'total_quantity': total_quantity,
            'time': total_time
        }
    else:
        print("No se encontró solución")
        return None

def main():
    if len(sys.argv) != 2:
        print("Uso: python solver_random_orders.py <archivo_entrada>")
        sys.exit(1)
    
    filename = sys.argv[1]
    
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        n, m, k, orders, aisles, l, r = parse_input(lines)
        
        print(f"Instancia: {filename}")
        print(f"Pedidos: {n}, Ítems: {m}, Pasillos: {k}")
        print(f"Rango: [{l}, {r}]")
        print()
        
        result = solve_with_progressive_search(orders, aisles, l, r, time_limit_minutes=10)
        
        if result:
            # Guardar resultado
            output_filename = f"result_{filename.replace('.txt', '').replace('instances/', '')}.json"
            with open(output_filename, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nResultado guardado en: {output_filename}")
        
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {filename}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 