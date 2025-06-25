"""
Script para resolver el problema de optimización fraccional
utilizando el algoritmo de Dinkelbach con estrategia incremental de pasillos.

Estrategia: Mantener un conjunto base de pasillos y agregar 10 pasillos nuevos aleatorios en cada iteración.

Uso (PowerShell):
Get-Content .\\datasets\\a\\instance_0010.txt | python solver_random_aisles_2.py

Dependencias:
pip install docplex
"""
import sys
import random
import time
from docplex.mp.model import Model
import threading
import numpy as np

def parse_input(lines):
    """Parsea los datos de entrada desde las líneas proporcionadas."""
    line_idx = 0
    
    # Primera línea: n, m, k
    n, m, k = map(int, lines[line_idx].strip().split())
    line_idx += 1
    
    # Siguientes n líneas: pedidos
    orders = []
    for i in range(n):
        parts = lines[line_idx].strip().split()
        line_idx += 1
        
        # Formato: num_items item1 qty1 item2 qty2 ...
        num_items = int(parts[0])
        total_quantity = 0
        items = {}
        
        for j in range(num_items):
            item_id = int(parts[1 + j*2])
            quantity = int(parts[1 + j*2 + 1])
            items[item_id] = quantity
            total_quantity += quantity
        
        orders.append({
            'id': i,
            'total_quantity': total_quantity,
            'items': items
        })
    
    # Siguientes k líneas: pasillos
    aisles = []
    for i in range(k):
        parts = lines[line_idx].strip().split()
        line_idx += 1
        
        # Formato: num_items item1 cap1 item2 cap2 ...
        num_items = int(parts[0])
        items = {}
        
        for j in range(num_items):
            item_id = int(parts[1 + j*2])
            capacity = int(parts[1 + j*2 + 1])
            items[item_id] = capacity
        
        aisles.append({
            'id': i,
            'items': items
        })
    
    # Última línea: L, R
    l_bound, r_bound = map(int, lines[line_idx].strip().split())
    
    return n, k, orders, aisles, l_bound, r_bound

def get_all_items(orders, aisles):
    """
    Obtiene todos los items únicos que aparecen tanto en pedidos como en pasillos.
    """
    items_in_orders = set()
    for order in orders:
        items_in_orders.update(order['items'].keys())
    
    items_in_aisles = set()
    for aisle in aisles:
        items_in_aisles.update(aisle['items'].keys())
    
    # Solo nos interesan los items que están en ambos
    relevant_items = items_in_orders.intersection(items_in_aisles)
    return relevant_items

def heuristic_warm_start(orders, aisles, l_bound, r_bound, verbose=False):
    """
    Heurística para generar un conjunto inicial inteligente de pasillos.
    Selecciona pedidos hasta llegar al rango [L,R] y determina pasillos necesarios.
    Limita el número de pasillos para no ser demasiado agresivo.
    """
    rng = np.random.default_rng(42)
    selected_orders = []
    total_quantity = 0

    # Seleccionar pedidos aleatórios hasta atingir el límite inferior
    order_indices = rng.permutation(len(orders))
    for o in order_indices:
        order_quantity = orders[o]['total_quantity']
        if total_quantity + order_quantity <= r_bound:
            selected_orders.append(o)
            total_quantity += order_quantity
            if total_quantity >= l_bound:
                break

    # Determinar pasillos necesarios para estos pedidos
    selected_aisles = set()
    for o in selected_orders:
        for item_id, quantity in orders[o]['items'].items():
            # Buscar pasillos que tienen este item
            for a in range(len(aisles)):
                if item_id in aisles[a]['items'] and aisles[a]['items'][item_id] > 0:
                    selected_aisles.add(a)

    # LIMITAR EL NÚMERO DE PASILLOS PARA EL WARM START - CONSERVADOR
    max_warm_start_aisles = 10  # Máximo 10 pasillos en el warm start
    original_count = len(selected_aisles)
    if len(selected_aisles) > max_warm_start_aisles:
        # Seleccionar aleatoriamente un subconjunto
        selected_aisles = set(rng.choice(list(selected_aisles), size=max_warm_start_aisles, replace=False))
        if verbose:
            print(f"  Limitando warm start a {max_warm_start_aisles} pasillos de {original_count} encontrados")

    # Calcular ratio estimado
    if selected_aisles:
        estimated_ratio = total_quantity / len(selected_aisles)
        if verbose:
            print(f"  Heurística warm start: total_quantity = {total_quantity}, aisles = {len(selected_aisles)}, ratio = {estimated_ratio:.4f}")
    else:
        estimated_ratio = 0.0
        if verbose:
            print(f"  Heurística warm start: No se encontraron pasillos necesarios")

    return estimated_ratio, selected_orders, selected_aisles

def solve_dinkelbach_with_max_lambda(orders, selected_aisles, l_bound, r_bound, selected_aisle_indices, k, initial_max_lambda, verbose=False):
    """
    Resuelve usando Dinkelbach con el enfoque de max_lambda inicial dado.
    """
    n = len(orders)
    num_aisles_to_use = len(selected_aisles)
    
    # Obtener todos los items relevantes
    relevant_items = set()
    for order in orders:
        relevant_items.update(order['items'].keys())
    for aisle in selected_aisles:
        relevant_items.update(aisle['items'].keys())
    
    # USAR EL MAX_LAMBDA INICIAL PASADO COMO PARÁMETRO
    max_lambda = initial_max_lambda
    
    max_iter = 100
    epsilon = 1e-6
    
    for iteration in range(max_iter):
        # Mostrar el lambda que se va a usar en esta iteración
        if verbose:
            print(f"    Iter {iteration}: λ_max_usado={max_lambda:.6f}", end="")
        
        # Crear modelo para esta iteración de Dinkelbach
        mdl = Model(name=f'dinkelbach_iter_{iteration}')

        # Configuración de multithreading y rendimiento de CPLEX
        mdl.context.cplex_parameters.threads = 12  # Número de threads
        mdl.context.cplex_parameters.mip.display = 0  # Silenciar CPLEX
        mdl.context.cplex_parameters.mip.tolerances.mipgap = 0.01  # Gap de tolerancia 1%
        
        # Parámetros adicionales de rendimiento (solo los válidos)
        try:
            mdl.context.cplex_parameters.parallel = 1  # Modo paralelo determinístico
            mdl.context.cplex_parameters.mip.strategy.heuristicfreq = 20  # Heurísticas más frecuentes
            mdl.context.cplex_parameters.emphasis.mip = 1  # Énfasis en factibilidad
        except:
            pass  # Ignorar si algún parámetro no está disponible

        # Variables de decisión
        x_vars = mdl.binary_var_list(n, name='x')
        y_vars = mdl.binary_var_list(num_aisles_to_use, name='y')  # Solo para pasillos seleccionados
        
        # Expresiones para objetivo
        total_quantity_expr = mdl.sum(orders[i]['total_quantity'] * x_vars[i] for i in range(n))
        total_aisles_expr = mdl.sum(y_vars[j] for j in range(num_aisles_to_use))  # Variables, no constante
        
        # Restricción de rango [L, R]
        mdl.add_range(l_bound, total_quantity_expr, r_bound)
        
        # Forzar que al menos un pasillo esté abierto
        mdl.add_constraint(total_aisles_expr >= 1, ctname="force_aisle_opening")
        
        # Restricciones de capacidad para TODOS los items relevantes
        constraints_added = 0
        for item_id in relevant_items:
            demand = mdl.sum(orders[i]['items'].get(item_id, 0) * x_vars[i] for i in range(n))
            capacity = mdl.sum(selected_aisles[j]['items'].get(item_id, 0) * y_vars[j] for j in range(num_aisles_to_use))
            mdl.add_constraint(demand <= capacity, ctname=f"capacity_item_{item_id}")
            constraints_added += 1
        
        if verbose and iteration == 0:
            print(f" ({constraints_added} restricciones de capacidad)", end="")
        
        # Función objetivo de Dinkelbach: max(f(x,y) - lambda * g(x,y))
        # USAR EL MAYOR LAMBDA ENCONTRADO
        mdl.set_objective('max', total_quantity_expr - max_lambda * total_aisles_expr)
        
        # Resolver
        solution = mdl.solve()
        
        if not solution:
            if verbose:
                print(f" → No factible")
            return None, None, None  # No factible
        
        # Obtener valores de las variables
        x_sol = [x_vars[i].solution_value for i in range(n)]
        y_sol_subset = [y_vars[j].solution_value for j in range(num_aisles_to_use)]
        
        # Expandir y_sol al tamaño completo k
        y_sol = [0.0] * k
        for i, aisle_idx in enumerate(selected_aisle_indices):
            y_sol[aisle_idx] = y_sol_subset[i]
        
        # Calcular numerador y denominador (ahora ambos son variables)
        numerator = sum(orders[i]['total_quantity'] * x_sol[i] for i in range(n))
        denominator = sum(y_sol_subset[j] for j in range(num_aisles_to_use))
        
        # Calcular el nuevo lambda para la siguiente iteración
        if denominator > 0:
            lambda_val = numerator / denominator
        else:
            lambda_val = 0.0
            
        # Valor objetivo actual (con el max_lambda usado en el modelo)
        obj_val = numerator - max_lambda * denominator
        
        if verbose:
            print(f" → λ_calculado={lambda_val:.6f}, f(x,y)={numerator:.2f}, g(x,y)={denominator} ({int(denominator)} pasillos), ratio={lambda_val:.6f}, obj={obj_val:.6f}")
        
        # CONDICIÓN DE TERMINACIÓN TEMPRANA: si el nuevo lambda <= max_lambda usado
        if lambda_val <= max_lambda + epsilon:
            if verbose:
                print(f"    *** TERMINANDO: λ_calculado ({lambda_val:.6f}) <= λ_max_usado ({max_lambda:.6f}) - ratio {lambda_val:.6f} ***")
            return x_sol, y_sol, lambda_val
        
        # Actualizar max_lambda para la siguiente iteración
        if lambda_val > max_lambda:
            max_lambda = lambda_val
            if verbose:
                print(f"    *** Actualizando λ_max a {max_lambda:.6f} ***")
        
        # Verificar convergencia estándar
        if abs(lambda_val - max_lambda) < epsilon:
            if verbose:
                print(f"    *** CONVERGENCIA: Diferencia {abs(lambda_val - max_lambda):.8f} < {epsilon} ***")
            return x_sol, y_sol, lambda_val
    
    # Si llegamos aquí, no convergió
    if verbose:
        print(f"    *** NO CONVERGIÓ después de {max_iter} iteraciones ***")
    return x_sol, y_sol, lambda_val if 'lambda_val' in locals() else 0.0

def solve_with_incremental_aisles(orders, aisles, l_bound, r_bound, time_limit_minutes=10, verbose=True):
    """
    Resuelve el problema usando estrategia incremental de pasillos:
    - Mantiene un conjunto base de pasillos exitosos
    - Agrega 10 pasillos nuevos aleatorios en cada iteración
    - Los pasillos base se mantienen, los nuevos se exploran
    """
    n = len(orders)
    k = len(aisles)
    
    best_ratio = -1
    best_x = None
    best_y = None
    
    # MAX_LAMBDA GLOBAL QUE PERSISTE ENTRE ITERACIONES
    max_lambda = 0.0
    
    # CONJUNTO BASE DE PASILLOS (inicialmente vacío)
    base_aisles = set()
    
    # Configuración
    aisles_per_iteration = 50  # Número de pasillos nuevos a agregar en cada iteración
    initial_aisles = 30       # Número inicial de pasillos para la primera iteración
    max_base_aisles = 101     # Máximo número de pasillos en el conjunto base
    time_limit_seconds = time_limit_minutes * 60
    start_time = time.time()
    
    # Función para calcular capacidad máxima posible
    def calculate_max_capacity(aisle_indices):
        total_capacity_per_item = {}
        current_aisles_data = [aisles[i] for i in aisle_indices]
        for aisle in current_aisles_data:
            for item_id, capacity in aisle['items'].items():
                total_capacity_per_item[item_id] = total_capacity_per_item.get(item_id, 0) + capacity
        
        max_possible_quantity = 0
        for order in orders:
            can_satisfy_order = True
            for item_id, demand in order['items'].items():
                available_capacity = total_capacity_per_item.get(item_id, 0)
                if available_capacity < demand:
                    can_satisfy_order = False
                    break
            if can_satisfy_order:
                max_possible_quantity += order['total_quantity']
        return max_possible_quantity
    
    if verbose:
        print(f"INICIANDO BÚSQUEDA CON PASILLOS INCREMENTALES")
        print(f"Pasillos disponibles: {k}")
        print(f"Límite de tiempo: {time_limit_minutes} minutos")
        print(f"Estrategia:")
        print(f"  - Pasillos iniciales: {initial_aisles}")
        print(f"  - Pasillos nuevos por iteración: {aisles_per_iteration}")
        print(f"  - Máximo pasillos base: {max_base_aisles}")
        print("-" * 70)
    
    # SIN WARM START: Empezar con conjunto base vacío
    base_aisles = set()
    if verbose:
        print(f"\nEMPEZANDO SIN WARM START - conjunto base vacío")
        print("-" * 70)
    
    iteration = 1
    
    while True:
        iteration_start_time = time.time()
        
        # Verificar límite de tiempo
        elapsed_time = time.time() - start_time
        if elapsed_time >= time_limit_seconds:
            if verbose:
                print(f"\n*** LÍMITE DE TIEMPO ALCANZADO ({time_limit_minutes} minutos) ***")
            break
        
        if verbose:
            print(f"\nITERACIÓN {iteration}:")
            print(f"  Pasillos base actuales: {len(base_aisles)}")
            print(f"  Tiempo transcurrido: {elapsed_time:.1f}s / {time_limit_seconds}s")
        
        # Determinar pasillos disponibles para selección aleatoria
        available_aisles = set(range(k)) - base_aisles
        
        # Determinar cuántos pasillos nuevos agregar
        if iteration == 1:
            # Primera iteración: usar número inicial de pasillos
            new_aisles_count = min(initial_aisles, len(available_aisles))
        else:
            # Iteraciones siguientes: agregar pasillos incrementales
            new_aisles_count = min(aisles_per_iteration, len(available_aisles))
        
        # Si no hay pasillos disponibles para agregar, terminar
        if new_aisles_count == 0:
            if verbose:
                print(f"  No hay más pasillos disponibles para agregar. Terminando.")
            break
        
        # Seleccionar pasillos nuevos aleatoriamente
        new_aisles = set(random.sample(list(available_aisles), new_aisles_count))
        
        # Conjunto total de pasillos para esta iteración
        current_aisles_set = base_aisles.union(new_aisles)
        current_aisle_indices = list(current_aisles_set)
        current_aisles_data = [aisles[i] for i in current_aisle_indices]
        
        if verbose:
            print(f"  Pasillos nuevos: {len(new_aisles)} ({sorted(list(new_aisles))[:5]}{'...' if len(new_aisles) > 5 else ''})")
            print(f"  Total pasillos: {len(current_aisle_indices)}")
        
        # Verificar capacidad
        max_possible_quantity = calculate_max_capacity(current_aisle_indices)
        
        if max_possible_quantity < l_bound:
            if verbose:
                print(f"  Capacidad insuficiente (max={max_possible_quantity} < L={l_bound})")
                print(f"  Agregando pasillos nuevos al conjunto base sin resolver...")
            
            # Agregar algunos pasillos nuevos al conjunto base para la siguiente iteración
            base_aisles.update(new_aisles)
            
            # Limitar el tamaño del conjunto base
            if len(base_aisles) > max_base_aisles:
                # Mantener solo los pasillos más recientes
                base_aisles = set(list(base_aisles)[-max_base_aisles:])
            
            iteration += 1
            continue
        
        if verbose:
            print(f"  Capacidad suficiente (max={max_possible_quantity} >= L={l_bound})")
            print(f"  Resolviendo con Dinkelbach...")
        
        # Resolver con Dinkelbach
        x_sol, y_sol, ratio = solve_dinkelbach_with_max_lambda(
            orders, current_aisles_data, l_bound, r_bound, 
            current_aisle_indices, k, max_lambda, verbose=True
        )
        
        if x_sol is not None:
            if verbose:
                print(f"  Ratio obtenido: {ratio:.6f}")
            
            # Actualizar mejor solución global
            if ratio > best_ratio:
                best_ratio = ratio
                best_x = x_sol
                best_y = y_sol
                if verbose:
                    print(f"    *** NUEVA MEJOR SOLUCIÓN GLOBAL: {ratio:.6f} ***")
                
                # Identificar pasillos usados en la mejor solución
                used_aisles = set()
                for i, aisle_idx in enumerate(current_aisle_indices):
                    if y_sol[aisle_idx] > 0.5:
                        used_aisles.add(aisle_idx)
                
                # SOLO agregar pasillos exitosos al conjunto base
                base_aisles.update(used_aisles)
                
                if verbose:
                    print(f"    Pasillos usados en mejor solución: {len(used_aisles)} ({sorted(list(used_aisles))[:5]}{'...' if len(used_aisles) > 5 else ''})")
                    print(f"    Agregando {len(used_aisles)} pasillos exitosos al conjunto base")
                    print(f"    Nuevo conjunto base: {len(base_aisles)} pasillos")
            else:
                # Si no mejoró, NO agregar pasillos al conjunto base
                if verbose:
                    print(f"    No mejoró - manteniendo conjunto base actual ({len(base_aisles)} pasillos)")
            
            # ACTUALIZAR MAX_LAMBDA GLOBALMENTE
            if ratio > max_lambda:
                max_lambda = ratio
                if verbose:
                    print(f"    *** ACTUALIZANDO λ_global a {max_lambda:.6f} ***")
        else:
            if verbose:
                print(f"  No factible - manteniendo conjunto base actual")
        
        # NUEVA LÓGICA: Al llegar a max_base_aisles, resetear al conjunto de la mejor solución
        if len(base_aisles) > max_base_aisles:
            if best_x is not None and best_y is not None:
                # Identificar pasillos usados en la mejor solución actual
                solution_aisles = set()
                for i in range(k):
                    if best_y[i] > 0.5:
                        solution_aisles.add(i)
                
                if solution_aisles:
                    old_base_size = len(base_aisles)
                    base_aisles = solution_aisles.copy()
                    if verbose:
                        print(f"    *** COMPACTACIÓN: Resetear conjunto base de {old_base_size} → {len(base_aisles)} pasillos (mejor solución) ***")
                        print(f"    Pasillos en la mejor solución: {sorted(list(solution_aisles))}")
                else:
                    # Fallback: mantener los pasillos más recientes
                    base_aisles = set(list(base_aisles)[-max_base_aisles:])
                    if verbose:
                        print(f"    Limitando conjunto base a {max_base_aisles} pasillos (fallback)")
            else:
                # Fallback: mantener los pasillos más recientes
                base_aisles = set(list(base_aisles)[-max_base_aisles:])
                if verbose:
                    print(f"    Limitando conjunto base a {max_base_aisles} pasillos (no hay solución)")
        
        iteration_time = time.time() - iteration_start_time
        if verbose:
            print(f"  Tiempo de iteración: {iteration_time:.2f}s")
            print(f"  Mejor ratio global: {best_ratio:.6f}")
            print(f"  λ_global actual: {max_lambda:.6f}")
        
        iteration += 1
        
        # Condición de terminación adicional: si ya exploramos muchos pasillos
        if len(base_aisles) >= max_base_aisles and len(available_aisles) < aisles_per_iteration:
            if verbose:
                print(f"\n*** TERMINANDO: Conjunto base completo y pocos pasillos disponibles ***")
            break
    
    total_time = time.time() - start_time
    if verbose:
        print(f"\n" + "="*70)
        print(f"BÚSQUEDA INCREMENTAL COMPLETADA")
        print(f"Iteraciones realizadas: {iteration - 1}")
        print(f"Tiempo total: {total_time:.2f}s ({total_time/60:.1f} minutos)")
        print(f"Pasillos en conjunto base final: {len(base_aisles)}")
        print(f"Mejor λ encontrado: {max_lambda:.6f}")
    
    return best_x, best_y, best_ratio

def main():
    # Leer datos de entrada
    print("INFO: Parseando datos de entrada...")
    
    # Verificar si hay argumento de archivo
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"ERROR: Archivo no encontrado: {filename}")
            return
    else:
        lines = sys.stdin.readlines()
    
    if not lines:
        print("ERROR: No se proporcionaron datos de entrada.")
        return
    
    n, k, orders, aisles, l_bound, r_bound = parse_input(lines)
    
    m = len(set().union(*(order['items'].keys() for order in orders)))
    
    print(f"INFO: n={n}, m={m}, k={k}, L={l_bound}, R={r_bound}")
    
    # Obtener items relevantes
    relevant_items = set()
    for order in orders:
        relevant_items.update(order['items'].keys())
    for aisle in aisles:
        relevant_items.update(aisle['items'].keys())
    
    print(f"Items relevantes para restricciones de capacidad: {len(relevant_items)}")
    print()
    
    # Ejecutar solver con búsqueda incremental
    start_time = time.time()
    best_x_sol, best_y_sol, best_ratio = solve_with_incremental_aisles(
        orders, aisles, l_bound, r_bound, time_limit_minutes=10
    )
    end_time = time.time()
    
    print(f"\nRESUMEN:")
    if best_x_sol is not None:
        print(f"Mejor ratio encontrado: {best_ratio:.4f}")
    else:
        print("No se encontró ninguna solución factible.")
        print(f"Tiempo total: {end_time - start_time:.2f} segundos")
        return
    
    # Mostrar mejor solución
    print()
    print("=" * 60)
    print("MEJOR SOLUCIÓN ENCONTRADA")
    print("=" * 60)
    
    # Pedidos seleccionados
    selected_orders = [i for i in range(n) if best_x_sol[i] > 0.5]
    print(f"Pedidos seleccionados ({len(selected_orders)}): {selected_orders}")
    
    # Pasillos seleccionados
    selected_aisles = [i for i in range(k) if best_y_sol[i] > 0.5]
    print(f"Pasillos seleccionados ({len(selected_aisles)}): {selected_aisles}")
    
    # Calcular cantidad total y número de pasillos
    total_quantity = sum(orders[i]['total_quantity'] for i in selected_orders)
    num_selected_aisles = len(selected_aisles)
    
    print(f"Cantidad total: {total_quantity}")
    print(f"Número de pasillos: {num_selected_aisles}")
    print(f"Ratio final: {total_quantity}/{num_selected_aisles} = {total_quantity/num_selected_aisles:.6f}")
    print(f"Tiempo total: {end_time - start_time:.2f} segundos")
    
    # Verificar restricciones
    print()
    print(f"Verificación de rango: {l_bound} <= {total_quantity} <= {r_bound}")
    if l_bound <= total_quantity <= r_bound:
        print("✓ Restricción de rango cumplida")
    else:
        print("✗ Restricción de rango NO cumplida")

if __name__ == "__main__":
    main() 