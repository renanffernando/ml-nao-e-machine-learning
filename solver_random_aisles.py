"""
Script simplificado para resolver el problema de optimización fraccional
utilizando el algoritmo de Dinkelbach con selección aleatoria de pasillos.

Uso (PowerShell):
Get-Content .\datasets\a\instance_0010.txt | python solver_random_aisles_2.py

Dependencias:
pip install docplex
"""
import sys
import random
import time
from docplex.mp.model import Model
import threading

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

def solve_dinkelbach_with_limited_aisles(n, k, orders, aisles, l_bound, r_bound, relevant_items, max_aisles=100, verbose=False):
    """
    Resuelve el problema usando Dinkelbach con un máximo de pasillos disponibles.
    """
    lambda_val = 0.0
    max_lambda = 0.0  # Mantener el mayor lambda encontrado
    epsilon = 1e-6  # Más precisión
    max_iter = 200  # Más iteraciones
    
    # Seleccionar máximo max_aisles pasillos
    num_aisles_to_use = min(max_aisles, k)
    if num_aisles_to_use < k:
        # Selección aleatoria de pasillos
        selected_aisle_indices = random.sample(range(k), num_aisles_to_use)
        selected_aisles = [aisles[i] for i in selected_aisle_indices]
    else:
        # Usar todos los pasillos
        selected_aisle_indices = list(range(k))
        selected_aisles = aisles
    
    if verbose:
        print(f"    Iniciando Dinkelbach con {num_aisles_to_use} pasillos (de {k} disponibles)...")
    
    for iteration in range(max_iter):
        # Mostrar el lambda que se va a usar en esta iteración
        if verbose:
            print(f"    Iter {iteration}: λ_max_usado={max_lambda:.6f}", end="")
        
        # Crear modelo para esta iteración de Dinkelbach
        mdl = Model(name=f'dinkelbach_iter_{iteration}')
        mdl.context.cplex_parameters.mip.display = 0  # Silenciar CPLEX
        
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
        
        # Restricciones de capacidad para todos los items relevantes
        for item_id in relevant_items:
            demand = mdl.sum(orders[i]['items'].get(item_id, 0) * x_vars[i] for i in range(n))
            capacity = mdl.sum(selected_aisles[j]['items'].get(item_id, 0) * y_vars[j] for j in range(num_aisles_to_use))  # Solo pasillos seleccionados
            mdl.add_constraint(demand <= capacity, ctname=f"capacity_item_{item_id}")
        
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
            current_ratio = numerator / denominator if denominator > 0 else 0
            num_open_aisles = sum(1 for y in y_sol_subset if y > 0.5)
            print(f" → λ_calculado={lambda_val:.6f}, f(x,y)={numerator:.2f}, g(x,y)={denominator:.0f} ({num_open_aisles} pasillos), ratio={current_ratio:.6f}, obj={obj_val:.6f}")
        
        # NUEVA CONDICIÓN: Si el ratio calculado es <= max_lambda, terminar
        if lambda_val <= max_lambda:
            ratio = numerator / denominator if denominator > 0 else 0
            if verbose:
                print(f"    *** TERMINANDO: λ_calculado ({lambda_val:.6f}) <= λ_max_usado ({max_lambda:.6f}) - ratio {ratio:.6f} ***")
            return x_sol, y_sol, ratio
        
        # Verificar convergencia
        if abs(obj_val) < epsilon:
            ratio = numerator / denominator if denominator > 0 else 0
            if verbose:
                print(f"    *** CONVERGIÓ en iteración {iteration} con ratio {ratio:.6f} ***")
            return x_sol, y_sol, ratio
        
        # ACTUALIZAR EL MAYOR LAMBDA ENCONTRADO para la siguiente iteración
        if lambda_val > max_lambda:
            max_lambda = lambda_val
            if verbose:
                print(f"    *** Actualizando λ_max a {max_lambda:.6f} ***")
        
    # Si no converge, devolver la última solución
    if 'x_sol' in locals():
        ratio = numerator / denominator if denominator > 0 else 0
        if verbose:
            print(f"    *** NO CONVERGIÓ después de {max_iter} iteraciones. Último ratio: {ratio:.6f} (λ_max={max_lambda:.6f}) ***")
        return x_sol, y_sol, ratio
    else:
        return None, None, None

def solve_with_random_aisles(orders, aisles, l_bound, r_bound, num_trials=5, max_aisles_per_trial=None, verbose=True):
    """
    Resuelve el problema usando pasillos aleatorios con múltiples trials.
    Si no se especifica max_aisles_per_trial, usa min(100, k).
    """
    # Para Windows, usamos un timeout simple con time
    def timeout_handler():
        pass
    
    n = len(orders)
    k = len(aisles)
    
    if max_aisles_per_trial is None:
        max_aisles_per_trial = min(100, k)
    
    best_ratio = -1
    best_x = None
    best_y = None
    
    # MAX_LAMBDA GLOBAL QUE PERSISTE ENTRE TRIALS
    max_lambda = 0.0
    
    # Función para calcular capacidad máxima posible (definida una sola vez)
    def calculate_max_capacity(aisle_indices):
        if verbose:
            print(f"        Calculando capacidad para {len(aisle_indices)} pasillos...")
        total_capacity_per_item = {}
        current_aisles = [aisles[i] for i in aisle_indices]
        for aisle in current_aisles:
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
        
        if verbose:
            print(f"        Capacidad calculada: {max_possible_quantity}")
        return max_possible_quantity
    
    if verbose:
        print(f"Número de trials: {num_trials}")
        if max_aisles_per_trial < k:
            print(f"Pasillos disponibles: {k}")
            print(f"Pasillos máximos por trial: {max_aisles_per_trial}")
        else:
            print(f"Pasillos disponibles: {k}")
        print("-" * 60)
    
    for trial in range(num_trials):
        if verbose:
            print(f"Trial {trial+1}/{num_trials}...")
        
        try:
            # Timeout simple con time
            trial_start_time = time.time()
            timeout_seconds = 30
            
            # Seleccionar pasillos aleatoriamente para este trial
            num_aisles_to_use = min(max_aisles_per_trial, k)
            if verbose:
                print(f"    Seleccionando {num_aisles_to_use} pasillos...")
                
            if num_aisles_to_use < k:
                selected_aisle_indices = random.sample(range(k), num_aisles_to_use)
                selected_aisles = [aisles[i] for i in selected_aisle_indices]
                if verbose:
                    print(f"    Iniciando con {num_aisles_to_use} pasillos (de {k} disponibles): {selected_aisle_indices[:5]}...")
            else:
                selected_aisle_indices = list(range(k))
                selected_aisles = aisles
                if verbose:
                    print(f"    Iniciando con {k} pasillos disponibles...")
            
            # VERIFICAR Y EXPANDIR SI ES NECESARIO PARA CUMPLIR L
            max_aisles_allowed = min(num_aisles_to_use * 2 - 1, k)  # Doble menos 1, pero no más que k
            
            if verbose:
                print(f"    Verificando capacidad inicial...")
            # Verificar capacidad inicial
            max_possible_quantity = calculate_max_capacity(selected_aisle_indices)
            
            # Si no alcanza L, intentar agregar más pasillos
            if max_possible_quantity < l_bound and len(selected_aisle_indices) < max_aisles_allowed:
                if verbose:
                    print(f"    Capacidad inicial insuficiente (max={max_possible_quantity} < L={l_bound}), expandiendo...")
                
                # Obtener pasillos disponibles para agregar
                remaining_aisles = [i for i in range(k) if i not in selected_aisle_indices]
                
                # Contador de seguridad para evitar bucles infinitos
                expansion_attempts = 0
                max_expansion_attempts = min(max_aisles_allowed - len(selected_aisle_indices), 10)  # Máximo 10 expansiones
                
                if verbose:
                    print(f"    Máximo {max_expansion_attempts} expansiones permitidas...")
                
                # Agregar pasillos uno por uno hasta alcanzar L o el límite máximo
                while (max_possible_quantity < l_bound and 
                       len(selected_aisle_indices) < max_aisles_allowed and 
                       remaining_aisles and
                       expansion_attempts < max_expansion_attempts):
                    
                    if verbose:
                        print(f"      Expansión {expansion_attempts + 1}/{max_expansion_attempts}...")
                    
                    # Seleccionar un pasillo aleatorio de los restantes
                    new_aisle = random.choice(remaining_aisles)
                    selected_aisle_indices.append(new_aisle)
                    remaining_aisles.remove(new_aisle)
                    expansion_attempts += 1
                    
                    # Recalcular capacidad
                    max_possible_quantity = calculate_max_capacity(selected_aisle_indices)
                    
                    if verbose:
                        print(f"      Agregado pasillo {new_aisle}, nueva capacidad max={max_possible_quantity} (pasillos: {len(selected_aisle_indices)})")
                
                # Actualizar selected_aisles con la nueva selección
                selected_aisles = [aisles[i] for i in selected_aisle_indices]
            
            # Verificación final: si aún no alcanza L, saltar este trial
            if max_possible_quantity < l_bound:
                if verbose:
                    print(f"    Capacidad final insuficiente (max={max_possible_quantity} < L={l_bound}) con {len(selected_aisle_indices)} pasillos, saltando...")
                continue
            
            if verbose:
                print(f"    Capacidad suficiente (max={max_possible_quantity} >= L={l_bound}) con {len(selected_aisle_indices)} pasillos")
                print(f"    Iniciando Dinkelbach...")
            
            # Verificar timeout antes de Dinkelbach
            if time.time() - trial_start_time > timeout_seconds:
                if verbose:
                    print(f"    *** TIMEOUT antes de Dinkelbach ***")
                continue
            
            # Resolver con Dinkelbach usando el max_lambda actual
            x_sol, y_sol, ratio = solve_dinkelbach_with_max_lambda(
                orders, selected_aisles, l_bound, r_bound, 
                selected_aisle_indices, k, max_lambda, verbose=True
            )
            
            if x_sol is not None:
                if verbose:
                    print(f"    Factible - Ratio: {ratio:.4f}")
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_x = x_sol
                    best_y = y_sol
                    if verbose:
                        print("      *** NUEVA MEJOR SOLUCIÓN ***")
                
                # ACTUALIZAR MAX_LAMBDA GLOBALMENTE con el ratio encontrado
                if ratio > max_lambda:
                    max_lambda = ratio
                    if verbose:
                        print(f"      *** ACTUALIZANDO λ_global a {max_lambda:.6f} ***")
            else:
                if verbose:
                    print("    No factible")
                    
        except Exception as e:
            if verbose:
                print(f"    *** ERROR en trial {trial+1}: {e} ***")
            continue
    
    return best_x, best_y, best_ratio

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
        mdl.context.cplex_parameters.mip.display = 0  # Silenciar CPLEX
        
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

def solve_with_progressive_search(orders, aisles, l_bound, r_bound, time_limit_minutes=10, verbose=True):
    """
    Resuelve el problema usando búsqueda progresiva:
    - Empieza con 8 pasillos, duplica hasta llegar a k
    - Empieza con 10 trials, reduce 1 cada iteración (mínimo 1)
    - Límite de tiempo: 10 minutos
    """
    n = len(orders)
    k = len(aisles)
    
    best_ratio = -1
    best_x = None
    best_y = None
    
    # INICIALIZAR MAX_LAMBDA UNA SOLA VEZ PARA TODA LA BÚSQUEDA
    max_lambda = 0.0
    
    # Función para calcular capacidad máxima posible (definida una sola vez)
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
    
    # Configuración inicial
    current_aisles = 8   # Empezar con 8 pasillos
    current_trials = 15  # Empezar con 10 trials
    time_limit_seconds = time_limit_minutes * 60
    start_time = time.time()
    
    if verbose:
        print(f"INICIANDO BÚSQUEDA PROGRESIVA")
        print(f"Pasillos disponibles: {k}")
        print(f"Límite de tiempo: {time_limit_minutes} minutos")
        print(f"Estrategia: Empezar con {current_aisles} pasillos, duplicar cada vez")
        print(f"Trials: Empezar con {current_trials}, reducir 1 cada iteración (mín 1)")
        print("-" * 70)
    
    phase = 1
    
    while current_aisles <= k:
        phase_start_time = time.time()
        
        # Verificar límite de tiempo
        elapsed_time = time.time() - start_time
        if elapsed_time >= time_limit_seconds:
            if verbose:
                print(f"\n*** LÍMITE DE TIEMPO ALCANZADO ({time_limit_minutes} minutos) ***")
            break
        
        # Ajustar número de pasillos (no exceder k)
        aisles_to_use = min(current_aisles, k)
        
        if verbose:
            print(f"\nFASE {phase}: {aisles_to_use} pasillos, {current_trials} trials")
            print(f"Tiempo transcurrido: {elapsed_time:.1f}s / {time_limit_seconds}s")
            print("-" * 50)
        
        # Ejecutar trials para esta fase
        phase_best_ratio = -1
        phase_best_x = None
        phase_best_y = None
        
        for trial in range(current_trials):
            # Verificar límite de tiempo en cada trial
            elapsed_time = time.time() - start_time
            if elapsed_time >= time_limit_seconds:
                if verbose:
                    print(f"*** LÍMITE DE TIEMPO ALCANZADO en trial {trial+1} ***")
                break
            
            if verbose:
                print(f"  Trial {trial+1}/{current_trials}...", end=" ")
            
            # Seleccionar pasillos aleatoriamente para este trial
            if aisles_to_use < k:
                selected_aisle_indices = random.sample(range(k), aisles_to_use)
                selected_aisles = [aisles[i] for i in selected_aisle_indices]
            else:
                selected_aisle_indices = list(range(k))
                selected_aisles = aisles
            
            # VERIFICAR Y EXPANDIR SI ES NECESARIO PARA CUMPLIR L
            max_aisles_allowed = min(aisles_to_use * 2 - 1, k)  # Doble menos 1, pero no más que k
            
            # Verificar capacidad inicial
            max_possible_quantity = calculate_max_capacity(selected_aisle_indices)
            
            # Si no alcanza L, intentar agregar más pasillos
            if max_possible_quantity < l_bound and len(selected_aisle_indices) < max_aisles_allowed:
                if verbose:
                    print(f"Capacidad inicial insuficiente (max={max_possible_quantity} < L={l_bound}), expandiendo...")
                
                # Obtener pasillos disponibles para agregar
                remaining_aisles = [i for i in range(k) if i not in selected_aisle_indices]
                
                # Contador de seguridad para evitar bucles infinitos
                expansion_attempts = 0
                max_expansion_attempts = max_aisles_allowed - len(selected_aisle_indices)
                
                # Agregar pasillos uno por uno hasta alcanzar L o el límite máximo
                while (max_possible_quantity < l_bound and 
                       len(selected_aisle_indices) < max_aisles_allowed and 
                       remaining_aisles and
                       expansion_attempts < max_expansion_attempts):
                    
                    # Seleccionar un pasillo aleatorio de los restantes
                    new_aisle = random.choice(remaining_aisles)
                    selected_aisle_indices.append(new_aisle)
                    remaining_aisles.remove(new_aisle)
                    expansion_attempts += 1
                    
                    # Recalcular capacidad
                    max_possible_quantity = calculate_max_capacity(selected_aisle_indices)
                    
                    if verbose:
                        print(f"    Agregado pasillo {new_aisle}, nueva capacidad max={max_possible_quantity} (pasillos: {len(selected_aisle_indices)})")
                
                # Actualizar selected_aisles con la nueva selección
                selected_aisles = [aisles[i] for i in selected_aisle_indices]
            
            # Verificación final: si aún no alcanza L, saltar este trial
            if max_possible_quantity < l_bound:
                if verbose:
                    print(f"Capacidad final insuficiente (max={max_possible_quantity} < L={l_bound}) con {len(selected_aisle_indices)} pasillos, saltando...")
                continue
            
            if verbose:
                print(f"Capacidad suficiente (max={max_possible_quantity} >= L={l_bound}) con {len(selected_aisle_indices)} pasillos")
            
            # Resolver con Dinkelbach usando el max_lambda actual
            x_sol, y_sol, ratio = solve_dinkelbach_with_max_lambda(
                orders, selected_aisles, l_bound, r_bound, 
                selected_aisle_indices, k, max_lambda, verbose=True
            )
            
            if x_sol is not None:
                if verbose:
                    print(f"Ratio: {ratio:.4f}")
                
                # Actualizar mejor de la fase
                if ratio > phase_best_ratio:
                    phase_best_ratio = ratio
                    phase_best_x = x_sol
                    phase_best_y = y_sol
                
                # Actualizar mejor global
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_x = x_sol
                    best_y = y_sol
                    if verbose:
                        print(f"    *** NUEVA MEJOR SOLUCIÓN GLOBAL: {ratio:.4f} ***")
                
                # ACTUALIZAR MAX_LAMBDA GLOBALMENTE
                if ratio > max_lambda:
                    max_lambda = ratio
                    if verbose:
                        print(f"    *** ACTUALIZANDO λ_global a {max_lambda:.6f} ***")
            else:
                if verbose:
                    print("No factible")
        
        # Resumen de la fase
        phase_time = time.time() - phase_start_time
        if verbose:
            print(f"\nFASE {phase} COMPLETADA:")
            print(f"  Mejor ratio de la fase: {phase_best_ratio:.4f}")
            print(f"  Mejor ratio global: {best_ratio:.4f}")
            print(f"  Tiempo de la fase: {phase_time:.2f}s")
            print(f"  λ_global actual: {max_lambda:.6f}")
        
        # Preparar siguiente fase
        current_aisles = min((int)(current_aisles * 1.16), k)  # Duplicar pasillos
        current_trials = max(current_trials - 1, 1)  # Reducir trials (mínimo 1)
        phase += 1
        
        # Si ya usamos todos los pasillos, terminar
        if aisles_to_use >= k:
            if verbose:
                print(f"\n*** COMPLETADO: Ya se usaron todos los {k} pasillos disponibles ***")
            break
    
    total_time = time.time() - start_time
    if verbose:
        print(f"\n" + "="*70)
        print(f"BÚSQUEDA PROGRESIVA COMPLETADA")
        print(f"Tiempo total: {total_time:.2f}s ({total_time/60:.1f} minutos)")
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
    
    # Ejecutar solver con búsqueda progresiva
    start_time = time.time()
    best_x_sol, best_y_sol, best_ratio = solve_with_progressive_search(
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
    
    # Verificar restricciones de capacidad (opcional, para debug)
    # verify_capacity_constraints(orders, aisles, selected_orders, selected_aisles)

if __name__ == "__main__":
    main() 