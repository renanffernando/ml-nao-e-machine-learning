from docplex.mp.model import Model

def solve_dinkelbach_2(orders, selected_aisles, selected_aisle_indices, l_bound, r_bound, k, initial_lambda, 
                      subset_orders=None, subset_aisles=None, warm_start_x=None, warm_start_y=None, model=None, all_aisles=None):
    """
    Versión optimizada con modelo FIJO reutilizable.
    - Si model=None: Crea modelo fijo con TODAS las variables (k pasillos, n pedidos)
    - Si model!=None: Reutiliza modelo, solo cambia bounds y warm start
    """
    n = len(orders)
    
    # Determinar qué pedidos pueden ser cubiertos por los pasillos seleccionados
    relevant_items = set()
    for aisle in selected_aisles:
        relevant_items.update(aisle['items'].keys())
    
    valid_order_indices = []
    for i, order in enumerate(orders):
        order_items = set(order['items'].keys())
        if order_items.issubset(relevant_items):
            valid_order_indices.append(i)
    
    print(f"    Pedidos válidos: {len(valid_order_indices)}/{n}")

    if len(valid_order_indices) == 0:
        print("    No hay pedidos válidos para resolver")
        return None, None, None, model
    
    max_lambda = initial_lambda
    epsilon = 1e-6
    iteration = 1
    max_iter = 100

    # CREAR MODELO FIJO (una sola vez)
    if model is None:
        print("    Creando modelo FIJO con todas las variables...")
        model = Model(name='dinkelbach_fixed_model')

        # Configuración de CPLEX
        model.context.cplex_parameters.threads = 16
        model.context.cplex_parameters.mip.display = 0
        model.context.cplex_parameters.mip.tolerances.mipgap = 0.03
        
        try:
            model.context.cplex_parameters.parallel = 1
            model.context.cplex_parameters.mip.strategy.heuristicfreq = 20
            model.context.cplex_parameters.emphasis.mip = 1
        except:
            pass

        # Variables para TODOS los pedidos y TODOS los pasillos
        model.x_vars = model.binary_var_list(n, name='x')
        model.y_vars = model.binary_var_list(k, name='y')
        
        # Restricción de rango [L, R] usando TODOS los pedidos
        model.total_quantity_expr = model.sum(orders[i]['total_quantity'] * model.x_vars[i] for i in range(n))
        model.add_range(l_bound, model.total_quantity_expr, r_bound)
        
        # Restricción: al menos un pasillo abierto
        model.total_aisles_expr = model.sum(model.y_vars[j] for j in range(k))
        model.add_constraint(model.total_aisles_expr >= 1, ctname="force_aisle_opening")
        
        # Restricciones de capacidad para TODOS los items usando TODOS los pasillos
        all_items = set()
        for aisle in all_aisles:
            all_items.update(aisle['items'].keys())
        
        for item_id in all_items:
            demand = model.sum(orders[i]['items'].get(item_id, 0) * model.x_vars[i] for i in range(n))
            capacity = model.sum(all_aisles[j]['items'].get(item_id, 0) * model.y_vars[j] for j in range(k))
            model.add_constraint(demand <= capacity, ctname=f"capacity_item_{item_id}")
        
        print(f"    Modelo fijo creado: {n} pedidos, {k} pasillos, {len(all_items)} items")

    while iteration < max_iter:
        print(f"    Iter {iteration}: λ_max_usado={max_lambda:.6f}", end="")
        
        # CONFIGURAR BOUNDS para el subset actual
        # Pasillos: libres si están en selected_aisle_indices, fijos a 0 si no
        for j in range(k):
            if j in selected_aisle_indices:
                model.y_vars[j].set_lb(0)
                model.y_vars[j].set_ub(1)
            else:
                model.y_vars[j].set_lb(0)
                model.y_vars[j].set_ub(0)
        
        # Pedidos: libres si son válidos, fijos a 0 si no
        for i in range(n):
            if i in valid_order_indices:
                model.x_vars[i].set_lb(0)
                model.x_vars[i].set_ub(1)
            else:
                model.x_vars[i].set_lb(0)
                model.x_vars[i].set_ub(0)
        
        # WARM START si está disponible
        if warm_start_x is not None and warm_start_y is not None:
            try:
                # Aplicar warm start
                for i in range(n):
                    if i < len(warm_start_x):
                        model.x_vars[i].set_start_value(warm_start_x[i])
                for j in range(k):
                    if j < len(warm_start_y):
                        model.y_vars[j].set_start_value(warm_start_y[j])
                print(" (warm start)", end="")
            except:
                pass
        
        # Función objetivo de Dinkelbach: max(f(x,y) - lambda * g(x,y))
        model.set_objective('max', model.total_quantity_expr - max_lambda * model.total_aisles_expr)
        
        # Resolver
        solution = model.solve()
        
        if not solution:
            print(f" → No factible")
            return None, None, None, model
        
        # Obtener solución completa (todos los n pedidos, todos los k pasillos)
        x_sol = [model.x_vars[i].solution_value for i in range(n)]
        y_sol = [model.y_vars[j].solution_value for j in range(k)]
        
        # Calcular numerador y denominador
        numerator = sum(orders[i]['total_quantity'] * x_sol[i] for i in range(n))
        denominator = sum(y_sol[j] for j in range(k))

        if denominator > 0:
            lambda_val = numerator / denominator
        else:
            lambda_val = 0.0
            
        obj_val = numerator - max_lambda * denominator
        
        print(f" → λ={lambda_val:.6f}, f={numerator:.2f}, g={denominator}, obj={obj_val:.6f}")
        
        # Condición de terminación
        if lambda_val <= max_lambda + epsilon:
            print(f"    *** TERMINANDO: λ_calculado ({lambda_val:.6f}) <= λ_max_usado ({max_lambda:.6f}) ***")
            return x_sol, y_sol, lambda_val, model
        
        if lambda_val > max_lambda:
            max_lambda = lambda_val
    
    
        iteration += 1

    print(f"    *** NO CONVERGIÓ después de {max_iter} iteraciones ***")
    return x_sol, y_sol, lambda_val if 'lambda_val' in locals() else 0.0, model 