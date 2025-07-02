from docplex.mp.model import Model


def solve_dinkelbach(orders, selected_aisles, selected_aisle_indices, l_bound, r_bound, k, initial_lambda):
    n = len(orders)
    m = len(selected_aisles)

    # Obtener todos los items relevantes
    relevant_items = set()
    for aisle in selected_aisles:
        relevant_items.update(aisle['items'].keys())

    # Filtrar pedidos que solo contengan ítems relevantes
    valid_orders = []
    valid_order_indices = []

    for i, order in enumerate(orders):
        # Verificar si todos los ítems del pedido están en ítems relevantes
        order_items = set(order['items'].keys())
        if order_items.issubset(relevant_items):
            valid_orders.append(order)
            valid_order_indices.append(i)

    n_valid = len(valid_orders)
    print(f"    Pedidos válidos: {n_valid}/{n} (eliminados {n - n_valid} pedidos)")

    # Si no hay pedidos válidos, retornar None
    if n_valid == 0:
        print("    No hay pedidos válidos para resolver")
        return None, None, None

    # USAR EL MAX_LAMBDA INICIAL PASADO COMO PARÁMETRO
    max_lambda = initial_lambda

    epsilon = 1e-6
    iteration = 1
    max_iter = 100

    while iteration < max_iter:

        print(f"    Iter {iteration}: λ_max_usado={max_lambda:.6f}", end="")

        mdl = Model(name=f'dinkelbach_iter_{iteration}')

        # Configuración de multithreading y rendimiento de CPLEX
        mdl.context.cplex_parameters.threads = 16  # Número de threads
        mdl.context.cplex_parameters.mip.display = 0  # Silenciar CPLEX
        mdl.context.cplex_parameters.mip.tolerances.mipgap = 0.03  # Gap de tolerancia 1%

        try:
            mdl.context.cplex_parameters.parallel = 1  # Modo paralelo determinístico
            mdl.context.cplex_parameters.mip.strategy.heuristicfreq = 20  # Heurísticas más frecuentes
            mdl.context.cplex_parameters.emphasis.mip = 1  # Énfasis en factibilidad
        except:
            pass  # Ignorar si algún parámetro no está disponible

        # Variables de decisión (solo para pedidos válidos)
        x_vars = mdl.binary_var_list(n_valid, name='x')
        y_vars = mdl.binary_var_list(m, name='y')

        # Expresiones para objetivo
        total_quantity_expr = mdl.sum(valid_orders[i]['total_quantity'] * x_vars[i] for i in range(n_valid))
        total_aisles_expr = mdl.sum(y_vars[j] for j in range(m))  # Variables, no constante

        # Restricción de rango [L, R]
        mdl.add_range(l_bound, total_quantity_expr, r_bound)

        # Forzar que al menos un pasillo esté abierto
        mdl.add_constraint(total_aisles_expr >= 1, ctname="force_aisle_opening")

        # Restricciones de capacidad para TODOS los items relevantes
        constraints_added = 0
        for item_id in relevant_items:
            demand = mdl.sum(valid_orders[i]['items'].get(item_id, 0) * x_vars[i] for i in range(n_valid))
            capacity = mdl.sum(selected_aisles[j]['items'].get(item_id, 0) * y_vars[j] for j in range(m))
            mdl.add_constraint(demand <= capacity, ctname=f"capacity_item_{item_id}")
            constraints_added += 1

        if iteration == 0:
            print(f" ({constraints_added} restricciones de capacidad)", end="")

        # Función objetivo de Dinkelbach: max(f(x,y) - lambda * g(x,y))
        mdl.set_objective('max', total_quantity_expr - max_lambda * total_aisles_expr)

        # Resolver
        solution = mdl.solve()

        if not solution:
            print(f" → No factible")
            return None, None, None

            # Obtener valores de variables para pedidos válidos
        x_sol_valid = [x_vars[i].solution_value for i in range(n_valid)]
        y_sol_subset = [y_vars[j].solution_value for j in range(m)]

        # Expandir x_sol al tamaño original n con ceros para pedidos eliminados
        x_sol = [0.0] * n
        for i, original_idx in enumerate(valid_order_indices):
            x_sol[original_idx] = x_sol_valid[i]

        # Expandir y_sol al tamaño completo k
        y_sol = [0.0] * k
        for i, aisle_idx in enumerate(selected_aisle_indices):
            y_sol[aisle_idx] = y_sol_subset[i]

        # Calcular numerador y denominador usando pedidos válidos
        numerator = sum(valid_orders[i]['total_quantity'] * x_sol_valid[i] for i in range(n_valid))
        denominator = sum(y_sol_subset[j] for j in range(m))

        if numerator < l_bound or numerator > r_bound:
            print("ERROR: invalid solution")

        if denominator > 0:
            lambda_val = numerator / denominator
        else:
            lambda_val = 0.0

        obj_val = numerator - max_lambda * denominator

        print(
            f" → λ_calculado={lambda_val:.6f}, f(x,y)={numerator:.2f}, g(x,y)={denominator} ({int(denominator)} pasillos), ratio={lambda_val:.6f}, obj={obj_val:.6f}")

        # CONDICIÓN DE TERMINACIÓN TEMPRANA: si el nuevo lambda <= max_lambda usado
        if lambda_val <= max_lambda + epsilon:
            print(
                f"    *** TERMINANDO: λ_calculado ({lambda_val:.6f}) <= λ_max_usado ({max_lambda:.6f}) - ratio {lambda_val:.6f} ***")
            return x_sol, y_sol, lambda_val

        if lambda_val > max_lambda:
            max_lambda = lambda_val
            print(f"    *** Actualizando λ_max a {max_lambda:.6f} ***")

        # Verificar convergencia estándar
        if abs(lambda_val - max_lambda) < epsilon:
            print(f"    *** CONVERGENCIA: Diferencia {abs(lambda_val - max_lambda):.8f} < {epsilon} ***")
            return x_sol, y_sol, lambda_val

        iteration += 1

    print(f"    *** NO CONVERGIÓ después de {max_iter} iteraciones ***")
    return x_sol, y_sol, lambda_val if 'lambda_val' in locals() else 0.0
