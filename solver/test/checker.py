def verify_solution(x_sol, y_sol, opt_val, orders, aisles, l, r):
    """
    Verifica que la solución cumpla con todas las restricciones del modelo de optimización fraccional.
    
    Args:
        x_sol: Lista de valores binarios para variables x (pedidos seleccionados)
        y_sol: Lista de valores binarios para variables y (pasillos seleccionados) 
        opt_val: Valor óptimo esperado (ratio cantidad_total / num_pasillos)
        orders: Lista de pedidos con estructura {'total_quantity': int, 'items': {item_id: quantity}}
        aisles: Lista de pasillos con estructura {'items': {item_id: capacity}}
        l: Límite inferior del rango
        r: Límite superior del rango
        
    Returns:
        tuple: (es_valida, mensaje_error, estadisticas)
    """

    n = len(orders)
    k = len(aisles)

    # Verificar dimensiones
    if len(x_sol) != n:
        return False, f"Dimensión incorrecta de x_sol: esperado {n}, obtenido {len(x_sol)}", {}

    if len(y_sol) != k:
        return False, f"Dimensión incorrecta de y_sol: esperado {k}, obtenido {len(y_sol)}", {}

    # Convertir a enteros (tolerancia para valores binarios)
    tolerance = 1e-6
    x_binary = []
    y_binary = []

    for i in range(n):
        if abs(x_sol[i] - 0) < tolerance:
            x_binary.append(0)
        elif abs(x_sol[i] - 1) < tolerance:
            x_binary.append(1)
        else:
            return False, f"Variable x[{i}] = {x_sol[i]} no es binaria", {}

    for j in range(k):
        if abs(y_sol[j] - 0) < tolerance:
            y_binary.append(0)
        elif abs(y_sol[j] - 1) < tolerance:
            y_binary.append(1)
        else:
            return False, f"Variable y[{j}] = {y_sol[j]} no es binaria", {}

    # Calcular pedidos y pasillos seleccionados
    selected_orders = [i for i in range(n) if x_binary[i] == 1]
    selected_aisles = [j for j in range(k) if y_binary[j] == 1]

    # 1. Verificar que al menos un pasillo esté seleccionado
    if len(selected_aisles) == 0:
        return False, "No hay pasillos seleccionados (violación de restricción de al menos un pasillo)", {}

    # 2. Calcular cantidad total
    total_quantity = sum(orders[i]['total_quantity'] for i in selected_orders)

    # 3. Verificar restricción de rango [L, R]
    if total_quantity < l:
        return False, f"Cantidad total {total_quantity} < límite inferior {l}", {}

    if total_quantity > r:
        return False, f"Cantidad total {total_quantity} > límite superior {r}", {}

    # 4. Verificar restricciones de capacidad
    # Obtener todos los ítems relevantes
    all_items = set()
    for order in orders:
        all_items.update(order['items'].keys())
    for aisle in aisles:
        all_items.update(aisle['items'].keys())

    for item_id in all_items:
        # Calcular demanda total para este ítem
        total_demand = sum(orders[i]['items'].get(item_id, 0) for i in selected_orders)

        # Calcular capacidad total disponible para este ítem
        total_capacity = sum(aisles[j]['items'].get(item_id, 0) for j in selected_aisles)

        # Verificar que la demanda no exceda la capacidad
        if total_demand > total_capacity:
            return False, f"Ítem {item_id}: demanda {total_demand} > capacidad {total_capacity}", {}

    # 5. Verificar valor objetivo
    num_aisles = len(selected_aisles)
    calculated_ratio = total_quantity / num_aisles

    # Tolerancia para comparación de flotantes
    ratio_tolerance = 1e-4
    if abs(calculated_ratio - opt_val) > ratio_tolerance:
        return False, f"Valor objetivo incorrecto: calculado {calculated_ratio:.6f}, esperado {opt_val:.6f}", {}

    # Generar estadísticas
    stats = {
        'selected_orders': selected_orders,
        'selected_aisles': selected_aisles,
        'num_selected_orders': len(selected_orders),
        'num_selected_aisles': num_aisles,
        'total_quantity': total_quantity,
        'calculated_ratio': calculated_ratio,
        'range_compliance': f"[{l}, {r}]",
        'capacity_items_checked': len(all_items)
    }

    return True, "Solución válida - todas las restricciones cumplidas", stats
