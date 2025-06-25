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
        for j in range(1, len(parts), 2):
            if j + 1 < len(parts):
                item_id = int(parts[j])
                quantity = int(parts[j + 1])
                items[item_id] = quantity
        orders.append({'id': order_id, 'items': items})
    
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

def test_simple_dinkelbach(orders, aisles, l_bound, r_bound, selected_orders):
    """
    Simple test of Dinkelbach with selected orders
    """
    print(f"  Testing with {len(selected_orders)} selected orders")
    
    # Verificar capacidad total disponible
    total_capacity = {}
    for aisle in aisles:
        for item_id, capacity in aisle['items'].items():
            total_capacity[item_id] = total_capacity.get(item_id, 0) + capacity
    
    print(f"  Total capacity: {total_capacity}")
    
    # Calcular demanda total de los pedidos seleccionados
    total_demand = {}
    for order_idx in selected_orders:
        order = orders[order_idx]
        for item_id, quantity in order['items'].items():
            total_demand[item_id] = total_demand.get(item_id, 0) + quantity
    
    print(f"  Total demand: {total_demand}")
    
    # Verificar si es factible
    for item_id, demand in total_demand.items():
        if total_capacity.get(item_id, 0) < demand:
            print(f"  INFEASIBLE: Item {item_id} demand={demand} > capacity={total_capacity.get(item_id, 0)}")
            return None, None, 0
    
    print(f"  Feasibility check passed!")
    
    # Crear modelo simple
    mdl = Model(name="test_dinkelbach")
    mdl.set_time_limit(30)  # 30 segundos máximo
    
    # Variables de decisión
    n_selected = len(selected_orders)
    k_aisles = len(aisles)
    
    print(f"  Creating {n_selected} order variables and {k_aisles} aisle variables")
    
    # x_vars: si el pedido i es seleccionado (solo para pedidos seleccionados)
    x_vars = mdl.binary_var_list(n_selected, name="x")
    
    # y_vars: si el pasillo j es abierto
    y_vars = mdl.binary_var_list(k_aisles, name="y")
    
    # Función objetivo simple: maximizar pedidos
    total_orders_expr = mdl.sum(x_vars[i] for i in range(n_selected))
    total_aisles_expr = mdl.sum(y_vars[j] for j in range(k_aisles))
    
    mdl.maximize(total_orders_expr)
    
    print(f"  Adding constraints...")
    
    # Restricciones de rango
    mdl.add_constraint(total_orders_expr >= l_bound)
    mdl.add_constraint(total_orders_expr <= r_bound)
    
    # Al menos un pasillo debe estar abierto
    mdl.add_constraint(total_aisles_expr >= 1)
    
    # Restricciones de capacidad por ítem
    all_items = set()
    for order_idx in selected_orders:
        order = orders[order_idx]
        all_items.update(order['items'].keys())
    for aisle in aisles:
        all_items.update(aisle['items'].keys())
    
    print(f"  Adding capacity constraints for {len(all_items)} items")
    
    for item_id in all_items:
        # Demanda total del ítem
        demand_expr = mdl.sum(
            orders[selected_orders[i]]['items'].get(item_id, 0) * x_vars[i] 
            for i in range(n_selected)
        )
        
        # Capacidad total del ítem
        capacity_expr = mdl.sum(
            aisles[j]['items'].get(item_id, 0) * y_vars[j] 
            for j in range(k_aisles)
        )
        
        mdl.add_constraint(demand_expr <= capacity_expr)
    
    print(f"  Solving model...")
    
    # Resolver
    solution = mdl.solve()
    
    if not solution:
        print(f"  No solution found")
        return None, None, 0
    
    # Extraer solución
    x_sol = [int(solution.get_value(x_vars[i])) for i in range(n_selected)]
    y_sol = [int(solution.get_value(y_vars[j])) for j in range(k_aisles)]
    
    num_orders = sum(x_sol)
    num_aisles = sum(y_sol)
    
    ratio = num_orders / num_aisles if num_aisles > 0 else 0
    
    print(f"  Solution found: {num_orders} orders, {num_aisles} aisles, ratio={ratio:.4f}")
    
    return x_sol, y_sol, ratio

def main():
    if len(sys.argv) != 2:
        print("Uso: python debug_orders.py <archivo_entrada>")
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
        
        # Test con diferentes números de pedidos
        test_sizes = [2, min(5, n), min(10, n), n]
        
        for test_size in test_sizes:
            if test_size > n:
                continue
                
            print(f"=== TESTING WITH {test_size} ORDERS ===")
            
            # Seleccionar pedidos aleatoriamente
            if test_size >= n:
                selected_orders = list(range(n))
            else:
                selected_orders = sorted(random.sample(range(n), test_size))
            
            print(f"Selected orders: {[orders[i]['id'] for i in selected_orders]}")
            
            start_time = time.time()
            x_sol, y_sol, ratio = test_simple_dinkelbach(orders, aisles, l, r, selected_orders)
            end_time = time.time()
            
            print(f"Time: {end_time - start_time:.2f}s")
            
            if x_sol is not None:
                selected_aisle_ids = [aisles[j]['id'] for j in range(len(y_sol)) if y_sol[j] == 1]
                print(f"Selected aisles: {selected_aisle_ids}")
            
            print()
        
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {filename}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 