#!/usr/bin/env python3

import sys
import json
import time
import random
from docplex.mp.model import Model

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

def debug_aisle_71(orders, aisles, l_bound, r_bound):
    """Debug específico del pasillo 71"""
    aisle_71 = aisles[71]
    print(f"Pasillo 71 tiene {len(aisle_71['items'])} ítems:")
    print(f"Items del pasillo 71: {aisle_71['items']}")
    
    # Ver qué pedidos puede satisfacer solo el pasillo 71
    satisfiable_orders = []
    total_quantity = 0
    
    for i, order in enumerate(orders):
        can_satisfy = True
        for item_id, demand in order['items'].items():
            available = aisle_71['items'].get(item_id, 0)
            if available < demand:
                can_satisfy = False
                break
        
        if can_satisfy:
            satisfiable_orders.append(i)
            total_quantity += order['total_quantity']
            print(f"  Pedido {i}: {order['items']} -> cantidad {order['total_quantity']}")
    
    print(f"\nPedidos que puede satisfacer el pasillo 71 solo: {satisfiable_orders}")
    print(f"Cantidad total: {total_quantity}")
    print(f"Ratio si usa solo pasillo 71: {total_quantity}/1 = {total_quantity}")
    
    # Verificar si esto cumple con L y R
    print(f"¿Cumple L <= cantidad <= R? {l_bound} <= {total_quantity} <= {r_bound}")
    
    return satisfiable_orders, total_quantity

def debug_optimal_solution(orders, aisles, l_bound, r_bound):
    """Debug de la solución óptima conocida [43, 71, 97]"""
    optimal_aisles = [43, 71, 97]
    print(f"\n=== SOLUCIÓN ÓPTIMA CONOCIDA ===")
    print(f"Pasillos: {optimal_aisles}")
    
    # Calcular capacidad total
    total_capacity_per_item = {}
    for aisle_idx in optimal_aisles:
        aisle = aisles[aisle_idx]
        for item_id, capacity in aisle['items'].items():
            total_capacity_per_item[item_id] = total_capacity_per_item.get(item_id, 0) + capacity
    
    print(f"Capacidad total por ítem: {len(total_capacity_per_item)} ítems diferentes")
    
    # Ver qué pedidos puede satisfacer
    satisfiable_orders = []
    total_quantity = 0
    
    for i, order in enumerate(orders):
        can_satisfy = True
        for item_id, demand in order['items'].items():
            available = total_capacity_per_item.get(item_id, 0)
            if available < demand:
                can_satisfy = False
                break
        
        if can_satisfy:
            satisfiable_orders.append(i)
            total_quantity += order['total_quantity']
    
    print(f"Pedidos satisfacibles: {satisfiable_orders}")
    print(f"Cantidad total: {total_quantity}")
    print(f"Ratio: {total_quantity}/3 = {total_quantity/3}")

def main():
    filename = sys.argv[1] if len(sys.argv) > 1 else "datasets/a/instance_0001.txt"
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    n, k, orders, aisles, l_bound, r_bound = parse_input(lines)
    print(f"n={n}, k={k}, L={l_bound}, R={r_bound}")
    
    # Debug pasillo 71
    satisfiable_71, total_71 = debug_aisle_71(orders, aisles, l_bound, r_bound)
    
    # Debug solución óptima
    debug_optimal_solution(orders, aisles, l_bound, r_bound)

if __name__ == "__main__":
    main() 