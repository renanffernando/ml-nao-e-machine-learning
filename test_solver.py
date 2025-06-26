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

def test_capacity_calculation(orders, aisles, l_bound, selected_indices):
    """Prueba simple del cálculo de capacidad"""
    print(f"Testing capacity with aisles: {selected_indices}")
    
    # Calcular capacidad total por item
    total_capacity_per_item = {}
    for idx in selected_indices:
        aisle = aisles[idx]
        for item_id, capacity in aisle['items'].items():
            total_capacity_per_item[item_id] = total_capacity_per_item.get(item_id, 0) + capacity
    
    print(f"Total capacity per item: {total_capacity_per_item}")
    
    # Verificar qué pedidos se pueden satisfacer
    max_possible_quantity = 0
    satisfiable_orders = []
    
    for i, order in enumerate(orders):
        can_satisfy = True
        for item_id, demand in order['items'].items():
            available = total_capacity_per_item.get(item_id, 0)
            if available < demand:
                can_satisfy = False
                break
        
        if can_satisfy:
            max_possible_quantity += order['total_quantity']
            satisfiable_orders.append(i)
    
    print(f"Satisfiable orders: {satisfiable_orders}")
    print(f"Max possible quantity: {max_possible_quantity}")
    print(f"L bound: {l_bound}")
    print(f"Can satisfy L? {max_possible_quantity >= l_bound}")
    
    return max_possible_quantity >= l_bound, max_possible_quantity

def simple_solver_test(orders, aisles, l_bound, r_bound):
    """Solver muy simple para probar"""
    n = len(orders)
    k = len(aisles)
    
    print(f"Orders: {n}, Aisles: {k}, L: {l_bound}, R: {r_bound}")
    
    # Probar con 3 pasillos aleatorios
    selected_indices = random.sample(range(k), min(3, k))
    print(f"\nTesting with 3 random aisles: {selected_indices}")
    
    can_satisfy, max_qty = test_capacity_calculation(orders, aisles, l_bound, selected_indices)
    
    if not can_satisfy:
        print("Cannot satisfy L with 3 aisles, trying expansion...")
        
        # Intentar agregar más pasillos
        remaining = [i for i in range(k) if i not in selected_indices]
        max_to_add = min(3, len(remaining))  # Agregar máximo 3 más
        
        for _ in range(max_to_add):
            if remaining and max_qty < l_bound:
                new_aisle = random.choice(remaining)
                selected_indices.append(new_aisle)
                remaining.remove(new_aisle)
                print(f"Added aisle {new_aisle}, new selection: {selected_indices}")
                
                can_satisfy, max_qty = test_capacity_calculation(orders, aisles, l_bound, selected_indices)
                if can_satisfy:
                    break
    
    if can_satisfy:
        print(f"\nFinal selection can satisfy L: {selected_indices}")
        return selected_indices
    else:
        print(f"\nCannot satisfy L even with expansion")
        return None

def main():
    print("Simple solver test starting...")
    
    # Leer datos del archivo
    if len(sys.argv) < 2:
        print("Usage: python test_solver.py <input_file>")
        return
        
    filename = sys.argv[1]
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"File not found: {filename}")
        return
    
    if not lines:
        print("No input data")
        return
    
    n, k, orders, aisles, l_bound, r_bound = parse_input(lines)
    print(f"Parsed: n={n}, k={k}, L={l_bound}, R={r_bound}")
    
    # Mostrar algunos datos
    print(f"\nFirst order: {orders[0]}")
    print(f"First aisle: {aisles[0]}")
    
    # Ejecutar test simple
    result = simple_solver_test(orders, aisles, l_bound, r_bound)
    
    if result:
        print(f"\nSuccess! Selected aisles: {result}")
    else:
        print(f"\nFailed to find solution")

if __name__ == "__main__":
    main() 