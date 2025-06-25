#!/usr/bin/env python3

import sys

def parse_input(lines):
    """Parse the input data from lines."""
    print("Parsing input...")
    n, m, k = map(int, lines[0].split())
    print(f"n={n}, m={m}, k={k}")
    
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
        print(f"Order {order_id}: {items}")
    
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
        print(f"Aisle {aisle_id}: {items}")
    
    l, r = map(int, lines[n + 1 + k].split())
    print(f"Range: [{l}, {r}]")
    
    return n, m, k, orders, aisles, l, r

def main():
    if len(sys.argv) != 2:
        print("Uso: python simple_debug.py <archivo_entrada>")
        sys.exit(1)
    
    filename = sys.argv[1]
    print(f"Reading file: {filename}")
    
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        print(f"File has {len(lines)} lines")
        
        n, m, k, orders, aisles, l, r = parse_input(lines)
        
        print(f"\nSummary:")
        print(f"Orders: {len(orders)}")
        print(f"Aisles: {len(aisles)}")
        print(f"Range: [{l}, {r}]")
        
        # Test capacity calculation
        print(f"\nTesting capacity calculation...")
        total_capacity = {}
        for aisle in aisles:
            for item_id, capacity in aisle['items'].items():
                total_capacity[item_id] = total_capacity.get(item_id, 0) + capacity
        
        print(f"Total capacity: {total_capacity}")
        
        # Test demand calculation for first few orders
        print(f"\nTesting demand calculation...")
        selected_orders = [0, 1] if len(orders) >= 2 else [0]
        total_demand = {}
        for order_idx in selected_orders:
            order = orders[order_idx]
            for item_id, quantity in order['items'].items():
                total_demand[item_id] = total_demand.get(item_id, 0) + quantity
        
        print(f"Demand for orders {selected_orders}: {total_demand}")
        
        # Check feasibility
        feasible = True
        for item_id, demand in total_demand.items():
            if total_capacity.get(item_id, 0) < demand:
                print(f"INFEASIBLE: Item {item_id} demand={demand} > capacity={total_capacity.get(item_id, 0)}")
                feasible = False
        
        if feasible:
            print("Basic feasibility check passed!")
        
        print("Debug completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 