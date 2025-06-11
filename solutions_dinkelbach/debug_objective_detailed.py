import cplex
import sys
import logging
from dinkelbach_solver import read_instance

def debug_objective_detailed():
    """Debug objective function calculation in detail"""
    
    # Read instance
    orders, aisles, wave_size_lb, wave_size_ub = read_instance('instance_0003.txt')
    
    # Get all items
    items = set()
    for order in orders:
        items.update(order.keys())
    
    print(f"Debugging detailed objective calculation:")
    
    # Simulate the selected orders from the solution
    selected_orders = [3, 6, 10, 11, 12, 15, 26, 47, 51, 58, 60, 63, 65, 69, 71, 72, 79]
    
    print(f"Selected orders: {selected_orders}")
    print(f"Number of selected orders: {len(selected_orders)}")
    
    # Method 1: Calculate N as sum of units per unique item (my current approach)
    selected_items = {}
    for i in selected_orders:
        order = orders[i]
        print(f"Order {i}: {order}")
        for item_id, quantity in order.items():
            if item_id not in selected_items:
                selected_items[item_id] = 0
            selected_items[item_id] += quantity
    
    N_method1 = sum(selected_items.values())
    print(f"\nMethod 1 - N (sum of units per unique item): {N_method1}")
    print(f"Selected items: {len(selected_items)} items")
    print(f"First 10 items: {dict(list(selected_items.items())[:10])}")
    
    # Method 2: Calculate N as sum of all units in selected orders
    N_method2 = sum(sum(orders[i].values()) for i in selected_orders)
    print(f"\nMethod 2 - N (sum of all units in orders): {N_method2}")
    
    # Method 3: Calculate objective coefficients like in CPLEX
    obj_coeffs = [0] * len(orders)
    for k in items:
        for i in range(len(orders)):
            if k in orders[i]:
                obj_coeffs[i] += orders[i][k]
    
    N_method3 = sum(obj_coeffs[i] for i in selected_orders)
    print(f"\nMethod 3 - N (using CPLEX coefficients): {N_method3}")
    
    print(f"\nComparison:")
    print(f"Method 1 (unique items): {N_method1}")
    print(f"Method 2 (sum all units): {N_method2}")
    print(f"Method 3 (CPLEX way): {N_method3}")
    
    if N_method1 != N_method2:
        print(f"ERROR: Methods 1 and 2 should be equal but aren't!")
    if N_method2 != N_method3:
        print(f"ERROR: Methods 2 and 3 should be equal but aren't!")

if __name__ == "__main__":
    debug_objective_detailed() 