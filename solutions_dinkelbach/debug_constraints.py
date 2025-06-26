import cplex
import sys
import logging
from dinkelbach_solver import read_instance

def debug_constraints():
    """Debug how constraints are being built"""
    
    # Read instance
    orders, aisles, wave_size_lb, wave_size_ub = read_instance('instance_0003.txt')
    
    # Get all items
    items = set()
    for order in orders:
        items.update(order.keys())
    
    n_orders = len(orders)
    n_aisles = len(aisles)
    
    print(f"Debugging constraint construction:")
    print(f"  Orders: {n_orders} (indices 0 to {n_orders-1})")
    print(f"  Aisles: {n_aisles} (indices {n_orders} to {n_orders + n_aisles - 1})")
    
    # Check Item 7 specifically
    k = 7
    print(f"\nItem {k} constraint construction:")
    
    # Left side: orders that need Item 7
    order_indices = []
    order_coeffs = []
    for i in range(n_orders):
        if k in orders[i]:
            order_indices.append(i)
            order_coeffs.append(orders[i][k])
            print(f"  Order {i} needs {orders[i][k]} units of item {k}")
    
    # Right side: aisles that have Item 7
    aisle_indices = []
    aisle_coeffs = []
    for j in range(n_aisles):
        if k in aisles[j]:
            aisle_indices.append(n_orders + j)
            aisle_coeffs.append(-aisles[j][k])  # Negative!
            print(f"  Aisle {j} (var index {n_orders + j}) has {aisles[j][k]} units of item {k}")
    
    print(f"\nConstraint for item {k}:")
    print(f"  Variable indices: {order_indices + aisle_indices}")
    print(f"  Coefficients: {order_coeffs + aisle_coeffs}")
    print(f"  Sense: <= 0")
    
    # This means: sum(order_coeffs * x_vars) + sum(aisle_coeffs * y_vars) <= 0
    # Which is: sum(order_coeffs * x_vars) - sum(aisles_capacity * y_vars) <= 0
    # Which is: sum(order_coeffs * x_vars) <= sum(aisles_capacity * y_vars)
    
    total_needed = sum(order_coeffs)
    total_available = sum(-coeff for coeff in aisle_coeffs)  # Remove negative sign
    
    print(f"\nInterpretation:")
    print(f"  Total needed: {total_needed}")
    print(f"  Total available: {total_available}")
    print(f"  Feasible? {total_needed <= total_available}")
    
    if total_needed > total_available:
        print(f"  ERROR: Item {k} is infeasible! Need {total_needed}, have {total_available}")

if __name__ == '__main__':
    debug_constraints() 