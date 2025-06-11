import cplex
import sys
import logging
from dinkelbach_solver import read_instance

def debug_objective():
    """Debug objective function construction"""
    
    # Read instance
    orders, aisles, wave_size_lb, wave_size_ub = read_instance('instance_0003.txt')
    
    # Get all items
    items = set()
    for order in orders:
        items.update(order.keys())
    
    print(f"Debugging objective function construction:")
    print(f"Orders: {len(orders)}")
    print(f"Items: {len(items)}")
    
    # Manual calculation of objective coefficients
    manual_coeffs = [0] * len(orders)
    
    # Method 1: My current approach (sum by item)
    for k in items:
        for i in range(len(orders)):
            if k in orders[i]:
                manual_coeffs[i] += orders[i][k]
    
    # Method 2: Direct sum per order
    direct_coeffs = [sum(order.values()) for order in orders]
    
    print(f"\nComparing objective coefficients:")
    print(f"Method 1 (sum by item): {manual_coeffs[:5]}...")
    print(f"Method 2 (direct sum):  {direct_coeffs[:5]}...")
    print(f"Are they equal? {manual_coeffs == direct_coeffs}")
    
    if manual_coeffs != direct_coeffs:
        print("ERROR: Methods give different results!")
        for i in range(min(10, len(orders))):
            if manual_coeffs[i] != direct_coeffs[i]:
                print(f"  Order {i}: Method1={manual_coeffs[i]}, Method2={direct_coeffs[i]}")
                print(f"    Order content: {orders[i]}")
    
    # Test with a simple example
    print(f"\nTesting with first few orders:")
    for i in range(min(3, len(orders))):
        print(f"Order {i}: {orders[i]}")
        print(f"  Direct sum: {sum(orders[i].values())}")
        print(f"  Item-by-item sum: {sum(orders[i][k] for k in orders[i])}")

if __name__ == "__main__":
    debug_objective() 