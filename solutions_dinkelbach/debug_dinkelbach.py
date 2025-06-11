import cplex
import sys
import logging
from dinkelbach_solver import read_instance

def debug_dinkelbach_vs_direct():
    """Compare what Dinkelbach is doing vs the correct direct solution"""
    
    # Read instance
    orders, aisles, wave_size_lb, wave_size_ub = read_instance('instance_0003.txt')
    items = set()
    for order in orders:
        items.update(order.keys())
    
    print("=== DEBUGGING DINKELBACH vs DIRECT ===")
    
    # From direct solver, we know the optimal solution uses:
    optimal_orders = [4, 9, 10, 19, 22, 29, 57, 63, 72]
    optimal_aisles = [7, 16, 89, 102]
    
    print(f"Optimal solution from direct solver:")
    print(f"  Selected orders: {optimal_orders}")
    print(f"  Used aisles: {optimal_aisles}")
    
    # Check what items are needed by these orders
    needed_items = {}
    for i in optimal_orders:
        order = orders[i]
        print(f"  Order {i}: {order}")
        for item_id, quantity in order.items():
            if item_id not in needed_items:
                needed_items[item_id] = 0
            needed_items[item_id] += quantity
    
    print(f"\nTotal items needed: {len(needed_items)}")
    print(f"First 10 needed items: {dict(list(needed_items.items())[:10])}")
    
    # Check if these 4 aisles can cover all needed items
    print(f"\nChecking coverage with optimal aisles {optimal_aisles}:")
    available_items = {}
    for j in optimal_aisles:
        aisle = aisles[j]
        print(f"  Aisle {j}: {aisle}")
        for item_id, capacity in aisle.items():
            if item_id not in available_items:
                available_items[item_id] = 0
            available_items[item_id] += capacity
    
    print(f"\nCoverage check:")
    all_covered = True
    for item_id, needed in needed_items.items():
        available = available_items.get(item_id, 0)
        covered = available >= needed
        if not covered:
            print(f"  Item {item_id}: NEED {needed}, HAVE {available} - NOT COVERED")
            all_covered = False
        else:
            print(f"  Item {item_id}: NEED {needed}, HAVE {available} - OK")
    
    print(f"\nAll items covered: {all_covered}")
    
    # Now let's see what happens in Dinkelbach with q=0
    print(f"\n=== TESTING DINKELBACH WITH q=0 ===")
    
    prob = cplex.Cplex()
    prob.set_log_stream(None)
    prob.set_error_stream(None)
    prob.set_warning_stream(None)
    prob.set_results_stream(None)
    
    prob.objective.set_sense(prob.objective.sense.maximize)
    
    n_orders = len(orders)
    n_aisles = len(aisles)
    
    # Variables
    x_names = [f'x_{i}' for i in range(n_orders)]
    x = prob.variables.add(
        obj=[sum(order.values()) for order in orders],
        lb=[0] * n_orders,
        ub=[1] * n_orders,
        types=['B'] * n_orders,
        names=x_names
    )
    
    y_names = [f'y_{j}' for j in range(n_aisles)]
    y = prob.variables.add(
        obj=[0] * n_aisles,  # q=0, so no penalty for aisles
        lb=[0] * n_aisles,
        ub=[1] * n_aisles,
        types=['B'] * n_aisles,
        names=y_names
    )
    
    # Wave size constraints
    N_expr = list(range(n_orders))
    N_coef = [sum(order.values()) for order in orders]
    
    prob.linear_constraints.add(
        lin_expr=[[N_expr, N_coef]],
        senses=['G'],
        rhs=[wave_size_lb],
        names=['wave_size_lb']
    )
    
    prob.linear_constraints.add(
        lin_expr=[[N_expr, N_coef]],
        senses=['L'],
        rhs=[wave_size_ub],
        names=['wave_size_ub']
    )
    
    # Item coverage constraints - THIS IS WHERE THE BUG MIGHT BE
    constraint_count = 0
    for k in items:
        order_indices = []
        order_coeffs = []
        for i in range(n_orders):
            if k in orders[i]:
                order_indices.append(i)
                order_coeffs.append(orders[i][k])
        
        aisle_indices = []
        aisle_coeffs = []
        for j in range(n_aisles):
            if k in aisles[j]:
                aisle_indices.append(n_orders + j)
                aisle_coeffs.append(-aisles[j][k])
        
        if order_indices:
            prob.linear_constraints.add(
                lin_expr=[[
                    order_indices + aisle_indices,
                    order_coeffs + aisle_coeffs
                ]],
                senses=['L'],
                rhs=[0],
                names=[f'item_{k}_coverage']
            )
            constraint_count += 1
    
    print(f"Added {constraint_count} coverage constraints")
    
    # Solve
    prob.solve()
    
    if prob.solution.get_status() == 101:
        obj_value = prob.solution.get_objective_value()
        x_sol = [prob.solution.get_values(f'x_{i}') for i in range(n_orders)]
        y_sol = [prob.solution.get_values(f'y_{j}') for j in range(n_aisles)]
        
        selected_orders = [i for i, xi in enumerate(x_sol) if xi > 0.5]
        used_aisles = [j for j, yj in enumerate(y_sol) if yj > 0.5]
        
        N = sum(sum(order.values()) * xi for i, (order, xi) in enumerate(zip(orders, x_sol)) if xi > 0.5)
        D = len(used_aisles)
        
        print(f"Dinkelbach with q=0 result:")
        print(f"  N = {N}")
        print(f"  D = {D}")
        print(f"  Ratio = {N/D if D > 0 else 0}")
        print(f"  Selected orders: {selected_orders}")
        print(f"  Used aisles: {used_aisles[:10]}...")  # Show first 10
        
        if D > 20:
            print(f"ERROR: Using too many aisles ({D})! Should be around 4.")
    else:
        print("No solution found")

if __name__ == "__main__":
    debug_dinkelbach_vs_direct() 