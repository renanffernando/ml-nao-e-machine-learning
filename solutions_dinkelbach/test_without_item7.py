import cplex
import sys
import logging
from dinkelbach_solver import read_instance

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def test_without_item7():
    """Test the model excluding Order 50 (which contains Item 7)"""
    
    # Read instance
    orders, aisles, wave_size_lb, wave_size_ub = read_instance('instance_0003.txt')
    
    # Remove Order 50 (contains Item 7)
    filtered_orders = [order for i, order in enumerate(orders) if i != 50]
    
    # Get all items from filtered orders
    items = set()
    for order in filtered_orders:
        items.update(order.keys())
    
    print(f"Filtered instance info:")
    print(f"  Orders: {len(filtered_orders)} (removed order 50)")
    print(f"  Aisles: {len(aisles)}")
    print(f"  Items: {len(items)}")
    print(f"  Wave size bounds: [{wave_size_lb}, {wave_size_ub}]")
    
    # Simple optimization: maximize total units with minimal aisles
    prob = cplex.Cplex()
    prob.set_log_stream(None)
    prob.set_error_stream(None)
    prob.set_warning_stream(None)
    prob.set_results_stream(None)
    prob.objective.set_sense(prob.objective.sense.maximize)
    
    n_orders = len(filtered_orders)
    n_aisles = len(aisles)
    
    # x[i] = 1 if order i is selected
    x_names = [f'x_{i}' for i in range(n_orders)]
    x = prob.variables.add(
        obj=[sum(order.values()) for order in filtered_orders],  # Maximize total units
        lb=[0] * n_orders,
        ub=[1] * n_orders,
        types=['B'] * n_orders,
        names=x_names
    )
    
    # y[j] = 1 if aisle j is used
    y_names = [f'y_{j}' for j in range(n_aisles)]
    y = prob.variables.add(
        obj=[0] * n_aisles,  # No penalty for using aisles in this test
        lb=[0] * n_aisles,
        ub=[1] * n_aisles,
        types=['B'] * n_aisles,
        names=y_names
    )
    
    # Wave size constraints
    N_expr = list(range(n_orders))
    N_coef = [sum(order.values()) for order in filtered_orders]
    
    # Lower bound
    prob.linear_constraints.add(
        lin_expr=[[N_expr, N_coef]],
        senses=['G'],
        rhs=[wave_size_lb],
        names=['wave_size_lb']
    )
    
    # Upper bound
    prob.linear_constraints.add(
        lin_expr=[[N_expr, N_coef]],
        senses=['L'],
        rhs=[wave_size_ub],
        names=['wave_size_ub']
    )
    
    # Item coverage constraints
    for k in items:
        order_indices = []
        order_coeffs = []
        for i in range(n_orders):
            if k in filtered_orders[i]:
                order_indices.append(i)
                order_coeffs.append(filtered_orders[i][k])
        
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
    
    # Solve
    print("\nSolving without Item 7 constraint...")
    prob.solve()
    
    if prob.solution.get_status() == 101:
        obj_value = prob.solution.get_objective_value()
        x_sol = [prob.solution.get_values(f'x_{i}') for i in range(n_orders)]
        y_sol = [prob.solution.get_values(f'y_{j}') for j in range(n_aisles)]
        
        N = sum(sum(order.values()) * xi for i, (order, xi) in enumerate(zip(filtered_orders, x_sol)) if xi > 0.5)
        D = sum(1 for y in y_sol if y > 0.5)
        
        print(f"Solution found:")
        print(f"  N (total units) = {N}")
        print(f"  D (used aisles) = {D}")
        print(f"  Ratio N/D = {N/D if D > 0 else 'inf'}")
        print(f"  Selected orders: {[i for i, xi in enumerate(x_sol) if xi > 0.5]}")
        print(f"  Number of used aisles: {D}")
    else:
        print("No solution found")

if __name__ == '__main__':
    test_without_item7() 