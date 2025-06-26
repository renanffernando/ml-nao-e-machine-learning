import cplex
import sys
import logging
import os
from dinkelbach_solver import read_instance

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

class DirectSolver:
    def __init__(self, instance_name):
        self.orders, self.aisles, self.wave_size_lb, self.wave_size_ub = read_instance(instance_name)
        self.items = set()
        for order in self.orders:
            self.items.update(order.keys())
        
    def solve_direct(self):
        """Solve directly by trying different values of D and maximizing N"""
        best_ratio = 0
        best_solution = None
        
        # Try different values of D (number of aisles)
        for D_target in range(1, min(20, len(self.aisles) + 1)):  # Limit search to reasonable values
            try:
                prob = cplex.Cplex()
                prob.set_log_stream(None)
                prob.set_error_stream(None)
                prob.set_warning_stream(None)
                prob.set_results_stream(None)
                
                prob.objective.set_sense(prob.objective.sense.maximize)
                
                n_orders = len(self.orders)
                n_aisles = len(self.aisles)
                
                # Large constant to prioritize N over minimizing D
                M = 1000
                
                # x[i] = 1 if order i is selected
                x_names = [f'x_{i}' for i in range(n_orders)]
                x = prob.variables.add(
                    obj=[sum(order.values()) for order in self.orders],  # Maximize N
                    lb=[0] * n_orders,
                    ub=[1] * n_orders,
                    types=['B'] * n_orders,
                    names=x_names
                )
                
                # y[j] = 1 if aisle j is used
                y_names = [f'y_{j}' for j in range(n_aisles)]
                y = prob.variables.add(
                    obj=[-1.0/D_target] * n_aisles,  # Penalize using aisles to approximate N/D
                    lb=[0] * n_aisles,
                    ub=[1] * n_aisles,
                    types=['B'] * n_aisles,
                    names=y_names
                )
                
                # Constraint: exactly D_target aisles used
                prob.linear_constraints.add(
                    lin_expr=[[list(range(n_orders, n_orders + n_aisles)), [1] * n_aisles]],
                    senses=['E'],
                    rhs=[D_target],
                    names=['exact_aisles']
                )
                
                # Wave size constraints
                N_expr = list(range(n_orders))
                N_coef = [sum(order.values()) for order in self.orders]
                
                # Lower bound
                prob.linear_constraints.add(
                    lin_expr=[[N_expr, N_coef]],
                    senses=['G'],
                    rhs=[self.wave_size_lb],
                    names=['wave_size_lb']
                )
                
                # Upper bound
                prob.linear_constraints.add(
                    lin_expr=[[N_expr, N_coef]],
                    senses=['L'],
                    rhs=[self.wave_size_ub],
                    names=['wave_size_ub']
                )
                
                # Item coverage constraints
                for k in self.items:
                    order_indices = []
                    order_coeffs = []
                    for i in range(n_orders):
                        if k in self.orders[i]:
                            order_indices.append(i)
                            order_coeffs.append(self.orders[i][k])
                    
                    aisle_indices = []
                    aisle_coeffs = []
                    for j in range(n_aisles):
                        if k in self.aisles[j]:
                            aisle_indices.append(n_orders + j)
                            aisle_coeffs.append(-self.aisles[j][k])
                    
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
                prob.solve()
                
                if prob.solution.get_status() == 101:  # Optimal
                    obj_value = prob.solution.get_objective_value()
                    x_sol = [prob.solution.get_values(f'x_{i}') for i in range(n_orders)]
                    y_sol = [prob.solution.get_values(f'y_{j}') for j in range(n_aisles)]
                    
                    # Calculate actual N and D
                    N = sum(sum(order.values()) * xi for i, (order, xi) in enumerate(zip(self.orders, x_sol)) if xi > 0.5)
                    D = sum(1 for y in y_sol if y > 0.5)
                    ratio = N / D if D > 0 else 0
                    
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_solution = {
                            'N': N,
                            'D': D,
                            'ratio': ratio,
                            'selected_orders': [i for i, xi in enumerate(x_sol) if xi > 0.5],
                            'used_aisles': [j for j, yj in enumerate(y_sol) if yj > 0.5]
                        }
                        
                    print(f"D={D_target}: N={N}, D_actual={D}, ratio={ratio:.6f}")
                    
            except Exception as e:
                # No feasible solution for this D
                print(f"D={D_target}: No feasible solution")
                continue
                
        return best_solution

if __name__ == '__main__':
    try:
        instance_file = sys.argv[1] if len(sys.argv) > 1 else 'instance_0003.txt'
        solver = DirectSolver(instance_file)
        
        print(f"Solving {instance_file} with direct approach...")
        solution = solver.solve_direct()
        
        if solution:
            print(f"\nBest solution found:")
            print(f"  N = {solution['N']}")
            print(f"  D = {solution['D']}")
            print(f"  Ratio = {solution['ratio']:.6f}")
            print(f"  Selected orders: {solution['selected_orders']}")
            print(f"  Used aisles: {solution['used_aisles']}")
        else:
            print("No feasible solution found")
            
    except Exception as e:
        print(f"Error: {e}") 