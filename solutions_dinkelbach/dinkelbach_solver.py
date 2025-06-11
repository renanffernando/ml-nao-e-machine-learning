import cplex
import sys
import logging
import os
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def read_instance(instance_name):
    """
    Reads a Wave Order Picking instance file from datasets/a/.
    Returns:
        orders: List of dictionaries {item_id: quantity}
        aisles: List of dictionaries {item_id: capacity}
        wave_size_lb: Lower bound for wave size
        wave_size_ub: Upper bound for wave size
    """
    try:
        file_path = os.path.normpath(instance_name if os.path.isabs(instance_name) else os.path.join('..', 'datasets', 'a', instance_name))
        logging.info(f"Reading instance file: {file_path}")
        
        with open(file_path, 'r') as f:
            # Read all lines and remove empty ones
            lines = [line.strip() for line in f.readlines() if line.strip()]
            logging.info(f"Read {len(lines)} lines from {file_path}")
            
            # First line: number of orders, items, and aisles
            n_orders, n_items, n_aisles = map(int, lines[0].split())
            current_line = 1
            
            # Read orders
            orders = []
            for _ in range(n_orders):
                order_items = {}
                values = list(map(int, lines[current_line].split()))
                n_items_in_order = values[0]
                for i in range(n_items_in_order):
                    item_id = values[2*i + 1]
                    quantity = values[2*i + 2]
                    order_items[item_id] = quantity
                orders.append(order_items)
                current_line += 1
            
            # Read aisles (same format as orders)
            aisles = []
            for _ in range(n_aisles):
                aisle_items = {}
                values = list(map(int, lines[current_line].split()))
                n_items_in_aisle = values[0]
                for i in range(n_items_in_aisle):
                    item_id = values[2*i + 1]
                    capacity = values[2*i + 2]
                    aisle_items[item_id] = capacity
                aisles.append(aisle_items)
                current_line += 1
            
            # Last line: wave size bounds
            wave_size_lb, wave_size_ub = map(int, lines[-1].split())
            # Ensure lower bound is at least 1
            wave_size_lb = max(1, wave_size_lb)
            
            logging.info(f"Instance: {n_orders} orders, {n_items} items, {n_aisles} aisles, wave bounds [{wave_size_lb}, {wave_size_ub}]")
            
            return orders, aisles, wave_size_lb, wave_size_ub
    except Exception as e:
        logging.error(f"Error reading instance file: {e}")
        raise

class DinkelbachSolver:
    def __init__(self, instance_name):
        self.orders, self.aisles, self.wave_size_lb, self.wave_size_ub = read_instance(instance_name)
        self.items = set()
        for order in self.orders:
            self.items.update(order.keys())
        
    def solve_parametric(self, q, warm_start_x=None, warm_start_y=None):
        """
        Solve the parametric problem F(q) = max{N(x) - qD(x)} where:
        N(x) = suma_i(suma_k(unidades[i][k] * x[i]))  # total units across all selected orders
        D(x) = suma_j(y[j])  # number of used aisles
        """
        try:
            prob = cplex.Cplex()
            prob.set_log_stream(None)
            prob.set_error_stream(None)
            prob.set_warning_stream(None)
            prob.set_results_stream(None)
            
            # Set time limit and other parameters
            prob.parameters.timelimit.set(300.0)  # 5 minutes
            prob.parameters.mip.tolerances.mipgap.set(1e-6)
            prob.parameters.mip.tolerances.absmipgap.set(1e-6)
            
            prob.objective.set_sense(prob.objective.sense.maximize)
            
            # Create variables for orders and aisles
            n_orders = len(self.orders)
            n_aisles = len(self.aisles)
            
            # x[i] = 1 if order i is selected
            x_names = [f'x_{i}' for i in range(n_orders)]
            x = prob.variables.add(
                obj=[sum(order.values()) for order in self.orders],  # N = sum of units per order
                lb=[0] * n_orders,
                ub=[1] * n_orders,
                types=['B'] * n_orders,
                names=x_names
            )
            
            # y[j] = 1 if aisle j is used
            y_names = [f'y_{j}' for j in range(n_aisles)]
            y = prob.variables.add(
                obj=[-q] * n_aisles,  # -qD
                lb=[0] * n_aisles,
                ub=[1] * n_aisles,
                types=['B'] * n_aisles,
                names=y_names
            )
            
            # Wave size constraints
            # lb <= sum_i(sum(order[i]) * x[i]) <= ub
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
            # For each item k:
            # suma_i(unidades[i][k] * x[i]) <= suma_j(capacidad[j][k] * y[j])
            for k in self.items:
                # Left side: sum of units required by selected orders
                order_indices = []
                order_coeffs = []
                for i in range(n_orders):
                    if k in self.orders[i]:
                        order_indices.append(i)
                        order_coeffs.append(self.orders[i][k])
                
                # Right side: sum of units available in selected aisles
                aisle_indices = []
                aisle_coeffs = []
                for j in range(n_aisles):
                    if k in self.aisles[j]:
                        aisle_indices.append(n_orders + j)  # Correct indexing for y variables
                        aisle_coeffs.append(-self.aisles[j][k])  # Negative because moving to RHS
                
                # Add constraint only if item is required
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
            
            # Apply warm start if provided
            if warm_start_x is not None and warm_start_y is not None:
                try:
                    # Set warm start values
                    var_names = x_names + y_names
                    var_values = warm_start_x + warm_start_y
                    prob.MIP_starts.add([var_names, var_values], prob.MIP_starts.effort_level.auto)
                    logging.info(f"  Applied warm start from previous iteration")
                except Exception as e:
                    logging.warning(f"  Could not apply warm start: {e}")
            
            # Solve
            prob.solve()
            
            status = prob.solution.get_status()
            
            if status == 101:  # Optimal solution found
                obj_value = prob.solution.get_objective_value()
                x_sol = [prob.solution.get_values(f'x_{i}') for i in range(n_orders)]
                y_sol = [prob.solution.get_values(f'y_{j}') for j in range(n_aisles)]
                
                # Calculate N and D
                N = sum(sum(order.values()) * xi for i, (order, xi) in enumerate(zip(self.orders, x_sol)) if xi > 0.5)
                D = sum(1 for y in y_sol if y > 0.5)  # Number of used aisles
                
                return obj_value, x_sol, y_sol, N, D
            elif status == 103:  # Infeasible
                return None, None, None, None, None
            elif status == 102:  # Unbounded
                return None, None, None, None, None
            else:
                return None, None, None, None, None
        except Exception as e:
            logging.error(f"Error solving parametric problem: {e}")
            raise
            
    def solve(self, epsilon=1e-6, max_iter=100):
        """Solve using Dinkelbach's algorithm"""
        try:
            # Start with q₀ = 0
            q = 0
            best_q = 0
            best_x = None
            best_y = None
            best_N = None
            best_D = None
            
            # Variables for warm start
            prev_x = None
            prev_y = None
            
            iteration = 0
            while iteration < max_iter:
                logging.info(f"Iteration {iteration + 1}: q = {q:.6f}")
                try:
                    # Solve parametric problem F(q) = max{N - qD} with warm start
                    obj, new_x, new_y, N, D = self.solve_parametric(q, prev_x, prev_y)
                    
                    if obj is None:
                        logging.warning('No feasible solution found')
                        break
                        
                    logging.info(f"  N = {N}, D = {D}, F(q) = {obj:.6f}")
                    
                    # Store current solution for next warm start
                    prev_x = new_x.copy() if new_x else None
                    prev_y = new_y.copy() if new_y else None
                    
                    # Check convergence: F(q) ≈ 0
                    if abs(obj) < epsilon:
                        # Converged - q is the optimal ratio
                        logging.info(f"CONVERGED in {iteration + 1} iterations")
                        logging.info(f"Optimal ratio = {q:.6f}")
                        logging.info(f"Selected orders: {[i for i, xi in enumerate(new_x) if xi > 0.5]}")
                        logging.info(f"Used aisles: {[j for j, yj in enumerate(new_y) if yj > 0.5]}")
                        return new_x, new_y, q
                    
                    # If F(q) > 0, update q and continue
                    if obj > epsilon:
                        # Update best solution
                        current_ratio = N / D if D > 0 else float('inf')
                        if current_ratio > best_q:
                            best_q = current_ratio
                            best_x = new_x
                            best_y = new_y
                            best_N = N
                            best_D = D
                        
                        # Update q for next iteration
                        q = current_ratio
                        logging.info(f"  Next q = {q:.6f}")
                        iteration += 1
                    else:
                        # F(q) ≤ 0, algorithm should have converged
                        logging.info(f"F(q) ≤ 0, converged with ratio = {q:.6f}")
                        return new_x, new_y, q
                        
                except Exception as e:
                    logging.error(f"Error in iteration {iteration + 1}: {e}")
                    raise
                
            logging.warning('Maximum iterations reached')
            if best_x is not None:
                logging.info(f'Best ratio found = {best_q:.6f}')
                logging.info(f'Selected orders: {[i for i, xi in enumerate(best_x) if xi > 0.5]}')
                logging.info(f'Used aisles: {[j for j, yj in enumerate(best_y) if yj > 0.5]}')
                return best_x, best_y, best_q
            else:
                logging.error('No solution found')
                return None, None, 0
        except Exception as e:
            logging.error(f"Error in Dinkelbach algorithm: {e}")
            raise

if __name__ == '__main__':
    try:
        instance_file = sys.argv[1] if len(sys.argv) > 1 else 'instance_0001.txt'
        solver = DinkelbachSolver(instance_file)
        solver.solve()
    except Exception as e:
        logging.error(f"Error running solver: {e}")
        sys.exit(1) 