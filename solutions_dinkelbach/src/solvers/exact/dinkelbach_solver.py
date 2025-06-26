import cplex
from src.solvers.exact.dinkelbach_instance_reader import read_dinkelbach_instance

class DinkelbachSolver:
    def __init__(self, instance_file):
        self.orders, self.aisles, self.wave_size_lb, self.wave_size_ub = read_dinkelbach_instance(instance_file)
        self.order_units = self._get_order_units()
        self.aisle_capacities = self._get_aisle_capacities()
        
    def _get_order_units(self):
        """Get total units for each order"""
        return [sum(order.values()) for order in self.orders]
        
    def _get_aisle_capacities(self):
        """Get total capacity for each aisle"""
        return [sum(aisle.values()) for aisle in self.aisles]
        
    def solve_parametric(self, q):
        """Solve the parametric problem for a given q value"""
        prob = cplex.Cplex()
        prob.objective.set_sense(prob.objective.sense.maximize)
        
        # Create variables for orders and aisles
        n_orders = len(self.orders)
        n_aisles = len(self.aisles)
        
        # x[i] = 1 if order i is selected
        x = prob.variables.add(
            obj=[self.order_units[i] for i in range(n_orders)],
            lb=[0] * n_orders,
            ub=[1] * n_orders,
            types=['B'] * n_orders,
            names=[f'x_{i}' for i in range(n_orders)]
        )
        
        # y[j] = 1 if aisle j is used
        y = prob.variables.add(
            obj=[-q * self.aisle_capacities[j] for j in range(n_aisles)],
            lb=[0] * n_aisles,
            ub=[1] * n_aisles,
            types=['B'] * n_aisles,
            names=[f'y_{j}' for j in range(n_aisles)]
        )
        
        # Wave size constraints
        total_units = prob.linear_constraints.add(
            lin_expr=[[range(n_orders), [self.order_units[i] for i in range(n_orders)]]],
            senses=['R'],
            rhs=[self.wave_size_ub],
            range_values=[self.wave_size_ub - self.wave_size_lb],
            names=['wave_size']
        )
        
        # Solve
        prob.set_log_stream(None)
        prob.set_error_stream(None)
        prob.set_warning_stream(None)
        prob.set_results_stream(None)
        
        prob.solve()
        
        if prob.solution.get_status() == prob.solution.status.optimal:
            obj_value = prob.solution.get_objective_value()
            x_sol = prob.solution.get_values(x)
            y_sol = prob.solution.get_values(y)
            return obj_value, x_sol, y_sol
        else:
            return None, None, None
            
    def solve(self, epsilon=1e-6, max_iter=100):
        """Solve using Dinkelbach's algorithm"""
        # Initial solution - select all orders and aisles
        x = [1] * len(self.orders)
        y = [1] * len(self.aisles)
        
        # Initial ratio q₀ = N(x)/D(x)
        q = sum(u * x[i] for i, u in enumerate(self.order_units)) / sum(c * y[j] for j, c in enumerate(self.aisle_capacities))
        print(f'Initial ratio (q₀) = {q:.4f}')
        
        iteration = 0
        while iteration < max_iter:
            # Solve parametric problem
            obj, new_x, new_y = self.solve_parametric(q)
            
            if obj is None:
                print('No feasible solution found')
                return None
                
            # Calculate new ratio
            N = sum(u * new_x[i] for i, u in enumerate(self.order_units))
            D = sum(c * new_y[j] for j, c in enumerate(self.aisle_capacities))
            new_q = N / D if D > 0 else float('inf')
            
            # Check convergence
            if abs(new_q - q) < epsilon:
                print(f'Converged in {iteration + 1} iterations')
                print(f'Final ratio = {new_q:.4f}')
                print(f'Selected orders: {[i for i, xi in enumerate(new_x) if xi > 0.5]}')
                print(f'Used aisles: {[j for j, yj in enumerate(new_y) if yj > 0.5]}')
                return new_x, new_y, new_q
                
            q = new_q
            iteration += 1
            
        print('Maximum iterations reached')
        return None 