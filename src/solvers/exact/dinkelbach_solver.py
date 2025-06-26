import sys
import time
import cplex as CPX
from checker import WaveOrderPicking
from .dinkelbach_instance_reader import read_dinkelbach_instance

class DinkelbachSolver(WaveOrderPicking):
    def __init__(self):
        super().__init__()
        self.order_units = None  # Units per order
        self.order_aisles = None  # Required aisles per order
        self.item_requirements = None  # Total units required per item
        self.aisle_capacities = None  # Total capacity per aisle
        
    def read_input(self, input_file_path):
        """Override parent's read_input to use the new format"""
        orders, aisles, wave_size_lb, wave_size_ub = read_dinkelbach_instance(input_file_path)
        self.orders = orders
        self.aisles = aisles
        self.wave_size_lb = wave_size_lb
        self.wave_size_ub = wave_size_ub
        
    def preprocess_data(self):
        """Preprocess data to speed up calculations"""
        # Calculate units per order
        self.order_units = [sum(order.values()) for order in self.orders]
        
        # For each order, find which aisles contain its items
        self.order_aisles = []
        for i, order in enumerate(self.orders):
            required_aisles = set()
            for item in order:
                for j, aisle in enumerate(self.aisles):
                    if item in aisle:
                        required_aisles.add(j)
            self.order_aisles.append(list(required_aisles))
            
        # Calculate total units required per item across all orders
        self.item_requirements = {}
        for i, order in enumerate(self.orders):
            for item, quantity in order.items():
                if item not in self.item_requirements:
                    self.item_requirements[item] = {}
                self.item_requirements[item][i] = quantity
                
        # Calculate aisle capacities
        self.aisle_capacities = []
        for aisle in self.aisles:
            total_capacity = sum(aisle.values())
            self.aisle_capacities.append(total_capacity)
    
    def solve_parametric_problem(self, q, time_limit=60, verbose=True):
        """
        Solves the parametric problem F(q) = max{N(x) - qD(x)} where:
        - N(x) is the total units selected
        - D(x) is the sum of aisle capacities visited
        - q is the current ratio estimate
        
        Returns:
            (selected_orders, visited_aisles, F(q)) if solution found, None otherwise
        """
        start_time = time.time()
        
        try:
            # Create CPLEX model
            model = CPX.Cplex()
            
            # Set parameters
            model.parameters.timelimit.set(time_limit)
            if not verbose:
                model.set_log_stream(None)
                model.set_error_stream(None)
                model.set_warning_stream(None)
                model.set_results_stream(None)
            
            if verbose:
                print(f"Solving parametric problem with q = {q:.4f}")
                print(f"Time limit: {time_limit:.2f} seconds")
            
            # Variables: order and aisle selection
            # Binary variables for orders: x[i] = 1 if order i is selected
            order_names = [f"order_{i}" for i in range(len(self.orders))]
            order_types = ['B'] * len(self.orders)
            model.variables.add(
                names=order_names,
                types=order_types
            )
            
            # Binary variables for aisles: y[j] = 1 if aisle j is visited
            aisle_names = [f"aisle_{j}" for j in range(len(self.aisles))]
            aisle_types = ['B'] * len(self.aisles)
            model.variables.add(
                names=aisle_names,
                types=aisle_types
            )
            
            # Wave size constraints
            # Lower bound: sum(units[i] * x[i]) >= wave_size_lb
            model.linear_constraints.add(
                lin_expr=[CPX.SparsePair(
                    ind=[f"order_{i}" for i in range(len(self.orders))],
                    val=[self.order_units[i] for i in range(len(self.orders))]
                )],
                senses=['G'],
                rhs=[self.wave_size_lb]
            )
            
            # Upper bound: sum(units[i] * x[i]) <= wave_size_ub
            model.linear_constraints.add(
                lin_expr=[CPX.SparsePair(
                    ind=[f"order_{i}" for i in range(len(self.orders))],
                    val=[self.order_units[i] for i in range(len(self.orders))]
                )],
                senses=['L'],
                rhs=[self.wave_size_ub]
            )
            
            # Item availability constraints
            # For each item, ensure selected aisles can provide enough units
            for item, orders in self.item_requirements.items():
                # Find aisles that have this item and their capacities
                aisles_with_item = []
                aisle_capacities = []
                for j, aisle in enumerate(self.aisles):
                    if item in aisle:
                        aisles_with_item.append(j)
                        aisle_capacities.append(aisle[item])
                
                if aisles_with_item:
                    # Create constraint: sum(aisle_capacity[j] * y[j]) >= sum(order_quantity[i] * x[i])
                    # for each order i that needs this item
                    order_ids = []
                    order_quantities = []
                    for order_id, quantity in orders.items():
                        order_ids.append(order_id)
                        order_quantities.append(quantity)
                    
                    model.linear_constraints.add(
                        lin_expr=[CPX.SparsePair(
                            ind=[f"aisle_{j}" for j in aisles_with_item] + [f"order_{i}" for i in order_ids],
                            val=aisle_capacities + [-q for q in order_quantities]
                        )],
                        senses=['G'],
                        rhs=[0.0]
                    )
            
            # Objective: maximize N(x) - qD(x)
            # N(x): sum(units[i] * x[i])
            # D(x): sum(capacity[j] * y[j])
            obj_order = [self.order_units[i] for i in range(len(self.orders))]  # Coefficients for orders
            obj_aisle = [-q * self.aisle_capacities[j] for j in range(len(self.aisles))]  # Coefficients for aisles
            
            model.objective.set_sense(model.objective.sense.maximize)
            model.objective.set_linear(
                [(f"order_{i}", obj_order[i]) for i in range(len(self.orders))] +
                [(f"aisle_{j}", obj_aisle[j]) for j in range(len(self.aisles))]
            )
            
            # Solve
            try:
                model.solve()
            except CPX.exceptions.CplexError as e:
                if verbose:
                    print(f"CPLEX error: {str(e)}")
                return None
            
            # Check solution status
            status = model.solution.get_status()
            if status in [model.solution.status.optimal, model.solution.status.MIP_optimal]:
                # Get solution
                order_values = model.solution.get_values(order_names)
                aisle_values = model.solution.get_values(aisle_names)
                
                selected_orders = [i for i in range(len(self.orders)) if order_values[i] > 0.5]
                visited_aisles = [j for j in range(len(self.aisles)) if aisle_values[j] > 0.5]
                
                # Calculate F(q)
                total_units = sum(self.order_units[i] for i in selected_orders)
                total_capacity = sum(self.aisle_capacities[j] for j in visited_aisles)
                F_q = total_units - q * total_capacity
                
                if verbose:
                    print(f"Solution found:")
                    print(f"- Selected orders: {len(selected_orders)}")
                    print(f"- Visited aisles: {len(visited_aisles)}")
                    print(f"- Total units: {total_units}")
                    print(f"- Total capacity: {total_capacity}")
                    print(f"- F(q): {F_q:.4f}")
                
                return selected_orders, visited_aisles, F_q
            else:
                if verbose:
                    print(f"No feasible solution found (status: {status})")
                return None
                
        except CPX.exceptions.CplexError as e:
            if verbose:
                print(f"CPLEX error: {str(e)}")
            return None
    
    def solve_with_dinkelbach(self, time_limit=600, epsilon=1e-4, max_iterations=100, verbose=True):
        """
        Solves the problem using Dinkelbach's algorithm.
        
        The algorithm iteratively solves the parametric problem F(q) = max{N(x) - qD(x)}
        where q is the current ratio estimate. It converges to the optimal ratio.
        
        Args:
            time_limit: Total time limit in seconds
            epsilon: Tolerance for convergence
            max_iterations: Maximum number of iterations
            verbose: If True, show detailed information
            
        Returns:
            (selected_orders, visited_aisles, best_ratio) if solution found, None otherwise
        """
        start_time = time.time()
        
        # Preprocess data
        if verbose:
            print("Preprocessing data...")
        self.preprocess_data()
        
        # Initial solution using a simple greedy approach
        # Sort orders by density (units / required aisles)
        densities = []
        for i in range(len(self.orders)):
            num_aisles = len(self.order_aisles[i]) if self.order_aisles[i] else 1
            density = self.order_units[i] / num_aisles
            densities.append((i, density))
        
        # Sort by density (highest to lowest)
        sorted_orders = [i for i, _ in sorted(densities, key=lambda x: x[1], reverse=True)]
        
        # Select orders until reaching minimum required units
        selected_orders = []
        total_units = 0
        for i in sorted_orders:
            if total_units < self.wave_size_lb:
                selected_orders.append(i)
                total_units += self.order_units[i]
                if total_units >= self.wave_size_lb:
                    break
        
        # Get required aisles for initial solution
        visited_aisles = []
        for order_idx in selected_orders:
            visited_aisles.extend(self.order_aisles[order_idx])
        visited_aisles = list(set(visited_aisles))  # Remove duplicates
        
        # Initial ratio
        total_capacity = sum(self.aisle_capacities[j] for j in visited_aisles)
        q = total_units / total_capacity if total_capacity else 0
        best_solution = (selected_orders, visited_aisles)
        best_ratio = q
        
        if verbose:
            print(f"\nInitial solution:")
            print(f"- Ratio (q₀): {q:.4f}")
            print(f"- Selected orders: {len(selected_orders)}")
            print(f"- Visited aisles: {len(visited_aisles)}")
            print(f"- Total units: {total_units}")
            print(f"- Total capacity: {total_capacity}")
        
        # Dinkelbach's algorithm main loop
        iteration = 0
        while iteration < max_iterations:
            # Check time limit
            elapsed_time = time.time() - start_time
            remaining_time = time_limit - elapsed_time
            
            if remaining_time < 10:  # Need at least 10 seconds
                if verbose:
                    print(f"\nTime limit reached after {iteration} iterations")
                break
            
            if verbose:
                print(f"\nIteration {iteration + 1}:")
                print(f"Current ratio (q): {q:.4f}")
            
            # Solve parametric problem
            result = self.solve_parametric_problem(
                q=q,
                time_limit=min(remaining_time - 5, 60),  # Max 60s per iteration
                verbose=verbose
            )
            
            if not result:
                if verbose:
                    print("Could not solve parametric problem")
                break
                
            selected_orders, visited_aisles, F_q = result
            
            # Calculate new ratio
            total_units = sum(self.order_units[i] for i in selected_orders)
            total_capacity = sum(self.aisle_capacities[j] for j in visited_aisles)
            new_ratio = total_units / total_capacity if total_capacity else 0
            
            if verbose:
                print(f"New ratio: {new_ratio:.4f}")
                print(f"F(q): {F_q:.4f}")
            
            # Check for convergence
            if abs(F_q) < epsilon:
                if verbose:
                    print(f"Converged after {iteration + 1} iterations")
                break
            
            # Update best solution if improved
            if new_ratio > best_ratio:
                best_solution = (selected_orders, visited_aisles)
                best_ratio = new_ratio
                
                if verbose:
                    print(f"Found better solution with ratio = {best_ratio:.4f}")
            
            # Update q for next iteration
            q = new_ratio
            iteration += 1
        
        if verbose:
            print(f"\nDinkelbach's algorithm completed:")
            print(f"- Best ratio found: {best_ratio:.4f}")
            print(f"- Total iterations: {iteration + 1}")
            print(f"- Total time: {time.time() - start_time:.2f} seconds")
        
        return best_solution[0], best_solution[1], best_ratio
    
    def write_solution(self, selected_orders, visited_aisles, output_file_path):
        """Write solution to file"""
        with open(output_file_path, 'w') as f:
            # Write number of selected orders
            f.write(f"{len(selected_orders)}\n")
            
            # Write selected orders
            for order in selected_orders:
                f.write(f"{order}\n")
            
            # Write number of visited aisles
            f.write(f"{len(visited_aisles)}\n")
            
            # Write visited aisles
            for aisle in visited_aisles:
                f.write(f"{aisle}\n")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python dinkelbach_solver.py <input_file> <output_file>")
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    solver = DinkelbachSolver()
    solver.read_input(input_file)
    
    result = solver.solve_with_dinkelbach(
        time_limit=600,
        epsilon=1e-4,
        max_iterations=100,
        verbose=True
    )
    
    if result:
        selected_orders, visited_aisles, best_ratio = result
        solver.write_solution(selected_orders, visited_aisles, output_file)
        print(f"\nBest ratio: {best_ratio:.4f}")
    else:
        print("\nNo solution found")