import sys
import time
import threading
from ortools.linear_solver import pywraplp
from checker import WaveOrderPicking

# For progress bar
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("To see a progress bar, install tqdm: pip install tqdm")

class ProgressBar:
    def __init__(self, total_seconds=600):
        self.total_seconds = total_seconds
        self.start_time = time.time()
        self.stop_flag = False
        self.current_info = ""
        
    def start(self):
        if TQDM_AVAILABLE:
            self.pbar = tqdm(total=self.total_seconds, desc="Time", unit="s")
            self.thread = threading.Thread(target=self._update_progress)
            self.thread.daemon = True
            self.thread.start()
        else:
            print(f"Time limit: {self.total_seconds} seconds")
            
    def _update_progress(self):
        last_elapsed = 0
        while not self.stop_flag and last_elapsed < self.total_seconds:
            elapsed = min(int(time.time() - self.start_time), self.total_seconds)
            if elapsed > last_elapsed:
                self.pbar.update(elapsed - last_elapsed)
                last_elapsed = elapsed
                if self.current_info:
                    self.pbar.set_postfix_str(self.current_info)
            time.sleep(0.1)
        
    def update_info(self, info):
        self.current_info = info
        if not TQDM_AVAILABLE:
            elapsed = int(time.time() - self.start_time)
            print(f"[{elapsed}s/{self.total_seconds}s] {info}")
            
    def stop(self):
        self.stop_flag = True
        if TQDM_AVAILABLE:
            self.pbar.close()

class WaveOrderPickingRatioSolver(WaveOrderPicking):
    def __init__(self):
        super().__init__()
        
    def solve_with_ratio_mip(self, time_limit=600, verbose=True):
        """
        Solves the problem using mixed integer linear programming with a direct formulation,
        without using the ratio transformation technique.
        """
        # Start progress bar
        progress = ProgressBar(time_limit)
        progress.start()
        
        start_time = time.time()
        
        # Create solver
        if verbose:
            print("\nCreating SCIP solver...")
        solver = pywraplp.Solver.CreateSolver('SCIP')
        if not solver:
            progress.stop()
            print("Could not create SCIP solver")
            return None, 0
            
        # Set time limit (in seconds)
        solver_time_limit = int((time_limit - (time.time() - start_time)) * 1000)
        solver.set_time_limit(solver_time_limit)
        if verbose:
            print(f"Time limit for solver: {solver_time_limit/1000:.2f} seconds")
        
        # Variables: order and aisle selection
        if verbose:
            print(f"Creating variables for {len(self.orders)} orders and {len(self.aisles)} aisles...")
        
        order_vars = {}
        for i in range(len(self.orders)):
            order_vars[i] = solver.BoolVar(f'order_{i}')
            
        aisle_vars = {}
        for j in range(len(self.aisles)):
            aisle_vars[j] = solver.BoolVar(f'aisle_{j}')
        
        progress.update_info(f"Variables created: {len(order_vars) + len(aisle_vars)}")
            
        # Constraint: wave size limits
        if verbose:
            print(f"Wave size limits: [{self.wave_size_lb}, {self.wave_size_ub}]")
        
        # Calculate units per order
        order_units = [sum(self.orders[i].values()) for i in range(len(self.orders))]
        total_units = solver.Sum([order_units[i] * order_vars[i] for i in range(len(self.orders))])
        solver.Add(total_units >= self.wave_size_lb)
        solver.Add(total_units <= self.wave_size_ub)
        
        # Constraint: if an order is selected, we need to visit aisles containing its items
        if verbose:
            print("Adding aisle coverage constraints...")
            
        # For each order, identify required aisles
        order_aisles = {}
        for i in range(len(self.orders)):
            order_aisles[i] = set()
            for item in self.orders[i]:
                for j in range(len(self.aisles)):
                    if item in self.aisles[j]:
                        order_aisles[i].add(j)
                        
        # Add constraints
        for i in range(len(self.orders)):
            for j in order_aisles[i]:
                solver.Add(aisle_vars[j] >= order_vars[i])
                
        progress.update_info("Aisle constraints added")
        
        # Definition for use with Charnes-Cooper model
        # (https://lpsolve.sourceforge.net/5.5/ratio.htm)
        # Objective: maximize total_units / num_aisles
        # We implement an approximation: 
        # max total_units - K * num_aisles
        # where K is a scale factor to balance the terms
        K = sum(order_units) / len(self.aisles)  # Average value of units per order / aisles
        
        if verbose:
            print(f"Setting up objective function: max total_units - {K} * num_aisles")
            
        objective = solver.Objective()
        for i in range(len(self.orders)):
            objective.SetCoefficient(order_vars[i], order_units[i])
        for j in range(len(self.aisles)):
            objective.SetCoefficient(aisle_vars[j], -K)
        objective.SetMaximization()
        
        # Solve the model
        if verbose:
            print("\nSolving the model...")
        
        progress.update_info("Solving...")
        status = solver.Solve()
        
        # Stop progress bar
        progress.stop()
        
        if status == pywraplp.Solver.OPTIMAL:
            status_str = "OPTIMAL"
        elif status == pywraplp.Solver.FEASIBLE:
            status_str = "FEASIBLE"
        elif status == pywraplp.Solver.INFEASIBLE:
            status_str = "INFEASIBLE"
        elif status == pywraplp.Solver.UNBOUNDED:
            status_str = "UNBOUNDED"
        elif status == pywraplp.Solver.ABNORMAL:
            status_str = "ABNORMAL"
        elif status == pywraplp.Solver.MODEL_INVALID:
            status_str = "MODEL INVALID"
        elif status == pywraplp.Solver.NOT_SOLVED:
            status_str = "NOT SOLVED"
        else:
            status_str = f"UNKNOWN ({status})"
            
        if verbose:
            print(f"Solution status: {status_str}")
            print(f"Solution time: {solver.WallTime()/1000:.2f} seconds")
        
        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            # Get selected orders
            selected_orders = [i for i in range(len(self.orders)) if order_vars[i].solution_value() > 0.5]
            
            # Get visited aisles
            visited_aisles = [j for j in range(len(self.aisles)) if aisle_vars[j].solution_value() > 0.5]
            
            if verbose:
                total_selected_units = sum(order_units[i] for i in selected_orders)
                print(f"Selected orders: {len(selected_orders)}")
                print(f"Visited aisles: {len(visited_aisles)}")
                print(f"Total units: {total_selected_units}")
                
            # Calculate objective value
            objective_value = total_selected_units / len(visited_aisles) if visited_aisles else 0
            
            return (selected_orders, visited_aisles), objective_value
        else:
            if verbose:
                print("No feasible solution found")
            return None, 0

    def write_solution(self, selected_orders, visited_aisles, output_file_path):
        """
        Writes the solution in the required format
        """
        with open(output_file_path, 'w') as file:
            file.write(f"{len(selected_orders)}\n")
            for order in selected_orders:
                file.write(f"{order}\n")
            file.write(f"{len(visited_aisles)}\n")
            for aisle in visited_aisles:
                file.write(f"{aisle}\n")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python ortools_solver_ratio.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    print(f"\n{'='*50}")
    print(f"SOLVING INSTANCE (RATIO METHOD): {input_file}")
    print(f"{'='*50}")
    
    solver = WaveOrderPickingRatioSolver()
    solver.read_input(input_file)
    
    start_time = time.time()
    result = solver.solve_with_ratio_mip(time_limit=600, verbose=True)
    end_time = time.time()
    
    print(f"\n{'='*50}")
    print(f"RESULTS:")
    print(f"{'='*50}")
    
    if result is not None and result[0] is not None:
        (selected_orders, visited_aisles), objective = result
        print(f"Feasible solution found:")
        print(f"- Objective value: {objective}")
        print(f"- Selected orders: {len(selected_orders)}")
        print(f"- Visited aisles: {len(visited_aisles)}")
        print(f"- Ratio (units/aisles): {objective}")
        print(f"- Total time: {end_time - start_time:.2f} seconds")
        solver.write_solution(selected_orders, visited_aisles, output_file)
        print(f"Solution saved to: {output_file}")
    else:
        print("No feasible solution found")
        
    print(f"{'='*50}\n") 