import numpy as np
import sys
from checker import WaveOrderPicking
import random
import time

class WaveOrderPickingSolver(WaveOrderPicking):
    def __init__(self):
        super().__init__()
        
    def solve(self, time_limit=600):
        """
        Implements a greedy strategy with some randomness to solve the problem
        """
        start_time = time.time()
        best_solution = None
        best_objective = 0
        
        while time.time() - start_time < time_limit:
            # Initialize empty solution
            selected_orders = []
            visited_aisles = []
            
            # Sort orders by total units (from highest to lowest)
            order_sizes = [(i, sum(order.values())) for i, order in enumerate(self.orders)]
            random.shuffle(order_sizes)  # Add randomness
            order_sizes.sort(key=lambda x: x[1], reverse=True)
            
            total_units = 0
            required_items = {}
            
            # Add orders until reaching the lower bound of units
            for order_idx, size in order_sizes:
                if total_units + size <= self.wave_size_ub:
                    selected_orders.append(order_idx)
                    total_units += size
                    
                    # Update required items
                    for item, quantity in self.orders[order_idx].items():
                        required_items[item] = required_items.get(item, 0) + quantity
                
                if total_units >= self.wave_size_lb:
                    break
            
            # If lower bound not reached, continue with next iteration
            if total_units < self.wave_size_lb:
                continue
            
            # Determine which aisles to visit to get required items
            # Create a list of aisles containing each item
            aisles_with_items = {}
            for item in required_items:
                aisles_with_items[item] = []
                for aisle_idx, aisle in enumerate(self.aisles):
                    if item in aisle:
                        aisles_with_items[item].append((aisle_idx, aisle[item]))
            
            # Greedy strategy: select aisles containing most required items
            aisle_scores = {}
            for item, aisles in aisles_with_items.items():
                for aisle_idx, quantity in aisles:
                    aisle_scores[aisle_idx] = aisle_scores.get(aisle_idx, 0) + min(quantity, required_items[item])
            
            # Sort aisles by score (from highest to lowest)
            sorted_aisles = sorted(aisle_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Select aisles until all required items are satisfied
            remaining_items = required_items.copy()
            for aisle_idx, _ in sorted_aisles:
                visited_aisles.append(aisle_idx)
                
                # Update remaining items
                for item, quantity in self.aisles[aisle_idx].items():
                    if item in remaining_items:
                        remaining_items[item] = max(0, remaining_items[item] - quantity)
                        if remaining_items[item] == 0:
                            del remaining_items[item]
                
                # If all items are satisfied, stop
                if not remaining_items:
                    break
            
            # Check if solution is feasible
            if self.is_solution_feasible(selected_orders, visited_aisles):
                objective = self.compute_objective_function(selected_orders, visited_aisles)
                
                if objective > best_objective:
                    best_objective = objective
                    best_solution = (selected_orders, visited_aisles)
        
        return best_solution, best_objective
    
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
        print("Usage: python python_solver.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    solver = WaveOrderPickingSolver()
    solver.read_input(input_file)
    
    # Set seed for reproducibility
    random.seed(42)
    
    # Solve the problem with a time limit of 10 seconds for this example
    (selected_orders, visited_aisles), objective = solver.solve(time_limit=10)
    
    if selected_orders and visited_aisles:
        print(f"Found solution with objective value: {objective}")
        print(f"Selected {len(selected_orders)} orders and {len(visited_aisles)} aisles")
        solver.write_solution(selected_orders, visited_aisles, output_file)
    else:
        print("No feasible solution found") 