from src.solvers.exact.dinkelbach_solver import DinkelbachSolver
import sys

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python run_dinkelbach.py <input_file> <output_file>")
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