import os
import sys
from collections import defaultdict

def read_instance(instance_file):
    with open(instance_file, 'r') as f:
        # Read wave size bounds and number of orders
        wave_size_lb, wave_size_ub, num_orders = map(int, f.readline().strip().split())
        
        # Read orders
        orders = []  # Lista de diccionarios {item_id: quantity}
        for _ in range(num_orders):
            items = list(map(int, f.readline().strip().split()))
            num_items = items[0]
            order = {}
            for i in range(1, 2*num_items + 1, 2):
                item_id = items[i]
                quantity = items[i + 1]
                order[item_id] = quantity
            orders.append(order)
            
        # Skip empty lines
        line = f.readline()
        while line and line.strip() == "":
            line = f.readline()
            
        # Read aisles
        aisles = []  # Lista de diccionarios {item_id: capacity}
        while line:
            if line.strip():
                items = list(map(int, line.strip().split()))
                aisle_items = {}
                for item_id in items[1:]:  # Skip count
                    aisle_items[item_id] = 1  # Cada ítem tiene capacidad 1
                aisles.append(aisle_items)
            line = f.readline()
            
        return wave_size_lb, wave_size_ub, orders, aisles

def read_solution(solution_file):
    with open(solution_file, 'r') as f:
        # Read selected orders
        num_orders = int(f.readline().strip())
        selected_orders = []
        for _ in range(num_orders):
            order_id = int(f.readline().strip())
            selected_orders.append(order_id)
            
        # Read visited aisles
        num_aisles = int(f.readline().strip())
        visited_aisles = []
        for _ in range(num_aisles):
            aisle_id = int(f.readline().strip())
            visited_aisles.append(aisle_id)
            
        return selected_orders, visited_aisles

def analyze_solution(instance_file, solution_file, verbose=True):
    # Read instance and solution
    wave_size_lb, wave_size_ub, orders, aisles = read_instance(instance_file)
    selected_orders, visited_aisles = read_solution(solution_file)
    
    if verbose:
        print(f"\nAnalyzing solution:")
        print(f"Wave size bounds: [{wave_size_lb}, {wave_size_ub}]")
        print(f"Selected orders: {selected_orders}")
        print(f"Visited aisles: {visited_aisles}")
    
    # Calculate total units and verify wave size bounds
    total_units = sum(sum(order.values()) for i, order in enumerate(orders) if i in selected_orders)
    
    if verbose:
        print(f"Total units: {total_units}")
        
    if not (wave_size_lb <= total_units <= wave_size_ub):
        return None, f"Wave size bounds violated: {total_units} not in [{wave_size_lb}, {wave_size_ub}]"
    
    # Verify that selected aisles can cover all items in selected orders
    required_items = defaultdict(int)  # item_id -> total quantity needed
    for order_id in selected_orders:
        for item_id, quantity in orders[order_id].items():
            required_items[item_id] += quantity
    
    if verbose:
        print("\nRequired items:")
        for item_id, quantity in required_items.items():
            print(f"Item {item_id}: {quantity} units")
            
        print("\nAvailable items in visited aisles:")
        for aisle_id in visited_aisles:
            print(f"Aisle {aisle_id}: {aisles[aisle_id]}")
    
    # Check if visited aisles can provide all required items with enough capacity
    all_items_covered = True
    for item_id, required_quantity in required_items.items():
        # Calculate total capacity for this item across visited aisles
        available_capacity = 0
        for aisle_id in visited_aisles:
            if item_id in aisles[aisle_id]:
                available_capacity += aisles[aisle_id][item_id]
        
        if available_capacity < required_quantity:
            all_items_covered = False
            if verbose:
                print(f"Warning: Item {item_id} requires {required_quantity} units but only {available_capacity} available in visited aisles")
    
    # Calculate ratio - solo si todos los items están cubiertos
    ratio = total_units / len(visited_aisles) if all_items_covered and visited_aisles else 0
    
    if not all_items_covered:
        print("Warning: Some items are not covered by visited aisles or have insufficient capacity - ratio set to 0")
    
    return {
        'selected_orders': len(selected_orders),
        'total_units': total_units,
        'visited_aisles': len(visited_aisles),
        'ratio': ratio,
        'all_items_covered': all_items_covered
    }, None

def main():
    if len(sys.argv) != 4:
        print("Usage: python analyze_solutions.py <instances_dir> <solutions_dir> <output_csv>")
        sys.exit(1)
        
    instances_dir = sys.argv[1]
    solutions_dir = sys.argv[2]
    output_csv = sys.argv[3]
    
    # Process all solutions
    results = []
    for instance_file in sorted(os.listdir(instances_dir)):
        if not instance_file.endswith('.txt'):
            continue
            
        instance_path = os.path.join(instances_dir, instance_file)
        solution_file = os.path.join(solutions_dir, instance_file)
        
        if not os.path.exists(solution_file):
            print(f"Warning: No solution file for {instance_file}")
            continue
            
        print(f"\nProcessing {instance_file}...")
        metrics, error = analyze_solution(instance_path, solution_file)
        
        if error:
            print(f"Error in {instance_file}: {error}")
            continue
            
        results.append((instance_file, metrics))
        
    # Write results to CSV
    with open(output_csv, 'w') as f:
        # Header
        f.write("instance,selected_orders,total_units,visited_aisles,ratio,all_items_covered\n")
        
        # Data
        for instance_file, metrics in results:
            f.write(f"{instance_file},{metrics['selected_orders']},{metrics['total_units']},{metrics['visited_aisles']},{metrics['ratio']:.2f},{metrics['all_items_covered']}\n")

if __name__ == "__main__":
    main()