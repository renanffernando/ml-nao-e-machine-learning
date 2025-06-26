import numpy as np

def read_dinkelbach_instance(file_path):
    """
    Reads a Wave Order Picking instance file in either format.
    
    Simple format:
    - First line: number of items (n)
    - Next n lines: item values
    
    Returns:
        orders: List of dictionaries {item_id: quantity}
        aisles: List of dictionaries {item_id: capacity}
        wave_size_lb: Lower bound for wave size
        wave_size_ub: Upper bound for wave size
    """
    with open(file_path, 'r') as f:
        # Read all lines and remove empty ones
        lines = [line.strip() for line in f.readlines() if line.strip()]
        
        # Convert all values to integers
        values = [int(x) for x in lines]
        
        # Create orders - each value represents required units
        order = {i: value for i, value in enumerate(values)}
        orders = [order]  # List with single order
        
        # Create aisles - each item gets its own aisle with capacity 1
        # This forces the solver to select aisles more carefully
        aisles = [{i: 1} for i, value in enumerate(values)]
        
        # Set wave size bounds based on total required units
        total_value = sum(values)
        wave_size_lb = total_value // 3  # At least one third of total value
        wave_size_ub = total_value  # At most all values
        
        return orders, aisles, wave_size_lb, wave_size_ub 