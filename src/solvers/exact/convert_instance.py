import sys

def convert_instance(input_file, output_file):
    """
    Converts an instance from the current format to the standard format.
    
    Current format:
    - First line: number of items (n)
    - Next n lines: item values
    
    Standard format:
    O I A  (number of orders, items, aisles)
    D item1 qty1 item2 qty2...  (for each order)
    D item1 qty1 item2 qty2...  (for each aisle)
    LB UB  (wave size bounds)
    """
    # Read current format
    with open(input_file, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
        values = [int(x) for x in lines]
    
    # Convert to standard format
    n = len(values)  # Number of items
    
    # Create a single order with all items
    order = {i: value for i, value in enumerate(values)}
    
    # Create one aisle per item
    aisles = [{i: value} for i, value in enumerate(values)]
    
    # Calculate wave size bounds
    total_value = sum(values)
    wave_size_lb = total_value // 2  # At least half of total value
    wave_size_ub = total_value  # At most all values
    
    # Write standard format
    with open(output_file, 'w') as f:
        # First line: O I A (1 order, n items, n aisles)
        f.write(f"1 {n} {n}\n")
        
        # Single order with all items
        items = list(order.items())
        f.write(f"{len(items)}")  # Number of items in order
        for item_id, qty in items:
            f.write(f" {item_id} {qty}")
        f.write("\n")
        
        # One aisle per item
        for i, aisle in enumerate(aisles):
            items = list(aisle.items())
            f.write(f"{len(items)}")  # Number of items in aisle
            for item_id, capacity in items:
                f.write(f" {item_id} {capacity}")
            f.write("\n")
        
        # Wave size bounds
        f.write(f"{wave_size_lb} {wave_size_ub}\n")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_instance.py <input_file> <output_file>")
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    convert_instance(input_file, output_file) 