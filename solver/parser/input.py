def parse(lines):
    line_idx = 0
    
    # Primera línea: n, m, k
    n, m, k = map(int, lines[line_idx].strip().split())
    line_idx += 1
    
    # Siguientes n líneas: pedidos
    orders = []
    for i in range(n):
        parts = lines[line_idx].strip().split()
        line_idx += 1
        
        # Formato: num_items item1 qty1 item2 qty2 ...
        num_items = int(parts[0])
        total_quantity = 0
        items = {}
        
        for j in range(num_items):
            item_id = int(parts[1 + j*2])
            quantity = int(parts[1 + j*2 + 1])
            items[item_id] = quantity
            total_quantity += quantity
        
        orders.append({
            'id': i,
            'total_quantity': total_quantity,
            'items': items
        })
    
    # Siguientes k líneas: pasillos
    aisles = []
    for i in range(k):
        parts = lines[line_idx].strip().split()
        line_idx += 1
        
        # Formato: num_items item1 cap1 item2 cap2 ...
        num_items = int(parts[0])
        items = {}
        
        for j in range(num_items):
            item_id = int(parts[1 + j*2])
            capacity = int(parts[1 + j*2 + 1])
            items[item_id] = capacity
        
        aisles.append({
            'id': i,
            'items': items
        })
    
    # Última línea: L, R
    l_bound, r_bound = map(int, lines[line_idx].strip().split())
    
    return n, m, k, orders, aisles, l_bound, r_bound


def read(filename):
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()

        if not lines:
            print("ERROR: No se proporcionaron datos de entrada.")
            return

    except FileNotFoundError:
        print(f"ERROR: Archivo no encontrado: {filename}")
        return

    return parse(lines)
