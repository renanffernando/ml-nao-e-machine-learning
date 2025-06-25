import sys
from solver_random_orders import parse_input, solve_dinkelbach_orders

def test_simple(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    n, m, k, orders, aisles, l, r = parse_input(lines)
    
    print(f"Testing {filename}")
    print(f"Pedidos: {n}, Items: {m}, Pasillos: {k}")
    print(f"Rango: [{l}, {r}]")
    
    # Probar con TODOS los pedidos (sin selección aleatoria)
    selected_orders = list(range(n))
    
    print(f"\nResolviendo con TODOS los {n} pedidos...")
    x_sol, y_sol, ratio = solve_dinkelbach_orders(
        orders, aisles, l, r, selected_orders, max_lambda=0.0, show_iterations=True
    )
    
    if x_sol is not None:
        num_orders = sum(x_sol)
        num_aisles = sum(y_sol)
        
        # Calcular cantidad total manualmente para verificar
        total_quantity = 0
        for i in range(len(selected_orders)):
            if x_sol[i] == 1:
                total_quantity += orders[selected_orders[i]]['total_quantity']
        
        print(f"\nRESULTADO:")
        print(f"Ratio reportado: {ratio:.6f}")
        print(f"Pedidos seleccionados: {num_orders}")
        print(f"Pasillos abiertos: {num_aisles}")
        print(f"Cantidad total: {total_quantity}")
        print(f"Ratio verificado: {total_quantity / num_aisles if num_aisles > 0 else 0:.6f}")
        
        if abs(ratio - (total_quantity / num_aisles)) > 0.001:
            print("*** ERROR: Ratio no coincide! ***")
    else:
        print("No se encontró solución")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python test_simple.py <archivo>")
        sys.exit(1)
    
    test_simple(sys.argv[1]) 