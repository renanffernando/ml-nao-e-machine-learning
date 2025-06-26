import sys
import random
from solver_random_orders import parse_input, solve_dinkelbach_orders

def test_with_limited_aisles(filename, num_aisles=4):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    n, m, k, orders, aisles, l, r = parse_input(lines)
    
    print(f"Testing {filename} with {num_aisles} random aisles")
    print(f"Pedidos: {n}, Items: {m}, Pasillos totales: {k}")
    print(f"Rango: [{l}, {r}]")
    
    # Seleccionar aleatoriamente num_aisles pasillos
    selected_aisle_indices = sorted(random.sample(range(k), min(num_aisles, k)))
    selected_aisles = [aisles[i] for i in selected_aisle_indices]
    
    print(f"Pasillos seleccionados: {selected_aisle_indices}")
    
    # Resolver con todos los pedidos pero solo los pasillos seleccionados
    selected_orders = list(range(n))
    
    print(f"\nResolviendo con TODOS los {n} pedidos y {len(selected_aisles)} pasillos...")
    x_sol, y_sol, ratio = solve_dinkelbach_orders(
        orders, selected_aisles, l, r, selected_orders, max_lambda=0.0, show_iterations=True
    )
    
    if x_sol is not None:
        num_orders = sum(x_sol)
        num_aisles = sum(y_sol)
        
        # Calcular cantidad total manualmente
        total_quantity = 0
        for i in range(len(selected_orders)):
            if x_sol[i] == 1:
                total_quantity += orders[selected_orders[i]]['total_quantity']
        
        print(f"\nRESULTADO:")
        print(f"Ratio reportado: {ratio:.6f}")
        print(f"Pedidos seleccionados: {num_orders}")
        print(f"Pasillos abiertos: {num_aisles} de {len(selected_aisles)} disponibles")
        print(f"Cantidad total: {total_quantity}")
        print(f"Ratio verificado: {total_quantity / num_aisles if num_aisles > 0 else 0:.6f}")
        
        if num_aisles == len(selected_aisles):
            print("*** PROBLEMA: Está abriendo TODOS los pasillos disponibles ***")
        
    else:
        print("No se encontró solución")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python test_limited_aisles.py <archivo>")
        sys.exit(1)
    
    # Probar con diferentes números de pasillos
    for num_aisles in [4, 8, 16]:
        print("=" * 60)
        test_with_limited_aisles(sys.argv[1], num_aisles)
        print() 