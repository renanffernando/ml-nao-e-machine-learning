import os
import sys
import time

# Agregar el directorio actual al path de Python
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'parser'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'algorithms'))

# Importar directamente los archivos
import checker
import tabu_search
import local_search
import local_search_big
import input as input_module
import output as output_module


def solve(orders, aisles, l, r, time_limit_minutes=10):
    if len(orders) >= 20000:
        return local_search_big.solve_with_incremental_aisles(orders, aisles, l, r, time_limit_minutes)
    return tabu_search.search(orders, aisles, l, r, time_limit_minutes)
    # return local_search.solve_with_incremental_aisles_2(orders, aisles, l, r, time_limit_minutes)


if __name__ == "__main__":
    filename = "input.txt" if len(sys.argv) == 1 else sys.argv[1]
    result = input_module.read(filename)
    if result is None:
        print("Error: No se pudieron leer los datos del archivo")
        sys.exit(1)

    n, m, k, orders, aisles, l_bound, r_bound = result

    start_time = time.time()
    best_x_sol, best_y_sol, best_ratio = solve(orders, aisles, l_bound, r_bound)
    end_time = time.time()

    # Mostrar información de la solución
    output_module.status(best_x_sol, best_y_sol, best_ratio, start_time, end_time)

    # Verificación con checker original
    if best_x_sol is not None and best_y_sol is not None:
        print("\n" + "=" * 60)
        print("VERIFICACIÓN DE LA SOLUCIÓN")
        print("=" * 60)

        is_valid, check_msg, stats \
            = checker.verify_solution(best_x_sol, best_y_sol, best_ratio, orders, aisles, l_bound, r_bound)

        if is_valid:
            print("✅ SOLUCIÓN VÁLIDA")
            print(f"   {check_msg}")
            print(f"\n📊 ESTADÍSTICAS:")
            print(f"   • Pedidos seleccionados: {stats['num_selected_orders']}/{n}")
            print(f"   • Pasillos seleccionados: {stats['num_selected_aisles']}/{k}")
            print(f"   • Cantidad total: {stats['total_quantity']}")
            print(f"   • Ratio calculado: {stats['calculated_ratio']:.6f}")
            print(f"   • Rango cumplido: {stats['range_compliance']}")
            print(f"   • Ítems verificados: {stats['capacity_items_checked']}")
        else:
            print("❌ SOLUCIÓN INVÁLIDA")
            print(f"   ERROR: {check_msg}")

        print("=" * 60)
