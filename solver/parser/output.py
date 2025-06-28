def status(x_sol, y_sol, opt_val, start_time, end_time):
    n = len(x_sol)
    k = len(y_sol)
    print(f"\nRESUMEN:")
    print(f"Tiempo total: {end_time - start_time:.2f} segundos")
    if x_sol is not None:
        print(f"Mejor ratio encontrado: {opt_val:.2f}")
    else:
        print("No se encontró ninguna solución factible.")
        return
    
    # Mostrar mejor solución
    print()
    print("=" * 60)
    print("MEJOR SOLUCIÓN ENCONTRADA")
    print("=" * 60)
    
    # Pedidos seleccionados
    selected_orders = [i for i in range(n) if x_sol[i] > 0.5]
    print(f"Pedidos seleccionados ({len(selected_orders)}): {selected_orders}")
    
    # Pasillos seleccionados
    selected_aisles = [i for i in range(k) if y_sol[i] > 0.5]
    print(f"Pasillos seleccionados ({len(selected_aisles)}): {selected_aisles}")
    
