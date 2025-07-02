import random
import time
from algorithms.dinkelbach import solve_dinkelbach


def verbose_print(msg, aisles, cutoff=25):
    print(f"    {msg} {len(aisles)} ({sorted(list(aisles))[:cutoff]}{'...' if len(aisles) > cutoff else ''})")


def solve_with_incremental_aisles(orders, aisles, l_bound, r_bound, time_limit_minutes=10, verbose=True):
    n = len(orders)
    k = len(aisles)

    best_ratio = 0
    best_x = None
    best_y = None

    # MAX_LAMBDA GLOBAL QUE PERSISTE ENTRE ITERACIONES
    max_lambda = 0.0

    # Configuración
    aisles_per_iteration = 140  # Numero de pasillos a evaluar
    time_limit_seconds = time_limit_minutes * 60
    start_time = time.time()

    # Función para calcular capacidad máxima posible
    def calculate_max_capacity(aisle_indices):
        total_capacity_per_item = {}
        current_aisles_data = [aisles[i] for i in aisle_indices]
        for aisle in current_aisles_data:
            for item_id, capacity in aisle['items'].items():
                total_capacity_per_item[item_id] = total_capacity_per_item.get(item_id, 0) + capacity

        max_possible_quantity = 0
        for order in orders:
            can_satisfy_order = True
            for item_id, demand in order['items'].items():
                available_capacity = total_capacity_per_item.get(item_id, 0)
                if available_capacity < demand:
                    can_satisfy_order = False
                    break
            if can_satisfy_order:
                max_possible_quantity += order['total_quantity']
        return max_possible_quantity

    if verbose:
        print(f"INICIANDO BÚSQUEDA CON PASILLOS INCREMENTALES")
        print(f"Pasillos disponibles: {k}")
        print(f"Límite de tiempo: {time_limit_minutes} minutos")
        print(f"Estrategia:")
        print(f"  - Pasillos por iteración: {aisles_per_iteration}")
        print("-" * 70)

    base_aisles = set()
    if verbose:
        print(f"\nEMPEZANDO SIN WARM START - conjunto base vacío")
        print("-" * 70)

    iteration = 1
    fail_count = 0
    while True:
        iteration_start_time = time.time()

        # Verificar límite de tiempo
        elapsed_time = time.time() - start_time
        if elapsed_time >= time_limit_seconds:
            if verbose:
                print(f"\n*** LÍMITE DE TIEMPO ALCANZADO ({time_limit_minutes} minutos) ***")
            break

        if verbose:
            print(f"  Tiempo transcurrido: {elapsed_time:.1f}s / {time_limit_seconds}s")
            print(f"\nITERACIÓN {iteration}:")
            verbose_print("Pasillos base actuales:", base_aisles)

        # Determinar pasillos disponibles para selección aleatoria
        available_aisles = set(range(k)) - base_aisles

        # Si no hay pasillos disponibles para agregar, terminar
        if len(available_aisles) == 0:
            if verbose:
                print(f"  No hay más pasillos disponibles para agregar. Terminando.")
            break

        # en caso de que no haya mas elementos
        if aisles_per_iteration == len(base_aisles) or fail_count > 3:
            base_aisles = set(random.sample(list(base_aisles), (int)(0.8 * len(base_aisles))))

        # Seleccionar pasillos nuevos aleatoriamente
        new_aisles = set(random.sample(list(available_aisles), aisles_per_iteration - len(base_aisles)))

        # Conjunto total de pasillos para esta iteración
        current_aisles_set = base_aisles.union(new_aisles)
        current_aisle_indices = list(current_aisles_set)
        current_aisles_data = [aisles[i] for i in current_aisle_indices]

        if verbose:
            verbose_print("Pasillos nuevos:", new_aisles)
            print(f"  Total pasillos: {len(current_aisle_indices)}")

        # Verificar capacidad
        max_possible_quantity = calculate_max_capacity(current_aisle_indices)

        if max_possible_quantity < l_bound:
            if verbose:
                print(f"  Capacidad insuficiente (max={max_possible_quantity} < L={l_bound})")
                print(f"  Agregando pasillos nuevos al conjunto base sin resolver...")

            iteration += 1
            continue

        if verbose:
            print(f"  Capacidad suficiente (max={max_possible_quantity} >= L={l_bound})")
            print(f"  Resolviendo con Dinkelbach...")

        # Resolver con Dinkelbach
        x_sol, y_sol, ratio = solve_dinkelbach(orders, current_aisles_data, current_aisle_indices, l_bound, r_bound, k,
                                               max_lambda)

        if x_sol is not None:
            if verbose:
                print(f"  Ratio obtenido: {ratio:.6f}")

            # Actualizar mejor solución global
            if ratio > best_ratio:
                fail_count -= 1
                best_ratio = ratio
                best_x = x_sol
                best_y = y_sol
                if verbose:
                    print(f"    *** NUEVA MEJOR SOLUCIÓN GLOBAL: {ratio:.6f} ***")

                # Identificar pasillos usados en la mejor solución
                used_aisles = set()
                for i, aisle_idx in enumerate(current_aisle_indices):
                    if y_sol[aisle_idx] > 0.5:
                        used_aisles.add(aisle_idx)

                # SOLO agregar pasillos exitosos al conjunto base
                base_aisles = used_aisles.union(set(random.sample(list(base_aisles), (int)(0.3 * len(base_aisles)))))

                if verbose:
                    verbose_print("Pasillos usados en mejor solución:", used_aisles)
                    print(f"    Agregando {len(used_aisles)} pasillos exitosos al conjunto base")
                    print(f"    Nuevo conjunto base: {len(base_aisles)} pasillos")
            else:
                fail_count += 1
                # Si no mejoró, NO agregar pasillos al conjunto base
                if verbose:
                    print(f"    No mejoró - manteniendo conjunto base actual ({len(base_aisles)} pasillos)")

            if ratio > max_lambda:
                max_lambda = ratio
                if verbose:
                    print(f"    *** ACTUALIZANDO λ_global a {max_lambda:.6f} ***")
        else:
            fail_count += 1
            if verbose:
                print(f"  No factible - manteniendo conjunto base actual")

        iteration_time = time.time() - iteration_start_time
        if verbose:
            print(f"  Tiempo de iteración: {iteration_time:.2f}s")
            print(f"  Mejor ratio global: {best_ratio:.6f}")
            print(f"  λ_global actual: {max_lambda:.6f}")

        iteration += 1

    total_time = time.time() - start_time
    if verbose:
        print(f"\n" + "=" * 70)
        print(f"BÚSQUEDA INCREMENTAL COMPLETADA")
        print(f"Iteraciones realizadas: {iteration - 1}")
        print(f"Tiempo total: {total_time:.2f}s ({total_time / 60:.1f} minutos)")
        print(f"Pasillos en conjunto base final: {len(base_aisles)}")
        print(f"Mejor λ encontrado: {max_lambda:.6f}")

    return best_x, best_y, best_ratio
