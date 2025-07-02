import random
import time
from algorithms.dinkelbach import solve_dinkelbach_2
import math


def verbose_print(msg, aisles, cutoff=25):
    print(f"    {msg} {len(aisles)} ({sorted(list(aisles))[:cutoff]}{'...' if len(aisles) > cutoff else ''})")


def solve_with_incremental_aisles_2(orders, aisles, l_bound, r_bound, time_limit_minutes=10, verbose=True):
    """
    Versión optimizada de búsqueda local que reutiliza un modelo FIJO de Dinkelbach.

    FILOSOFÍA MODELO FIJO:
    - Crea el modelo CPLEX una vez con todas las variables y restricciones
    - Solo cambia bounds de variables (fijar/liberar) entre iteraciones
    - Usa warm start con la solución anterior
    - NO recrea restricciones → MUCHO más rápido

    Mejoras vs versión original:
    - ~3-5x más rápido por reutilización de modelo
    - Convergencia más rápida por warm start
    - Menor uso de memoria
    """
    n = len(orders)
    k = len(aisles)

    best_ratio = 0
    best_x = None
    best_y = None

    # Solución anterior para warm start
    previous_x = None
    previous_y = None

    # Modelo reutilizable
    reusable_model = None

    # MAX_LAMBDA GLOBAL QUE PERSISTE ENTRE ITERACIONES
    max_lambda = 0.0

    # Configuración
    aisles_per_iteration = min(len(aisles), 120)  # Numero de pasillos a evaluar
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
        print(f"INICIANDO BÚSQUEDA LOCAL")
        print(f"Pasillos disponibles: {k}")
        print(f"Límite de tiempo: {time_limit_minutes} minutos")
        print(f"Estrategia optimizada:")
        print(f"  - Pasillos por iteración: {aisles_per_iteration}")
        print(f"  - Modelo FIJO reutilizable: SÍ")
        print(f"  - Warm start automático: SÍ")
        print(f"  - Solo cambiar bounds + objetivo")
        print("-" * 70)

    base_aisles = set()
    if verbose:
        print(f"\nEMPEZANDO - conjunto base vacío")
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

        # En caso de que no haya más elementos o muchos fallos
        if aisles_per_iteration == len(base_aisles) or fail_count > 10:
            base_aisles = set(random.sample(list(base_aisles), (int)(0.92 * len(base_aisles) + 0.5)))

        # Determinar pasillos disponibles para selección aleatoria
        available_aisles = set(range(k)) - base_aisles

        # Si no hay pasillos disponibles para agregar, terminar
        if len(available_aisles) == 0:
            if verbose:
                print(f"  No hay más pasillos disponibles para agregar. Terminando.")
            break

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
            status = "CREANDO" if reusable_model is None else "REUTILIZANDO"
            print(f"  Resolviendo con Dinkelbach v2 ({status} modelo fijo)...")

        # Crear mapeo de índices locales para los pasillos seleccionados
        selected_aisle_indices = current_aisle_indices
        subset_aisles = list(range(len(current_aisle_indices)))  # Todos los pasillos en el subset están activos
        subset_orders = None  # Todos los pedidos activos

        # Resolver con Dinkelbach v2 (optimizado)
        x_sol, y_sol, ratio, reusable_model = solve_dinkelbach_2(
            orders=orders,
            selected_aisles=current_aisles_data,
            selected_aisle_indices=current_aisle_indices,
            l_bound=l_bound,
            r_bound=r_bound,
            k=k,
            initial_lambda=max_lambda,
            subset_orders=None,
            subset_aisles=None,
            warm_start_x=previous_x,
            warm_start_y=previous_y,
            model=reusable_model,
            all_aisles=aisles
        )

        if x_sol is not None:
            if verbose:
                print(f"  Ratio obtenido: {ratio:.6f}")

            # Actualizar mejor solución global
            if ratio > best_ratio:
                fail_count -= 1
                best_ratio = ratio
                best_x = x_sol
                best_y = y_sol

                # Guardar para warm start en próxima iteración
                previous_x = x_sol.copy()
                previous_y = y_sol.copy()

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
                    print(f"    Warm start guardado para próxima iteración")
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
            print(f"  Modelo fijo: {'REUTILIZADO' if reusable_model is not None else 'NO CREADO'}")

        iteration += 1

    # Limpiar el modelo al final
    if reusable_model is not None:
        try:
            reusable_model.end()
            if verbose:
                print(f"  Modelo fijo limpiado correctamente")
        except:
            pass

    total_time = time.time() - start_time
    if verbose:
        print(f"\n" + "=" * 70)
        print(f"BÚSQUEDA LOCAL CON MODELO FIJO COMPLETADA")
        print(f"Iteraciones realizadas: {iteration - 1}")
        print(f"Tiempo total: {total_time:.2f}s ({total_time / 60:.1f} minutos)")
        print(f"Pasillos en conjunto base final: {len(base_aisles)}")
        print(f"Mejor λ encontrado: {max_lambda:.6f}")
        print(f"Optimizaciones aplicadas:")
        print(f"  ✓ Modelo FIJO reutilizado entre iteraciones")
        print(f"  ✓ Warm start con solución anterior")
        print(f"  ✓ Solo cambio de bounds + objetivo (no restricciones)")
        print(f"  ✓ Pedidos inválidos fijados a 0 (no eliminados)")

    return best_x, best_y, best_ratio
