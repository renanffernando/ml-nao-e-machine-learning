import time
from math import ceil, log2
from collections import deque

import matplotlib.pyplot as plt
from random import sample, choice
from dinkelbach import solve_dinkelbach_2
from utils import calculate_max_capacity, progress_bar, verbose_print


def search(orders, aisles, l_bound, r_bound, time_limit_minutes=10, verbose=True, plot=True):
    # Start timing
    start_time = time.time()
    time_limit = time_limit_minutes * 60

    # Parameters
    n = len(orders)
    k = len(aisles)
    tabu_tenure = 200
    neighborhood_size = 10
    neighborhood_width = 1
    initial_k = min(k, 10)
    max_fail = ceil(1.3 ** log2(n))  # Number of trials before soft reset

    all_aisle_indices = set(range(k))
    current_aisle_indices = set(sample(list(all_aisle_indices), initial_k))
    current_aisles_data = [aisles[i] for i in current_aisle_indices]

    previous_x = None
    previous_y = None
    reusable_model = None

    while True:
        # Verificar capacidad
        max_possible_quantity = calculate_max_capacity(orders, aisles, current_aisle_indices)

        if max_possible_quantity < l_bound:
            if verbose:
                print(f'  Capacidad insuficiente (max={max_possible_quantity} < L={l_bound})')
                print(f'  Agregando pasillos nuevos al conjunto base sin resolver...')
            if initial_k < k:  # increase the base pool size
                initial_k += 1
                current_aisle_indices = set(sample(list(all_aisle_indices), initial_k))
                print(f'    Increasing initial aisles set to {initial_k}.\n')
        else:
            break

    # Initial solve
    x_sol, y_sol, ratio, reusable_model = solve_dinkelbach_2(
        orders=orders,
        selected_aisles=current_aisles_data,
        selected_aisle_indices=current_aisle_indices,
        l_bound=l_bound,
        r_bound=r_bound,
        k=k,
        initial_lambda=0,
        warm_start_x=previous_x,
        warm_start_y=previous_y,
        model=reusable_model,
        all_aisles=aisles
    )

    used_aisles = set()
    if ratio is not None:
        used_aisles = set(i for i in current_aisle_indices if y_sol[i] > 0.5)
    else:
        ratio = 0
    best_solution = {
        'aisles': used_aisles,
        'aisles_poll': current_aisle_indices,
        'x_sol': x_sol,
        'y_sol': y_sol,
        'ratio': ratio
    }

    best_objective_history = [ratio]
    tabu_list = deque(maxlen=tabu_tenure)
    iteration = 0

    optimality = False
    fail_count = 0
    best_neighbor = None
    while (elapsed_time := time.time() - start_time) < time_limit:
        iteration += 1
        if verbose:
            print(f'\nStarting iteration {iteration} with {len(current_aisle_indices)} aisles:')
            print(f"  Tiempo transcurrido: {elapsed_time:.1f}s / {time_limit}s")

        neighborhood = set()
        for _ in range(neighborhood_size):
            candidate_indices = set(current_aisle_indices).copy()
            move_type = choice(['add', 'remove', 'swap'])
            remaining = all_aisle_indices - candidate_indices

            if move_type != 'remove':
                if remaining:
                    move_size = min(neighborhood_width, len(remaining))
                    new_aisles = set(sample(list(remaining), move_size))
                    candidate_indices = candidate_indices.union(new_aisles)
                else:
                    continue

            if move_type != 'add':
                if best_neighbor is not None:
                    used_aisles = set(best_neighbor['aisles']).copy()
                    if len(used_aisles) > 1:
                        move_size = min(neighborhood_width, len(used_aisles))
                        candidate_indices -= set(sample(list(used_aisles), move_size))
                        if len(candidate_indices) == 0: continue

            candidate_list = tuple(candidate_indices)
            if candidate_list not in neighborhood:
                neighborhood.add(candidate_list)

            # Check if the full model is going to be checked:
            if len(candidate_list) == k:
                neighborhood = {candidate_list}
                optimality = True
                break  # is going to solve to optimality

        best_neighbor = None
        best_neighbor_obj = float('-inf')

        for i, current_aisles in enumerate(neighborhood):
            progress_bar('\nEvaluating neighborhood', i, len(neighborhood))
            if current_aisles in tabu_list:
                continue  # Basic tabu rule: skip recent moves

            current_aisles_data = [aisles[i] for i in current_aisles]
            x_sol, y_sol, ratio, reusable_model = solve_dinkelbach_2(
                orders=orders,
                selected_aisles=current_aisles_data,
                selected_aisle_indices=current_aisles,
                l_bound=l_bound,
                r_bound=r_bound,
                k=k,
                initial_lambda=0,
                warm_start_x=best_solution['x_sol'],
                warm_start_y=best_solution['y_sol'],
                model=reusable_model,
                all_aisles=aisles
            )
            if ratio is None: continue
            if ratio > best_neighbor_obj:
                used_aisles = set(i for i in current_aisles if y_sol[i] > 0.5)
                if best_neighbor is not None and used_aisles.issubset(best_solution['aisles_poll']):
                    continue  # avoid loops

                best_neighbor_obj = ratio
                best_neighbor = {
                    'aisles': tuple(used_aisles),
                    'aisles_poll': tuple(current_aisles),
                    'x_sol': tuple(x_sol),
                    'y_sol': tuple(y_sol),
                    'ratio': ratio
                }
        progress_bar('\nFinished evaluating neighborhood', 1, 1)
        print()

        fail_count += 1
        if best_neighbor is not None:
            best_objective_history.append(best_neighbor['ratio'])
            if best_neighbor_obj > best_solution['ratio']:
                fail_count = 0
                best_solution = best_neighbor
                if verbose:
                    print(f'    *** NUEVA MEJOR SOLUCIÓN GLOBAL: {best_neighbor_obj:.4f} ***')
                    verbose_print('Pasillos usados en mejor solución:', best_solution['aisles'])
                    print(f'    Warm start guardado para próxima iteración\n')

            current_aisle_indices = best_neighbor['aisles_poll']
            tabu_list.append(tuple(current_aisle_indices))

        if fail_count >= max_fail:
            neighborhood_width = min(ceil(2 * neighborhood_width), k)
            max_fail = max(ceil(0.5 * max_fail), 1)
            fail_count = 0

        if verbose:
            count = len(best_solution['aisles'])
            poll_count = len(best_solution['aisles_poll'])
            print(
                f'Finished iteration {iteration} with {count}/{poll_count} aisles and {fail_count}/{max_fail} fails in a row.')
            print(f'    Best obj={best_solution["ratio"]:.4f}')
            if best_neighbor is not None:
                print(f'    Best neighbor obj={best_neighbor["ratio"]:.4f}')
                count = len(best_neighbor['aisles'])
                poll_count = len(best_neighbor['aisles_poll'])
                print(f'    Best neighbor using {count}/{poll_count} aisles')
                print(f'    Neighbors size={len(best_neighbor["aisles_poll"])} ± {neighborhood_width}')
            if best_neighbor is None:
                print(f'    No valid solution in the neighbourhood.')

        if optimality: break  # solved to optimality

    total_time = time.time() - start_time
    if verbose:
        print(f'\n' + '=' * 70)
        print(f'BÚSQUEDA LOCAL CON MODELO FIJO COMPLETADA')
        print(f'Iteraciones realizadas: {iteration}')
        print(f'Tiempo total: {total_time:.2f}s ({total_time / 60:.1f} minutos)')
        print(f"Mejor λ encontrado: {best_solution['ratio']:.4f}")
        print(f'Optimizaciones aplicadas:')
        print(f'  ✓ Modelo FIJO reutilizado entre iteraciones')
        print(f'  ✓ Warm start con solución anterior')
        print(f'  ✓ Pedidos inválidos fijados a 0 (no eliminados)')
        print(f'\n*** Solved to optimality: {optimality} ***')

    if plot:
        plt.plot(best_objective_history, marker='o')
        plt.xlabel('Iteration')
        plt.ylabel('Objective Value')
        plt.title('Tabu Search Progress')
        plt.grid(True)
        plt.show()

    return best_solution['x_sol'], best_solution['y_sol'], best_solution['ratio']
