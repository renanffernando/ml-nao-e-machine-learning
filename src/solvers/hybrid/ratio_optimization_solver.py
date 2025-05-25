import sys
import time
import random
import math
import threading
import numpy as np
from ortools.linear_solver import pywraplp
from checker import WaveOrderPicking
import json

# Para la barra de progreso
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Para ver una barra de progreso, instale tqdm: pip install tqdm")

class ProgressBar:
    """
    Clase para mostrar una barra de progreso en consola.
    Puede funcionar basada en tiempo o en iteraciones.
    """
    def __init__(self, max_iterations=None, max_time=None, start_time=None, 
                 description="Progreso"):
        """
        Inicializa la barra de progreso.
        
        Args:
            max_iterations: Número máximo de iteraciones (si se basa en iteraciones)
            max_time: Tiempo máximo en segundos (si se basa en tiempo)
            start_time: Tiempo de inicio (para modo basado en tiempo)
            description: Descripción de la barra de progreso
        """
        self.max_iterations = max_iterations
        self.max_time = max_time
        self.start_time = start_time or time.time()
        self.description = description
        self.last_update = self.start_time
        self.best_value = 0
        
        # Determinar si estamos basados en tiempo o iteraciones
        self.time_based = max_time is not None
        self.iter_based = max_iterations is not None
        
        # Si ninguno está definido, usar modo de tiempo por defecto
        if not self.time_based and not self.iter_based:
            self.time_based = True
            self.max_time = 600  # 10 minutos por defecto
            
        # Imprimir encabezado inicial
        if self.time_based:
            print(f"{self.description}: 0% [0/{self.max_time}s]")
        else:
            print(f"{self.description}: 0% [0/{self.max_iterations} iter]")
            
    def update(self, iteration=None, best_value=None):
        """
        Actualiza la barra de progreso.
        
        Args:
            iteration: Iteración actual (para modo basado en iteraciones)
            best_value: Mejor valor encontrado hasta ahora
        """
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        # Actualizar mejor valor si se proporciona
        if best_value is not None:
            self.best_value = best_value
            
        # Calcular porcentaje de progreso
        if self.time_based:
            if self.max_time > 0:
                progress = min(100, elapsed / self.max_time * 100)
            else:
                progress = 0
            time_info = f"{int(elapsed)}/{self.max_time}s"
        else:
            if self.max_iterations > 0 and iteration is not None:
                progress = min(100, iteration / self.max_iterations * 100)
            else:
                progress = 0
            time_info = f"{iteration or 0}/{self.max_iterations} iter ({int(elapsed)}s)"
            
        # Solo actualizar si ha pasado al menos 1 segundo desde la última actualización
        # o si es una actualización por iteración
        if current_time - self.last_update >= 1.0 or (not self.time_based and iteration is not None):
            self.last_update = current_time
            
            # Crear barra visual
            bar_length = 50
            filled_length = int(bar_length * progress / 100)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            
            # Mostrar progreso
            best_info = f", Mejor={self.best_value:.2f}" if self.best_value > 0 else ""
            print(f"\r{self.description}: {progress:.1f}% [{bar}] {time_info}{best_info}", end='')
            
            # Salto de línea si completado
            if progress >= 100:
                print()

class RatioOptimizationSolver(WaveOrderPicking):
    def __init__(self):
        super().__init__()
        self.order_units = None  # Unidades por orden
        self.item_locations = None  # Pasillos que contienen cada item
        self.cache = {}  # Caché para evaluaciones de soluciones
        self.best_solutions = []  # Para almacenar mejores soluciones encontradas
        
    def preprocess_data(self):
        """Preprocesa los datos para acelerar los cálculos"""
        # Calcular unidades por orden
        self.order_units = [sum(order.values()) for order in self.orders]
        
        # Para cada item, encontrar en qué pasillos está disponible
        self.item_locations = {}
        for item_id in set().union(*[set(order.keys()) for order in self.orders]):
            self.item_locations[item_id] = [
                j for j, aisle in enumerate(self.aisles) if item_id in aisle
            ]
            
        # Calcular qué pasillos requiere cada orden
        self.order_aisles = []
        for i, order in enumerate(self.orders):
            required_aisles = set()
            for item_id in order:
                # Agregar todos los pasillos que contienen este item
                if item_id in self.item_locations:
                    for aisle_idx in self.item_locations[item_id]:
                        required_aisles.add(aisle_idx)
            self.order_aisles.append(list(required_aisles))
    
    def solve_with_target_ratio(self, target_ratio, time_limit=60, verbose=True, previous_solution=None):
        """
        Solves the problem trying to find a solution with ratio >= target_ratio.
        
        Args:
            target_ratio: Minimum target value we want to achieve
            time_limit: Time limit in seconds
            verbose: If True, show detailed information
            previous_solution: Previous solution for warm start
            
        Returns:
            (selected_orders, visited_aisles) if solution found, None otherwise
        """
        start_time = time.time()
        
        # Create solver
        if verbose:
            print(f"Solving with target ratio >= {target_ratio:.2f}")
            if previous_solution:
                prev_orders, prev_aisles = previous_solution
                print(f"Using alternative warm start: {len(prev_orders)} orders, {len(prev_aisles)} aisles")
            
        solver = pywraplp.Solver.CreateSolver('SCIP')
        if not solver:
            print("Could not create SCIP solver")
            return None
            
        # Apply time limit to solver (in milliseconds)
        # Leave a small margin (0.5 seconds) for additional processing
        solver_time_limit = int((time_limit - 0.5) * 1000) if time_limit > 1 else int(time_limit * 1000)
        solver_time_limit = max(1000, solver_time_limit)  # At least 1 second
        solver.SetTimeLimit(solver_time_limit)
        
        if verbose:
            print(f"Time limit for solver: {solver_time_limit/1000:.2f} seconds")
            
        # Variables: order and aisle selection
        order_vars = {}
        for i in range(len(self.orders)):
            order_vars[i] = solver.BoolVar(f'order_{i}')
            
        aisle_vars = {}
        for j in range(len(self.aisles)):
            aisle_vars[j] = solver.BoolVar(f'aisle_{j}')
            
        # Wave size constraints
        total_units = solver.Sum([self.order_units[i] * order_vars[i] for i in range(len(self.orders))])
        solver.Add(total_units >= self.wave_size_lb)
        solver.Add(total_units <= self.wave_size_ub)
        
        # Ensure at least one aisle is visited if there are selected orders
        solver.Add(solver.Sum([aisle_vars[j] for j in range(len(self.aisles))]) >= 
                  solver.Sum([order_vars[i] for i in range(len(self.orders))]) / len(self.orders))
        
        # Key: Target ratio constraint
        # We want: total_units / sum(aisle_vars) >= target_ratio
        # Equivalent to: total_units - target_ratio * sum(aisle_vars) >= 0
        total_aisles = solver.Sum([aisle_vars[j] for j in range(len(self.aisles))])
        solver.Add(total_units - target_ratio * total_aisles >= 0)
        
        # Objective: maximize units and prefer selecting orders from previous solution
        objective = solver.Objective()
        
        # Main objective: maximize units
        for i in range(len(self.orders)):
            coefficient = self.order_units[i]
            
            # If we have a previous solution, give a small bonus to orders that were in it
            if previous_solution:
                prev_orders, _ = previous_solution
                if i in prev_orders:
                    # Small bonus to favor previous orders (0.01 * units)
                    coefficient += 0.01 * self.order_units[i]
                    
            objective.SetCoefficient(order_vars[i], coefficient)
        
        # If we have previous solution, also prefer same aisles
        if previous_solution:
            _, prev_aisles = previous_solution
            for j in range(len(self.aisles)):
                coef = 0
                if j in prev_aisles:
                    # Small incentive to reuse aisles
                    coef = 0.1
                objective.SetCoefficient(aisle_vars[j], coef)
                
        objective.SetMaximization()
        
        # Solve the model
        status = solver.Solve()
        
        # Check if time ran out
        time_used = time.time() - start_time
        if verbose:
            print(f"Time used: {time_used:.2f} seconds")
        
        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            # Get selected orders
            selected_orders = [i for i in range(len(self.orders)) if order_vars[i].solution_value() > 0.5]
            
            # Get visited aisles
            visited_aisles = [j for j in range(len(self.aisles)) if aisle_vars[j].solution_value() > 0.5]
            
            # Calculate actual objective value
            total_units = sum(self.order_units[i] for i in selected_orders)
            objective_value = total_units / len(visited_aisles) if visited_aisles else 0
            
            if verbose:
                print(f"Solution found: Value={objective_value:.2f}, "
                      f"Orders={len(selected_orders)}, Aisles={len(visited_aisles)}")
                print(f"Time: {solver.WallTime()/1000:.2f} seconds")
                
            return selected_orders, visited_aisles
        else:
            if verbose:
                print(f"No feasible solution with ratio >= {target_ratio:.2f}")
            return None
    
    def solve_with_binary_search(self, max_iterations=20, max_total_time=600, verbose=True, 
                              target_ratios=None, binary_search=True):
        """
        Solves the problem using binary search to maximize the ratio.
        
        Args:
            max_iterations: Maximum number of iterations
            max_total_time: Maximum total time in seconds
            verbose: If True, show detailed information
            target_ratios: List of specific ratios to try before binary search
            binary_search: If True, perform binary search after trying target_ratios
            
        Returns:
            (selected_orders, visited_aisles, best_ratio) if solution found, None otherwise
        """
        start_time = time.time()
        iteration = 0
        best_solution = None
        best_ratio = 0
        
        # For progress reporting
        if verbose:
            progress = ProgressBar(max_iterations=max_iterations, 
                                max_time=max_total_time,
                                start_time=start_time,
                                description="Optimizing ratio")
                                
        # Calculate theoretical maximum value (to set upper bound)
        relax_upper_bound, _ = self.calculate_upper_bound()
        if verbose:
            print(f"Theoretical upper bound (LP relaxation): {relax_upper_bound:.2f}")
            
        # Generate an initial solution with greedy algorithm to have a lower bound
        initial_solution = self.generate_greedy_solution()
        
        if initial_solution:
            selected_orders, visited_aisles = initial_solution
            initial_ratio = sum(self.order_units[i] for i in selected_orders) / len(visited_aisles)
            if verbose:
                print(f"Initial greedy solution: ratio = {initial_ratio:.2f}")
            best_solution = initial_solution
            best_ratio = initial_ratio
            
        # Try specific target ratios first
        if target_ratios:
            for target in target_ratios:
                # Check if we have enough time left
                time_elapsed = time.time() - start_time
                time_remaining = max_total_time - time_elapsed
                
                if time_remaining < 30:  # Need at least 30 seconds
                    if verbose:
                        print(f"Insufficient remaining time ({time_remaining:.2f}s). Stopping search.")
                    break
                    
                if verbose:
                    print(f"\nTrying target ratio = {target:.2f}")
                    
                # Try to solve with this target ratio
                solution = self.solve_with_target_ratio(
                    target_ratio=target,
                    time_limit=min(time_remaining - 10, 120),  # Max 120s per attempt
                    verbose=verbose,
                    previous_solution=best_solution
                )
                
                if solution:
                    selected_orders, visited_aisles = solution
                    ratio = sum(self.order_units[i] for i in selected_orders) / len(visited_aisles)
                    
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_solution = solution
                        
                iteration += 1
                if verbose:
                    progress.update(iteration)
                    
        if not binary_search:
            if verbose:
                progress.stop()
            return best_solution[0], best_solution[1], best_ratio if best_solution else None
            
        # Binary search phase
        if verbose:
            print("\nStarting binary search phase")
            
        # Initialize bounds
        low_ratio = best_ratio if best_ratio > 0 else 0
        high_ratio = relax_upper_bound  # Use exactly the LP value, no extra margin
        
        while iteration < max_iterations:
            # Check if we have enough time left
            time_elapsed = time.time() - start_time
            time_remaining = max_total_time - time_elapsed
            
            # If less than 30 seconds remain, finish
            if time_remaining < 30:
                if verbose:
                    print(f"Insufficient remaining time ({time_remaining:.2f}s). Stopping search.")
                break
            
            # Calculate midpoint
            mid_ratio = (low_ratio + high_ratio) / 2
            
            if verbose:
                print(f"\nIteration {iteration+1}: Trying ratio = {mid_ratio:.2f} [range: {low_ratio:.2f} - {high_ratio:.2f}]")
                print(f"Time remaining: {time_remaining:.2f}s")
            
            # Use all remaining time available, minus a small margin
            time_limit = time_remaining - 10  # Leave 10 seconds margin to finish cleanly
                
            # Try to solve with this target ratio
            solution = self.solve_with_target_ratio(
                target_ratio=mid_ratio, 
                time_limit=time_limit, 
                verbose=verbose,
                previous_solution=best_solution  # Pass best solution so far
            )
            
            iteration += 1
            if verbose:
                progress.update(iteration)
            
            if solution:
                # Solution found, try higher ratio
                selected_orders, visited_aisles = solution
                ratio = sum(self.order_units[i] for i in selected_orders) / len(visited_aisles)
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_solution = solution
                    
                # Continue searching in upper half
                low_ratio = mid_ratio
            else:
                # No solution found, try lower ratio
                high_ratio = mid_ratio
                
            # If bounds are too close, stop
            if high_ratio - low_ratio < 0.01:
                if verbose:
                    print(f"Bounds too close ({high_ratio:.3f} - {low_ratio:.3f} < 0.01). Stopping search.")
                break
                
        if verbose:
            progress.stop()
            
        # Return best solution found
        if best_solution:
            return best_solution[0], best_solution[1], best_ratio
        else:
            return None
            
    def calculate_upper_bound(self):
        """
        Calculates a theoretical upper bound using LP relaxation.
        
        Returns:
            (upper_bound, lp_solution) where upper_bound is the maximum theoretical value
            and lp_solution is a solution from the LP relaxation
        """
        # Create LP solver
        solver_lp = pywraplp.Solver.CreateSolver('GLOP')
        
        if not solver_lp:
            print("Error: could not create LP solver")
            return 0, None  # Fallback
        
        # Continuous variables (LP relaxation)
        order_vars = {}
        for i in range(len(self.orders)):
            order_vars[i] = solver_lp.NumVar(0, 1, f'order_{i}')
            
        aisle_vars = {}
        for j in range(len(self.aisles)):
            aisle_vars[j] = solver_lp.NumVar(0, 1, f'aisle_{j}')
            
        # Constraints
        # Wave size limits
        total_units = solver_lp.Sum([self.order_units[i] * order_vars[i] for i in range(len(self.orders))])
        solver_lp.Add(total_units >= self.wave_size_lb)
        solver_lp.Add(total_units <= self.wave_size_ub)
        
        # Aisle requirements for each order
        for i in range(len(self.orders)):
            required_aisles = self.order_aisles[i]
            for aisle in required_aisles:
                solver_lp.Add(aisle_vars[aisle] >= order_vars[i])
                
        # Objective: maximize ratio (linear approximation)
        # Maximize units and minimize aisles
        objective = solver_lp.Objective()
        for i in range(len(self.orders)):
            objective.SetCoefficient(order_vars[i], self.order_units[i])
            
        for j in range(len(self.aisles)):
            objective.SetCoefficient(aisle_vars[j], -100)  # Negative weight to minimize aisles
            
        objective.SetMaximization()
        
        # Solve
        status = solver_lp.Solve()
        
        if status == solver_lp.OPTIMAL:
            # Extract solution
            lp_orders = [i for i in range(len(self.orders)) if order_vars[i].solution_value() > 0.5]
            lp_aisles = [j for j in range(len(self.aisles)) if aisle_vars[j].solution_value() > 0.5]
            
            # Calculate units in LP solution
            lp_units = sum(self.order_units[i] * order_vars[i].solution_value() for i in range(len(self.orders)))
            
            # Calculate theoretical ratio
            lp_ratio = lp_units / sum(aisle_vars[j].solution_value() for j in range(len(self.aisles)))
            
            # Return ratio as upper bound and solution
            return lp_ratio, (lp_orders, lp_aisles)
        else:
            # In case of failure, return estimated value
            print("Could not solve LP relaxation for upper bound")
            return self.wave_size_ub, None

    def generate_greedy_solution(self):
        """
        Genera una solución inicial factible usando un enfoque greedy.
        
        Returns:
            (selected_orders, visited_aisles) si se encuentra solución, None en caso contrario
        """
        # Ordenar órdenes por densidad (unidades / número de pasillos requeridos)
        densities = []
        for i in range(len(self.orders)):
            num_aisles = len(self.order_aisles[i]) if self.order_aisles[i] else 1
            density = self.order_units[i] / num_aisles
            densities.append((i, density))
        
        # Ordenar por densidad (de mayor a menor)
        sorted_orders = [i for i, _ in sorted(densities, key=lambda x: x[1], reverse=True)]
        
        # Seleccionar órdenes hasta alcanzar límite mínimo
        selected_orders = []
        total_units = 0
        
        # Primera fase: llegar al mínimo requerido
        for i in sorted_orders:
            if total_units < self.wave_size_lb:
                selected_orders.append(i)
                total_units += self.order_units[i]
            else:
                break
                
        # Segunda fase: agregar más órdenes si es posible (sin exceder el máximo)
        for i in sorted_orders:
            if i not in selected_orders:
                if total_units + self.order_units[i] <= self.wave_size_ub:
                    selected_orders.append(i)
                    total_units += self.order_units[i]
                    
        # Obtener los pasillos necesarios
        visited_aisles = self._get_required_aisles(selected_orders)
        
        # Verificar que la solución es factible
        if total_units >= self.wave_size_lb and total_units <= self.wave_size_ub and visited_aisles:
            # Solución factible
            return selected_orders, visited_aisles
        else:
            # No se encontró solución factible
            return None

    def improve_solution_locally(self, initial_solution, max_time=300, verbose=True):
        """
        Mejora una solución mediante búsqueda local, intentando maximizar el ratio.
        
        Args:
            initial_solution: Tupla (selected_orders, visited_aisles) con la solución inicial
            max_time: Tiempo máximo en segundos
            verbose: Si es True, mostrar información detallada
            
        Returns:
            (selected_orders, visited_aisles, ratio) con la solución mejorada
        """
        start_time = time.time()
        
        # Extraer la solución inicial
        curr_orders, curr_aisles = initial_solution
        curr_orders = set(curr_orders)  # Convertir a conjunto para operaciones más rápidas
        curr_aisles = set(curr_aisles)
        
        # Calcular ratio actual
        curr_units = sum(self.order_units[i] for i in curr_orders)
        curr_ratio = curr_units / len(curr_aisles) if curr_aisles else 0
        
        if verbose:
            print(f"\nIniciando mejora local. Ratio inicial = {curr_ratio:.2f}")
            print(f"Órdenes iniciales: {len(curr_orders)}, Pasillos iniciales: {len(curr_aisles)}")
        
        # Registrar la mejor solución
        best_orders = set(curr_orders)
        best_aisles = set(curr_aisles)
        best_ratio = curr_ratio
        
        # Para seguimiento de mejoras
        iterations = 0
        last_improvement = 0
        
        # Generar mapeo de órdenes a pasillos necesarios
        order_to_aisles = {}
        for i in range(len(self.orders)):
            needed_aisles = set()
            for item, qty in self.orders[i].items():
                for j in range(len(self.aisles)):
                    if item in self.aisles[j]:
                        needed_aisles.add(j)
            order_to_aisles[i] = needed_aisles
        
        # Generar mapeo de pasillos a órdenes que los necesitan
        aisle_to_orders = {}
        for j in range(len(self.aisles)):
            dependent_orders = set()
            for i in range(len(self.orders)):
                if j in order_to_aisles[i]:
                    dependent_orders.add(i)
            aisle_to_orders[j] = dependent_orders
        
        # Calcular el valor de cada orden (unidades / pasillos adicionales)
        def calculate_order_value(order_idx, current_aisles):
            order_units = self.order_units[order_idx]
            needed_aisles = order_to_aisles[order_idx]
            additional_aisles = len(needed_aisles - current_aisles)
            if additional_aisles == 0:
                return float('inf')  # Valor infinito si no necesita pasillos adicionales
            return order_units / additional_aisles
        
        # Calcular el valor de cada pasillo (unidades perdidas / pasillos ahorrados)
        def calculate_aisle_value(aisle_idx, current_aisles, current_orders):
            # Órdenes que dependen exclusivamente de este pasillo
            exclusive_dependent_orders = set()
            for order_idx in current_orders:
                if aisle_idx in order_to_aisles[order_idx]:
                    # Verificar si algún otro pasillo en current_aisles puede satisfacer esta orden
                    can_be_satisfied = False
                    for item, qty in self.orders[order_idx].items():
                        if item in self.aisles[aisle_idx]:  # Si este pasillo proporciona el item
                            # Verificar si otro pasillo lo proporciona
                            for other_aisle in current_aisles:
                                if other_aisle != aisle_idx and item in self.aisles[other_aisle]:
                                    can_be_satisfied = True
                                    break
                            if not can_be_satisfied:
                                break
                    if not can_be_satisfied:
                        exclusive_dependent_orders.add(order_idx)
            
            # Si hay órdenes que dependen exclusivamente de este pasillo, no podemos eliminarlo
            if exclusive_dependent_orders:
                return -float('inf')
            
            # Calcular unidades perdidas si eliminamos el pasillo
            lost_units = 0
            for order_idx in aisle_to_orders[aisle_idx].intersection(current_orders):
                # Verificar si la orden solo puede ser satisfecha por este pasillo
                order_depends_exclusively = False
                for item, qty in self.orders[order_idx].items():
                    if item in self.aisles[aisle_idx]:
                        item_covered = False
                        for other_aisle in current_aisles:
                            if other_aisle != aisle_idx and item in self.aisles[other_aisle]:
                                item_covered = True
                                break
                        if not item_covered:
                            order_depends_exclusively = True
                            break
                
                if order_depends_exclusively:
                    lost_units += self.order_units[order_idx]
            
            # Valor es negativo si perdemos unidades (queremos maximizar)
            if lost_units > 0:
                return -lost_units
            else:
                return 1  # Valor positivo si no perdemos unidades
        
        # Función para verificar si una solución es factible
        def is_feasible(orders, aisles):
            # Verificar tamaño total de wave
            total_units = sum(self.order_units[i] for i in orders)
            if total_units < self.wave_size_lb or total_units > self.wave_size_ub:
                return False
            
            # Verificar que todas las órdenes tienen sus ítems disponibles en los pasillos seleccionados
            for order_idx in orders:
                for item, qty in self.orders[order_idx].items():
                    item_available = False
                    for aisle_idx in aisles:
                        if item in self.aisles[aisle_idx]:
                            item_available = True
                            break
                    if not item_available:
                        return False
            
            return True
        
        # Bucle principal de búsqueda local
        while time.time() - start_time < max_time:
            iterations += 1
            improved = False
            
            # 1. Intentar agregar órdenes beneficiosas
            candidate_orders = sorted(
                [i for i in range(len(self.orders)) if i not in curr_orders],
                key=lambda x: calculate_order_value(x, curr_aisles),
                reverse=True  # Mayor valor primero
            )
            
            for order_idx in candidate_orders[:50]:  # Probar las 50 órdenes más prometedoras
                # Calcular pasillos adicionales necesarios
                needed_aisles = order_to_aisles[order_idx]
                new_aisles = curr_aisles.union(needed_aisles)
                
                # Verificar si agregar esta orden mejora el ratio
                new_orders = curr_orders.union({order_idx})
                new_units = curr_units + self.order_units[order_idx]
                new_ratio = new_units / len(new_aisles) if new_aisles else 0
                
                # Verificar si mejora el ratio y es factible
                if new_ratio > curr_ratio:
                    # Verificar límites de tamaño de wave
                    if self.wave_size_lb <= new_units <= self.wave_size_ub:
                        curr_orders = new_orders
                        curr_aisles = new_aisles
                        curr_units = new_units
                        curr_ratio = new_ratio
                        improved = True
                        
                        if curr_ratio > best_ratio:
                            best_orders = set(curr_orders)
                            best_aisles = set(curr_aisles)
                            best_ratio = curr_ratio
                            last_improvement = iterations
                            
                            if verbose and iterations % 10 == 0:
                                print(f"Iteración {iterations}: Ratio mejorado a {best_ratio:.2f} "
                                      f"({len(best_orders)} órdenes, {len(best_aisles)} pasillos)")
                
                # Si hemos estado mejorando continuamente, no necesitamos probar todas las órdenes
                if improved and iterations - last_improvement < 5:
                    break
            
            # 2. Intentar eliminar pasillos poco utilizados
            if not improved:
                candidate_aisles = sorted(
                    list(curr_aisles),
                    key=lambda x: calculate_aisle_value(x, curr_aisles, curr_orders),
                    reverse=True  # Mayor valor primero (menos críticos)
                )
                
                for aisle_idx in candidate_aisles[:20]:  # Probar los 20 pasillos menos críticos
                    # Verificar si podemos eliminar este pasillo
                    new_aisles = curr_aisles - {aisle_idx}
                    
                    # No podemos eliminar todos los pasillos
                    if not new_aisles:
                        continue
                    
                    # Identificar órdenes que ya no podemos satisfacer
                    unsatisfiable_orders = set()
                    for order_idx in curr_orders:
                        for item, qty in self.orders[order_idx].items():
                            # Si este item estaba en el pasillo que eliminamos
                            if item in self.aisles[aisle_idx]:
                                # Verificar si hay otro pasillo que lo proporcione
                                item_covered = False
                                for other_aisle in new_aisles:
                                    if item in self.aisles[other_aisle]:
                                        item_covered = True
                                        break
                                
                                if not item_covered:
                                    unsatisfiable_orders.add(order_idx)
                                    break
                    
                    # Calcular nuevo conjunto de órdenes y unidades
                    new_orders = curr_orders - unsatisfiable_orders
                    
                    # Si eliminamos todas las órdenes, no es una buena movida
                    if not new_orders:
                        continue
                    
                    new_units = sum(self.order_units[i] for i in new_orders)
                    
                    # Verificar límites de tamaño de wave
                    if new_units < self.wave_size_lb:
                        continue
                    
                    new_ratio = new_units / len(new_aisles)
                    
                    # Verificar si mejora el ratio
                    if new_ratio > curr_ratio:
                        curr_orders = new_orders
                        curr_aisles = new_aisles
                        curr_units = new_units
                        curr_ratio = new_ratio
                        improved = True
                        
                        if curr_ratio > best_ratio:
                            best_orders = set(curr_orders)
                            best_aisles = set(curr_aisles)
                            best_ratio = curr_ratio
                            last_improvement = iterations
                            
                            if verbose and iterations % 10 == 0:
                                print(f"Iteración {iterations}: Ratio mejorado a {best_ratio:.2f} "
                                      f"({len(best_orders)} órdenes, {len(best_aisles)} pasillos)")
                        
                        # Si mejoramos, no necesitamos seguir probando
                        break
            
            # 3. Realizar algunos intercambios aleatorios para escapar de óptimos locales
            if not improved and iterations - last_improvement > 20:
                # Seleccionar algunas órdenes aleatoriamente para eliminar
                if len(curr_orders) > 10:
                    orders_to_remove = random.sample(list(curr_orders), min(10, len(curr_orders) // 10))
                    
                    new_orders = curr_orders - set(orders_to_remove)
                    new_units = sum(self.order_units[i] for i in new_orders)
                    
                    # Intentar agregar órdenes beneficiosas en su lugar
                    candidate_orders = sorted(
                        [i for i in range(len(self.orders)) if i not in new_orders],
                        key=lambda x: calculate_order_value(x, curr_aisles),
                        reverse=True
                    )
                    
                    for order_idx in candidate_orders[:30]:
                        potential_new_units = new_units + self.order_units[order_idx]
                        if self.wave_size_lb <= potential_new_units <= self.wave_size_ub:
                            # Verificar si podemos agregar esta orden con los pasillos actuales
                            can_add = True
                            for item, qty in self.orders[order_idx].items():
                                item_covered = False
                                for aisle_idx in curr_aisles:
                                    if item in self.aisles[aisle_idx]:
                                        item_covered = True
                                        break
                                if not item_covered:
                                    can_add = False
                                    break
                            
                            if can_add:
                                new_orders.add(order_idx)
                                new_units = potential_new_units
                    
                    # Recalcular pasillos necesarios
                    new_aisles = set()
                    for order_idx in new_orders:
                        for item, qty in self.orders[order_idx].items():
                            for aisle_idx in range(len(self.aisles)):
                                if item in self.aisles[aisle_idx]:
                                    new_aisles.add(aisle_idx)
                                    break
                    
                    new_ratio = new_units / len(new_aisles) if new_aisles else 0
                    
                    # Actualizar si mejora o para diversificar (con cierta probabilidad)
                    if new_ratio > curr_ratio or random.random() < 0.3:
                        curr_orders = new_orders
                        curr_aisles = new_aisles
                        curr_units = new_units
                        curr_ratio = new_ratio
                        
                        if curr_ratio > best_ratio:
                            best_orders = set(curr_orders)
                            best_aisles = set(curr_aisles)
                            best_ratio = curr_ratio
                            last_improvement = iterations
            
            # Condición de parada: si no hemos mejorado en muchas iteraciones
            if iterations - last_improvement > 50:
                if verbose:
                    print(f"Sin mejoras en 50 iteraciones. Terminando búsqueda local.")
                break
        
        elapsed_time = time.time() - start_time
        
        if verbose:
            print(f"\nMejora local completada en {elapsed_time:.2f} segundos")
            print(f"Iteraciones: {iterations}")
            print(f"Ratio final: {best_ratio:.2f}")
            print(f"Órdenes: {len(best_orders)} / Pasillos: {len(best_aisles)}")
        
        return list(best_orders), list(best_aisles), best_ratio

    def solve_with_hybrid_approach(self, max_total_time=600, verbose=True):
        """
        Resuelve el problema usando un enfoque híbrido:
        1. Primero usa búsqueda binaria para encontrar una solución inicial rápida
        2. Luego mejora esta solución con búsqueda local
        
        Args:
            max_total_time: Tiempo máximo total en segundos
            verbose: Si es True, mostrar información detallada
            
        Returns:
            (selected_orders, visited_aisles, best_ratio) con la mejor solución encontrada
        """
        start_time = time.time()
        
        if verbose:
            print("\nIniciando enfoque híbrido de optimización...")
            
        # Calcular límite superior teórico
        relax_upper_bound, _ = self.calculate_upper_bound()
        if verbose:
            print(f"Límite superior teórico (relajación LP): {relax_upper_bound:.2f}")
            
        # Fase 1: Obtener solución inicial con búsqueda binaria rápida (30% del tiempo)
        binary_search_time = max_total_time * 0.3
        phase1_end_time = start_time + binary_search_time
        
        if verbose:
            print(f"\nFase 1: Búsqueda binaria rápida (máximo {binary_search_time:.2f} segundos)")
        
        # Generar solución greedy como punto de partida
        initial_solution = self.generate_greedy_solution()
        if initial_solution:
            initial_orders, initial_aisles = initial_solution
            initial_ratio = self.calculate_ratio(initial_orders, initial_aisles)
            if verbose:
                print(f"Solución greedy inicial: ratio = {initial_ratio:.2f}, "
                     f"{len(initial_orders)} órdenes, {len(initial_aisles)} pasillos")
            best_solution = initial_solution
            best_ratio = initial_ratio
        else:
            print("No se pudo generar solución greedy inicial")
            return None
        
        # Realizar una búsqueda binaria rápida (pocas iteraciones)
        # Límites de búsqueda
        low_ratio = initial_ratio
        high_ratio = relax_upper_bound
        binary_iterations = 0
        
        # Loop hasta que se agote el tiempo o hagamos suficientes iteraciones
        while time.time() < phase1_end_time and binary_iterations < 3:
            # Calcular punto medio
            mid_ratio = (low_ratio + high_ratio) / 2
            
            if verbose:
                time_elapsed = time.time() - start_time
                time_remaining = binary_search_time - time_elapsed
                print(f"\nBúsqueda binaria - Iteración {binary_iterations+1}: Probando ratio = {mid_ratio:.2f}")
                print(f"Tiempo restante para fase 1: {time_remaining:.2f}s")
            
            # Tiempo disponible para esta iteración
            iter_time_limit = phase1_end_time - time.time() - 1  # 1 segundo de margen
            iter_time_limit = max(10, iter_time_limit)  # Al menos 10 segundos
            
            # Resolver con este ratio objetivo
            solution = self.solve_with_target_ratio(
                target_ratio=mid_ratio,
                time_limit=iter_time_limit,
                verbose=verbose,
                previous_solution=best_solution
            )
            
            binary_iterations += 1
            
            if solution:
                # Solución factible, podemos buscar un ratio mayor
                selected_orders, visited_aisles = solution
                ratio = self.calculate_ratio(selected_orders, visited_aisles)
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_solution = solution
                
                low_ratio = mid_ratio
                
                if verbose:
                    print(f"Solución factible encontrada con ratio = {ratio:.2f}")
                    print(f"Seleccionadas {len(selected_orders)} órdenes y {len(visited_aisles)} pasillos")
            else:
                # No hay solución factible, reducir el ratio
                high_ratio = mid_ratio
                
                if verbose:
                    print(f"No se encontró solución factible para ratio >= {mid_ratio:.2f}")
        
        if verbose:
            print(f"\nFase 1 completada. Mejor ratio encontrado: {best_ratio:.2f}")
            print(f"Tiempo usado en fase 1: {time.time() - start_time:.2f} segundos")
        
        # Fase 2: Mejorar la solución con búsqueda local (70% del tiempo restante)
        time_elapsed = time.time() - start_time
        local_search_time = max_total_time - time_elapsed - 1  # 1 segundo de margen
        
        if verbose:
            print(f"\nFase 2: Búsqueda local (máximo {local_search_time:.2f} segundos)")
            
        # Si tenemos una solución de la fase 1, la mejoramos
        if best_solution:
            improved_orders, improved_aisles, improved_ratio = self.improve_solution_locally(
                initial_solution=best_solution,
                max_time=local_search_time,
                verbose=verbose
            )
            
            # Verificar si mejoramos
            if improved_ratio > best_ratio:
                best_solution = (improved_orders, improved_aisles)
                best_ratio = improved_ratio
                
        # Devolver la mejor solución encontrada
        if best_solution:
            return best_solution[0], best_solution[1], best_ratio
        else:
            return None

if __name__ == "__main__":
    # Verificar argumentos
    if len(sys.argv) < 3:
        print("Uso: python3 ratio_optimization_solver.py <archivo_instancia> <archivo_salida>")
        sys.exit(1)
        
    instance_file = sys.argv[1]
    output_file = sys.argv[2]
    
    print(f"Resolviendo instancia {instance_file}")
    print(f"Guardando solución en {output_file}")
    
    # Crear instancia del solver
    solver = RatioOptimizationSolver()
    
    # Cargar datos
    print("Cargando datos...")
    orders, aisles, order_units, order_aisles = solver.read_instance(instance_file)
    
    # Resolver usando búsqueda binaria
    print("\nIniciando optimización...")
    start_time = time.time()
    
    # Preprocesar datos
    solver.preprocess_data()
    
    # Calcular cota superior teórica
    relax_upper_bound, _ = solver.calculate_upper_bound()
    print(f"Límite superior teórico (relajación LP): {relax_upper_bound:.2f}")
    
    # Resolver usando el nuevo enfoque híbrido (10 minutos máximo total)
    result = solver.solve_with_hybrid_approach(
        max_total_time=600,  # 10 minutos total
        verbose=True
    )
    
    if result:
        selected_orders, visited_aisles, best_ratio = result
        
        # Verificar la solución
        units = sum(order_units[i] for i in selected_orders)
        ratio = units / len(visited_aisles) if visited_aisles else 0
        
        print(f"\nSolución encontrada:")
        print(f"  Órdenes seleccionadas: {len(selected_orders)}")
        print(f"  Pasillos visitados: {len(visited_aisles)}")
        print(f"  Unidades totales: {units}")
        print(f"  Ratio: {ratio:.2f}")
        
        # Guardar solución
        print(f"\nGuardando solución en {output_file}...")
        solver.write_solution(output_file, selected_orders, visited_aisles)
        
        # Tiempo total
        total_time = time.time() - start_time
        print(f"Tiempo total de procesamiento: {total_time:.2f} segundos")
    else:
        print("\nNo se encontró solución factible.")
        # Crear archivo vacío
        with open(output_file, 'w') as f:
            f.write("No se encontró una solución factible\n") 