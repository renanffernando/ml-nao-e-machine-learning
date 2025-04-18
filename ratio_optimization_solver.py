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
        Resuelve el problema tratando de encontrar una solución con ratio >= target_ratio.
        
        Args:
            target_ratio: Valor objetivo mínimo que queremos alcanzar
            time_limit: Tiempo límite en segundos
            verbose: Si es True, mostrar información detallada
            previous_solution: Solución previa para warm start
            
        Returns:
            (selected_orders, visited_aisles) si se encuentra solución, None en caso contrario
        """
        start_time = time.time()
        
        # Crear el solver
        if verbose:
            print(f"Resolviendo con ratio objetivo >= {target_ratio:.2f}")
            if previous_solution:
                prev_orders, prev_aisles = previous_solution
                print(f"Usando warm start alternativo: {len(prev_orders)} órdenes, {len(prev_aisles)} pasillos")
            
        solver = pywraplp.Solver.CreateSolver('SCIP')
        if not solver:
            print("No se pudo crear el solver SCIP")
            return None
            
        # Aplicar límite de tiempo al solver (en milisegundos)
        # Dejamos un pequeño margen (0.5 segundos) para el procesamiento adicional
        solver_time_limit = int((time_limit - 0.5) * 1000) if time_limit > 1 else int(time_limit * 1000)
        solver_time_limit = max(1000, solver_time_limit)  # Al menos 1 segundo
        solver.SetTimeLimit(solver_time_limit)
        
        if verbose:
            print(f"Tiempo límite para el solver: {solver_time_limit/1000:.2f} segundos")
            
        # Variables: selección de órdenes y pasillos
        order_vars = {}
        for i in range(len(self.orders)):
            order_vars[i] = solver.BoolVar(f'order_{i}')
            
        aisle_vars = {}
        for j in range(len(self.aisles)):
            aisle_vars[j] = solver.BoolVar(f'aisle_{j}')
                    
        # Restricción: límites de tamaño de wave
        total_units = solver.Sum([self.order_units[i] * order_vars[i] for i in range(len(self.orders))])
        solver.Add(total_units >= self.wave_size_lb)
        solver.Add(total_units <= self.wave_size_ub)
        
        # Restricción: todos los items requeridos deben estar disponibles
        all_items = set()
        for order in self.orders:
            all_items.update(order.keys())
            
        for item in all_items:
            # Total requerido para este item
            required_expr = solver.Sum([
                self.orders[i].get(item, 0) * order_vars[i] 
                for i in range(len(self.orders)) 
                if item in self.orders[i]
            ])
            
            # Total disponible para este item
            available_expr = solver.Sum([
                self.aisles[j].get(item, 0) * aisle_vars[j]
                for j in range(len(self.aisles))
                if item in self.aisles[j]
            ])
            
            # Restricción: disponible >= requerido
            solver.Add(available_expr >= required_expr)
            
        # Restricción: si una orden está seleccionada, necesitamos visitar pasillos que tengan sus items
        for i in range(len(self.orders)):
            for item in self.orders[i]:
                # Pasillos que contienen este item
                relevant_aisles = [j for j in range(len(self.aisles)) if item in self.aisles[j]]
                if relevant_aisles:
                    solver.Add(
                        solver.Sum([aisle_vars[j] for j in relevant_aisles]) >= order_vars[i]
                    )
        
        # Asegurar que al menos un pasillo sea visitado si hay órdenes seleccionadas
        solver.Add(solver.Sum([aisle_vars[j] for j in range(len(self.aisles))]) >= 
                  solver.Sum([order_vars[i] for i in range(len(self.orders))]) / len(self.orders))
        
        # La clave: Restricción del ratio objetivo
        # Queremos: total_units / sum(aisle_vars) >= target_ratio
        # Equivalente a: total_units - target_ratio * sum(aisle_vars) >= 0
        total_aisles = solver.Sum([aisle_vars[j] for j in range(len(self.aisles))])
        solver.Add(total_units - target_ratio * total_aisles >= 0)
        
        # Objetivo: maximizar unidades y preferir seleccionar órdenes de la solución anterior
        objective = solver.Objective()
        
        # Objetivo principal: maximizar unidades
        for i in range(len(self.orders)):
            coefficient = self.order_units[i]
            
            # Si tenemos una solución previa, damos un pequeño bonus a las órdenes que estaban en ella
            if previous_solution:
                prev_orders, _ = previous_solution
                if i in prev_orders:
                    # Pequeño bonus para favorecer órdenes previas (0.01 * unidades)
                    coefficient += 0.01 * self.order_units[i]
                    
            objective.SetCoefficient(order_vars[i], coefficient)
        
        # Si tenemos solución previa, también preferimos los mismos pasillos
        if previous_solution:
            _, prev_aisles = previous_solution
            for j in range(len(self.aisles)):
                coef = 0
                if j in prev_aisles:
                    # Pequeño incentivo para reutilizar pasillos
                    coef = 0.1
                objective.SetCoefficient(aisle_vars[j], coef)
                
        objective.SetMaximization()
        
        # Resolver el modelo
        status = solver.Solve()
        
        # Verificar si se agotó el tiempo
        time_used = time.time() - start_time
        if verbose:
            print(f"Tiempo utilizado: {time_used:.2f} segundos")
        
        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            # Obtener órdenes seleccionadas
            selected_orders = [i for i in range(len(self.orders)) if order_vars[i].solution_value() > 0.5]
            
            # Obtener pasillos visitados
            visited_aisles = [j for j in range(len(self.aisles)) if aisle_vars[j].solution_value() > 0.5]
            
            # Calcular valor objetivo real
            total_units = sum(self.order_units[i] for i in selected_orders)
            objective_value = total_units / len(visited_aisles) if visited_aisles else 0
            
            if verbose:
                print(f"Solución encontrada: Valor={objective_value:.2f}, "
                      f"Órdenes={len(selected_orders)}, Pasillos={len(visited_aisles)}")
                print(f"Tiempo: {solver.WallTime()/1000:.2f} segundos")
                
            return selected_orders, visited_aisles
        else:
            if verbose:
                print(f"No hay solución factible con ratio >= {target_ratio:.2f}")
            return None
    
    def solve_with_binary_search(self, max_iterations=20, max_total_time=600, verbose=True, 
                              target_ratios=None, binary_search=True):
        """
        Resuelve el problema usando búsqueda binaria para maximizar el ratio.
        
        Args:
            max_iterations: Número máximo de iteraciones
            max_total_time: Tiempo máximo total en segundos
            verbose: Si es True, mostrar información detallada
            target_ratios: Lista de ratios específicos a probar antes de la búsqueda binaria
            binary_search: Si es True, realizar búsqueda binaria después de probar los target_ratios
            
        Returns:
            (selected_orders, visited_aisles, best_ratio) si se encuentra solución, None en caso contrario
        """
        start_time = time.time()
        iteration = 0
        best_solution = None
        best_ratio = 0
        
        # Para reportar progreso
        if verbose:
            progress = ProgressBar(max_iterations=max_iterations, 
                                max_time=max_total_time,
                                start_time=start_time,
                                description="Optimizando ratio")
                                
        # Calcular valor teórico máximo (para establecer límite superior)
        relax_upper_bound, _ = self.calculate_upper_bound()
        if verbose:
            print(f"Límite superior teórico (relajación LP): {relax_upper_bound:.2f}")
            
        # Generamos una solución inicial con algoritmo greedy para tener un lower bound
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
            initial_ratio = 0
        
        # Si tenemos una lista de ratios objetivo específicos, probamos primero estos
        if target_ratios:
            for target in target_ratios:
                # Verificar si tenemos suficiente tiempo restante
                time_elapsed = time.time() - start_time
                time_remaining = max_total_time - time_elapsed
                
                # Si queda menos de 30 segundos, terminamos
                if time_remaining < 30:
                    if verbose:
                        print(f"Tiempo restante insuficiente ({time_remaining:.2f}s). Terminando búsqueda.")
                    break
                
                if verbose:
                    print(f"\nIteración {iteration+1}: Probando ratio objetivo específico = {target:.2f}")
                    print(f"Tiempo restante: {time_remaining:.2f}s")
                
                # Intentar resolver con este ratio objetivo (usar todo el tiempo restante)
                time_limit = time_remaining - 10  # Dejamos 10 segundos de margen para terminar limpiamente
                
                # Intentar resolver con este ratio objetivo
                solution = self.solve_with_target_ratio(
                    target_ratio=target, 
                    time_limit=time_limit, 
                    verbose=verbose,
                    previous_solution=best_solution  # Pasar la mejor solución hasta ahora
                )
                
                iteration += 1
                
                if solution:
                    # Encontramos una solución factible
                    selected_orders, visited_aisles = solution
                    ratio = self.calculate_ratio(selected_orders, visited_aisles)
                    
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_solution = solution
                        
                    if verbose:
                        print(f"Solución factible encontrada con ratio = {ratio:.2f}")
                        print(f"Seleccionadas {len(selected_orders)} órdenes y {len(visited_aisles)} pasillos")
                        progress.update(iteration, best_ratio)
                else:
                    if verbose:
                        print(f"No se encontró solución factible para ratio >= {target:.2f}")
                        progress.update(iteration, best_ratio)
        
        # Si no queremos hacer búsqueda binaria, terminamos aquí
        if not binary_search:
            return best_solution[0], best_solution[1], best_ratio if best_solution else None
            
        # Hacer búsqueda binaria para maximizar el ratio
        # Límites inicial y final
        low_ratio = initial_ratio if initial_ratio > 0 else 0
        high_ratio = relax_upper_bound  # Usar exactamente el valor LP, sin margen extra
        
        while iteration < max_iterations:
            # Verificar si tenemos suficiente tiempo restante
            time_elapsed = time.time() - start_time
            time_remaining = max_total_time - time_elapsed
            
            # Si queda menos de 30 segundos, terminamos
            if time_remaining < 30:
                if verbose:
                    print(f"Tiempo restante insuficiente ({time_remaining:.2f}s). Terminando búsqueda.")
                break
            
            # Calcular punto medio
            mid_ratio = (low_ratio + high_ratio) / 2
            
            if verbose:
                print(f"\nIteración {iteration+1}: Probando ratio = {mid_ratio:.2f} [rango: {low_ratio:.2f} - {high_ratio:.2f}]")
                print(f"Tiempo restante: {time_remaining:.2f}s")
            
            # Usar todo el tiempo restante disponible, menos un pequeño margen
            time_limit = time_remaining - 10  # Dejamos 10 segundos de margen para terminar limpiamente
                
            # Intentar resolver con este ratio objetivo
            solution = self.solve_with_target_ratio(
                target_ratio=mid_ratio, 
                time_limit=time_limit, 
                verbose=verbose,
                previous_solution=best_solution  # Pasar la mejor solución hasta ahora
            )
            
            iteration += 1
            
            if solution:
                # Encontramos una solución factible
                selected_orders, visited_aisles = solution
                ratio = self.calculate_ratio(selected_orders, visited_aisles)
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_solution = solution
                    
                # Podemos buscar un ratio más alto
                low_ratio = mid_ratio
                
                if verbose:
                    print(f"Solución factible encontrada con ratio = {ratio:.2f}")
                    print(f"Seleccionadas {len(selected_orders)} órdenes y {len(visited_aisles)} pasillos")
            else:
                # No encontramos solución factible, reducir el ratio objetivo
                high_ratio = mid_ratio
                
                if verbose:
                    print(f"No se encontró solución factible para ratio >= {mid_ratio:.2f}")
            
            # Actualizar la barra de progreso
            if verbose:
                progress.update(iteration, best_ratio)
                
            # Condición de parada: si el rango es muy pequeño
            if high_ratio - low_ratio < 0.01:
                if verbose:
                    print(f"Convergencia alcanzada. Rango final: [{low_ratio:.2f} - {high_ratio:.2f}]")
                break
                
        # Finalizar y mostrar mejor solución encontrada
        if verbose:
            if best_solution:
                print(f"\nMejor solución encontrada: ratio = {best_ratio:.2f}")
                print(f"Seleccionadas {len(best_solution[0])} órdenes y {len(best_solution[1])} pasillos")
            else:
                print("\nNo se encontró ninguna solución factible")
                
        # Devolver la mejor solución o None si no se encontró ninguna
        if best_solution:
            return best_solution[0], best_solution[1], best_ratio
        else:
            return None
    
    def solve_with_fixed_aisles(self, num_aisles, time_limit=60, verbose=True):
        """
        Resuelve el problema fijando el número máximo de pasillos a utilizar
        
        Args:
            num_aisles: Número de pasillos a utilizar
            time_limit: Tiempo límite en segundos
            verbose: Si es True, mostrar información detallada
        """
        start_time = time.time()
        
        # Crear el solver
        if verbose:
            print(f"Resolviendo con número fijo de pasillos: {num_aisles}")
            
        solver = pywraplp.Solver.CreateSolver('SCIP')
        if not solver:
            print("No se pudo crear el solver SCIP")
            return None
            
        # NO ESTABLECER LÍMITE DE TIEMPO para la ejecución interna del solver
        
        # Variables: selección de órdenes y pasillos
        order_vars = {}
        for i in range(len(self.orders)):
            order_vars[i] = solver.BoolVar(f'order_{i}')
            
        aisle_vars = {}
        for j in range(len(self.aisles)):
            aisle_vars[j] = solver.BoolVar(f'aisle_{j}')
        
        # Restricción: número de pasillos exactamente igual a num_aisles
        solver.Add(solver.Sum([aisle_vars[j] for j in range(len(self.aisles))]) == num_aisles)
        
        # Restricción: límites de tamaño de wave
        total_units = solver.Sum([self.order_units[i] * order_vars[i] for i in range(len(self.orders))])
        solver.Add(total_units >= self.wave_size_lb)
        solver.Add(total_units <= self.wave_size_ub)
        
        # Restricción: todos los items requeridos deben estar disponibles
        all_items = set()
        for order in self.orders:
            all_items.update(order.keys())
            
        for item in all_items:
            # Total requerido para este item
            required_expr = solver.Sum([
                self.orders[i].get(item, 0) * order_vars[i] 
                for i in range(len(self.orders)) 
                if item in self.orders[i]
            ])
            
            # Total disponible para este item
            available_expr = solver.Sum([
                self.aisles[j].get(item, 0) * aisle_vars[j]
                for j in range(len(self.aisles))
                if item in self.aisles[j]
            ])
            
            # Restricción: disponible >= requerido
            solver.Add(available_expr >= required_expr)
            
        # Restricción: si una orden está seleccionada, necesitamos visitar pasillos que tengan sus items
        for i in range(len(self.orders)):
            for item in self.orders[i]:
                # Pasillos que contienen este item
                relevant_aisles = [j for j in range(len(self.aisles)) if item in self.aisles[j]]
                if relevant_aisles:
                    solver.Add(
                        solver.Sum([aisle_vars[j] for j in relevant_aisles]) >= order_vars[i]
                    )
        
        # Objetivo: maximizar unidades 
        objective = solver.Objective()
        for i in range(len(self.orders)):
            objective.SetCoefficient(order_vars[i], self.order_units[i])
        objective.SetMaximization()
        
        # Resolver el modelo
        status = solver.Solve()
        
        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            # Obtener órdenes seleccionadas
            selected_orders = [i for i in range(len(self.orders)) if order_vars[i].solution_value() > 0.5]
            
            # Obtener pasillos visitados
            visited_aisles = [j for j in range(len(self.aisles)) if aisle_vars[j].solution_value() > 0.5]
            
            # Calcular valor objetivo real
            total_units = sum(self.order_units[i] for i in selected_orders)
            objective_value = total_units / len(visited_aisles) if visited_aisles else 0
            
            if verbose:
                print(f"Solución con {num_aisles} pasillos: Ratio={objective_value:.2f}, "
                      f"Órdenes={len(selected_orders)}, Unidades={total_units}")
                      
            return selected_orders, visited_aisles
        else:
            if verbose:
                print(f"No hay solución factible con exactamente {num_aisles} pasillos")
            return None
    
    def _get_required_aisles(self, selected_orders):
        """
        Determina los pasillos necesarios para cubrir todos los items
        de las órdenes seleccionadas.
        """
        # Recolectar los items requeridos y sus cantidades
        required_items = {}
        for i in selected_orders:
            for item_id, qty in self.orders[i].items():
                required_items[item_id] = required_items.get(item_id, 0) + qty
        
        # Inicializar pasillos visitados y items pendientes
        visited_aisles = []
        remaining_items = required_items.copy()
        
        # Calcular cuántos items de los requeridos tiene cada pasillo
        aisle_coverage = []
        for j, aisle in enumerate(self.aisles):
            items_covered = 0
            for item_id, required_qty in remaining_items.items():
                if item_id in aisle:
                    items_covered += min(aisle[item_id], required_qty)
            aisle_coverage.append((j, items_covered))
        
        # Ordenar pasillos por cobertura de items (de mayor a menor)
        aisle_coverage.sort(key=lambda x: x[1], reverse=True)
        
        # Seleccionar pasillos hasta cubrir todos los items
        for j, _ in aisle_coverage:
            if not remaining_items:
                break
                
            # Ver si este pasillo proporciona algún item requerido
            items_provided = False
            for item_id in list(remaining_items.keys()):
                if item_id in self.aisles[j]:
                    qty_available = self.aisles[j][item_id]
                    if qty_available > 0:
                        items_provided = True
                        remaining_items[item_id] = max(0, remaining_items[item_id] - qty_available)
                        if remaining_items[item_id] == 0:
                            del remaining_items[item_id]
            
            if items_provided:
                visited_aisles.append(j)
                
        return visited_aisles
    
    def calculate_ratio(self, selected_orders, visited_aisles):
        """
        Calcula el ratio para una solución dada.
        
        Args:
            selected_orders: Lista de índices de órdenes seleccionadas
            visited_aisles: Lista de índices de pasillos visitados
            
        Returns:
            El ratio (unidades totales / pasillos visitados)
        """
        total_units = sum(self.order_units[i] for i in selected_orders)
        return total_units / len(visited_aisles) if visited_aisles else 0
        
    def write_solution(self, output_file, selected_orders, visited_aisles):
        """
        Escribe la solución en un archivo.
        
        Args:
            output_file: Nombre del archivo de salida
            selected_orders: Lista de índices de órdenes seleccionadas
            visited_aisles: Lista de índices de pasillos visitados
        """
        with open(output_file, 'w') as f:
            # Escribir número de órdenes y pasillos
            f.write(f"{len(selected_orders)} {len(visited_aisles)}\n")
            
            # Escribir órdenes
            f.write(' '.join(map(str, sorted(selected_orders))) + "\n")
            
            # Escribir pasillos
            f.write(' '.join(map(str, sorted(visited_aisles))) + "\n")

    def read_instance(self, input_file):
        """
        Lee los datos de una instancia desde un archivo.
        
        Args:
            input_file: Ruta al archivo de entrada
            
        Returns:
            (orders, aisles, order_units, order_aisles) tupla con los datos procesados
        """
        # Usar el método heredado read_input 
        self.read_input(input_file)
        
        # Preprocesar para calcular order_units y order_aisles
        self.preprocess_data()
        
        return self.orders, self.aisles, self.order_units, self.order_aisles
        
    def calculate_upper_bound(self):
        """
        Calcula un límite superior teórico usando relajación LP.
        
        Returns:
            (upper_bound, lp_solution) donde upper_bound es el valor teórico máximo
            y lp_solution es una solución de la relajación LP
        """
        # Crear solver LP
        solver_lp = pywraplp.Solver.CreateSolver('GLOP')
        
        if not solver_lp:
            print("Error: no se pudo crear el solver LP")
            return 0, None  # Fallback
        
        # Variables continuas (relajación LP)
        order_vars = {}
        for i in range(len(self.orders)):
            order_vars[i] = solver_lp.NumVar(0, 1, f'order_{i}')
            
        aisle_vars = {}
        for j in range(len(self.aisles)):
            aisle_vars[j] = solver_lp.NumVar(0, 1, f'aisle_{j}')
            
        # Restricciones
        # Límites de tamaño del wave
        total_units = solver_lp.Sum([self.order_units[i] * order_vars[i] for i in range(len(self.orders))])
        solver_lp.Add(total_units >= self.wave_size_lb)
        solver_lp.Add(total_units <= self.wave_size_ub)
        
        # Requerimientos de pasillos para cada orden
        for i in range(len(self.orders)):
            required_aisles = self.order_aisles[i]
            for aisle in required_aisles:
                solver_lp.Add(aisle_vars[aisle] >= order_vars[i])
                
        # Objetivo: maximizar ratio (aproximación lineal)
        # Maximizamos unidades y minimizamos pasillos
        objective = solver_lp.Objective()
        for i in range(len(self.orders)):
            objective.SetCoefficient(order_vars[i], self.order_units[i])
            
        for j in range(len(self.aisles)):
            objective.SetCoefficient(aisle_vars[j], -100)  # Peso negativo para minimizar pasillos
            
        objective.SetMaximization()
        
        # Resolver
        status = solver_lp.Solve()
        
        if status == pywraplp.Solver.OPTIMAL:
            # Extraer la solución
            lp_orders = [i for i in range(len(self.orders)) if order_vars[i].solution_value() > 0.5]
            lp_aisles = [j for j in range(len(self.aisles)) if aisle_vars[j].solution_value() > 0.5]
            
            # Calcular unidades en la solución LP
            lp_units = sum(self.order_units[i] * order_vars[i].solution_value() for i in range(len(self.orders)))
            
            # Calcular ratio teórico
            lp_ratio = lp_units / sum(aisle_vars[j].solution_value() for j in range(len(self.aisles)))
            
            # Retornar el ratio como upper bound y la solución
            return lp_ratio, (lp_orders, lp_aisles)
        else:
            # En caso de fallo, retornar un valor estimado
            print("No se pudo resolver la relajación LP para el límite superior")
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