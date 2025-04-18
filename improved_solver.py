import sys
import time
import random
import math
import threading
import numpy as np
from ortools.linear_solver import pywraplp
from checker import WaveOrderPicking

# Para la barra de progreso
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Para ver una barra de progreso, instale tqdm: pip install tqdm")

class ProgressBar:
    def __init__(self, total_seconds=600):
        self.total_seconds = total_seconds
        self.start_time = time.time()
        self.stop_flag = False
        self.current_info = ""
        
    def start(self):
        if TQDM_AVAILABLE:
            self.pbar = tqdm(total=self.total_seconds, desc="Tiempo", unit="s")
            self.thread = threading.Thread(target=self._update_progress)
            self.thread.daemon = True
            self.thread.start()
        else:
            print(f"Tiempo límite: {self.total_seconds} segundos")
            
    def _update_progress(self):
        last_elapsed = 0
        while not self.stop_flag and last_elapsed < self.total_seconds:
            elapsed = min(int(time.time() - self.start_time), self.total_seconds)
            if elapsed > last_elapsed:
                self.pbar.update(elapsed - last_elapsed)
                last_elapsed = elapsed
                if self.current_info:
                    self.pbar.set_postfix_str(self.current_info)
            time.sleep(0.1)
        
    def update_info(self, info):
        self.current_info = info
        if not TQDM_AVAILABLE:
            elapsed = int(time.time() - self.start_time)
            print(f"[{elapsed}s/{self.total_seconds}s] {info}")
            
    def stop(self):
        self.stop_flag = True
        if TQDM_AVAILABLE:
            self.pbar.close()

class ImprovedSolver(WaveOrderPicking):
    def __init__(self):
        super().__init__()
        self.order_units = None  # Unidades por orden
        self.item_locations = None  # Pasillos que contienen cada item
        self.cache = {}  # Caché para evaluaciones de soluciones
        self.best_solutions = []  # Lista de mejores soluciones encontradas
        
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
    
    def generate_initial_solution_mip(self, time_limit=30, verbose=True, focus_on_units=False, 
                                     objective_type="ratio", min_orders=None):
        """
        Genera una solución inicial usando programación lineal entera mixta
        
        Args:
            time_limit: Límite de tiempo en segundos
            verbose: Mostrar información detallada
            focus_on_units: Si es True, el modelo enfocará en maximizar unidades
            objective_type: Tipo de función objetivo ("ratio", "units", "aisles")
            min_orders: Número mínimo de órdenes a seleccionar
        """
        start_time = time.time()
        
        # Crear el solver
        if verbose:
            print(f"Creando solver SCIP para solución inicial (objetivo: {objective_type})...")
        solver = pywraplp.Solver.CreateSolver('SCIP')
        if not solver:
            print("No se pudo crear el solver SCIP")
            return None
            
        # Establecer límite de tiempo (en segundos)
        solver_time_limit = int((time_limit - (time.time() - start_time)) * 1000)
        solver.set_time_limit(solver_time_limit)
        if verbose:
            print(f"Límite de tiempo para el solver inicial: {solver_time_limit/1000:.2f} segundos")
        
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
        
        # Restricción opcional: mínimo de órdenes
        if min_orders is not None:
            solver.Add(solver.Sum([order_vars[i] for i in range(len(self.orders))]) >= min_orders)
        
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
        
        # Configurar la función objetivo según el tipo especificado
        objective = solver.Objective()
        
        if objective_type == "units":
            # Maximizar unidades
            for i in range(len(self.orders)):
                objective.SetCoefficient(order_vars[i], self.order_units[i])
            objective.SetMaximization()
            
        elif objective_type == "aisles":
            # Minimizar pasillos
            for j in range(len(self.aisles)):
                objective.SetCoefficient(aisle_vars[j], -1)
            objective.SetMaximization()
            
        else:  # "ratio" (default)
            # Maximizar ratio unidades/pasillos
            if focus_on_units:
                # Para instancias grandes, enfocamos en conseguir órdenes
                K = sum(self.order_units) / (len(self.aisles) * 2)  # Factor más pequeño
            else:
                # Para instancias pequeñas, balance entre órdenes y pasillos
                K = sum(self.order_units) / len(self.aisles)
            
            for i in range(len(self.orders)):
                objective.SetCoefficient(order_vars[i], self.order_units[i])
            for j in range(len(self.aisles)):
                objective.SetCoefficient(aisle_vars[j], -K)
            objective.SetMaximization()
        
        # Resolver el modelo
        if verbose:
            print("Resolviendo modelo MIP para solución inicial...")
        status = solver.Solve()
        
        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            # Obtener órdenes seleccionadas
            selected_orders = [i for i in range(len(self.orders)) if order_vars[i].solution_value() > 0.5]
            
            # Obtener pasillos visitados
            visited_aisles = [j for j in range(len(self.aisles)) if aisle_vars[j].solution_value() > 0.5]
            
            # Calcular ratio
            total_units = sum(self.order_units[i] for i in selected_orders)
            ratio = total_units / len(visited_aisles) if visited_aisles else 0
            
            if verbose:
                print(f"Solución MIP inicial encontrada: {len(selected_orders)} órdenes, {len(visited_aisles)} pasillos")
                print(f"Unidades: {total_units}, Ratio: {ratio:.2f}")
                print(f"Tiempo de solución MIP: {solver.WallTime()/1000:.2f} segundos")
            
            return selected_orders, visited_aisles
        else:
            if verbose:
                print("No se pudo encontrar una solución inicial con MIP")
            return None
    
    def _get_required_aisles(self, selected_orders):
        """
        Determina los pasillos necesarios para cubrir todos los items
        de las órdenes seleccionadas, tratando de minimizar el número de pasillos.
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
    
    def optimize_aisles(self, selected_orders, visited_aisles):
        """
        Intenta reducir el número de pasillos visitados manteniendo las órdenes.
        Usa un algoritmo greedy para seleccionar pasillos más eficientes.
        
        Retorna: nueva lista de pasillos visitados (potencialmente menor)
        """
        # Usar el algoritmo greedy para obtener el conjunto mínimo de pasillos
        optimized_aisles = self._get_required_aisles(selected_orders)
        
        # Verificar si la solución es factible con los pasillos optimizados
        if self.is_solution_feasible(selected_orders, optimized_aisles):
            # Si encontramos una selección de pasillos más pequeña, usarla
            if len(optimized_aisles) < len(visited_aisles):
                return optimized_aisles
        
        # Si la optimización no mejoró o no es factible, mantener los pasillos originales
        return visited_aisles
    
    def evaluate_solution(self, selected_orders, visited_aisles):
        """Evalúa la calidad de una solución"""
        # Usar caché para evitar recálculos
        key = (tuple(sorted(selected_orders)), tuple(sorted(visited_aisles)))
        if key in self.cache:
            return self.cache[key]
            
        # Verificar si la solución es factible
        if not self.is_solution_feasible(selected_orders, visited_aisles):
            self.cache[key] = 0
            return 0
            
        # Calcular valor objetivo
        objective = self.compute_objective_function(selected_orders, visited_aisles)
        self.cache[key] = objective
        return objective
    
    def get_neighbor_solution(self, selected_orders, visited_aisles, neighborhood_size=1, strategy=None):
        """
        Genera una solución vecina realizando cambios
        
        Args:
            selected_orders: Lista de órdenes seleccionadas
            visited_aisles: Lista de pasillos visitados
            neighborhood_size: Tamaño del vecindario (número de elementos a modificar)
            strategy: Estrategia específica para generar el vecino
        """
        # Hacer una copia para no modificar la original
        new_selected_orders = selected_orders.copy()
        
        # Elegir operación si no se especificó
        if strategy is None:
            strategies = ["add", "remove", "swap", "perturb"]
            strategy = random.choice(strategies)
        
        if strategy == "add":  # Agregar órdenes
            # Encontrar órdenes que no están seleccionadas
            unselected_orders = [i for i in range(len(self.orders)) if i not in new_selected_orders]
            if unselected_orders:
                # Elegir aleatoriamente órdenes para agregar
                candidates = random.sample(unselected_orders, min(neighborhood_size, len(unselected_orders)))
                
                # Verificar si podemos agregar sin exceder el límite superior
                total_units = sum(self.order_units[i] for i in new_selected_orders)
                for i in candidates:
                    if total_units + self.order_units[i] <= self.wave_size_ub:
                        new_selected_orders.append(i)
                        total_units += self.order_units[i]
                
        elif strategy == "remove":  # Quitar órdenes
            if len(new_selected_orders) > 1:  # Asegurar que quede al menos una orden
                # Elegir aleatoriamente órdenes para quitar
                to_remove = random.sample(new_selected_orders, min(neighborhood_size, max(1, len(new_selected_orders) // 4)))
                
                # Verificar si podemos quitar sin incumplir el límite inferior
                remaining_units = sum(self.order_units[i] for i in new_selected_orders) - sum(self.order_units[i] for i in to_remove)
                if remaining_units >= self.wave_size_lb:
                    new_selected_orders = [i for i in new_selected_orders if i not in to_remove]
                
        elif strategy == "swap":  # Intercambiar órdenes
            if new_selected_orders:
                # Encontrar órdenes que no están seleccionadas
                unselected_orders = [i for i in range(len(self.orders)) if i not in new_selected_orders]
                if unselected_orders:
                    # Elegir órdenes para intercambiar
                    orders_to_remove = random.sample(new_selected_orders, min(neighborhood_size, len(new_selected_orders)))
                    orders_to_add = random.sample(unselected_orders, min(neighborhood_size, len(unselected_orders)))
                    
                    # Calcular el cambio en unidades
                    units_to_remove = sum(self.order_units[i] for i in orders_to_remove)
                    units_to_add = sum(self.order_units[i] for i in orders_to_add)
                    
                    # Verificar si el intercambio respeta los límites
                    current_units = sum(self.order_units[i] for i in new_selected_orders)
                    new_units = current_units - units_to_remove + units_to_add
                    
                    if self.wave_size_lb <= new_units <= self.wave_size_ub:
                        new_selected_orders = [i for i in new_selected_orders if i not in orders_to_remove]
                        new_selected_orders.extend(orders_to_add)
        
        elif strategy == "perturb":  # Perturbación grande
            # Generar una solución completamente nueva utilizando las mejores soluciones anteriores
            if len(self.best_solutions) > 0 and random.random() < 0.7:
                # Utilizar una de las mejores soluciones anteriores como base
                best_idx = random.randint(0, len(self.best_solutions) - 1)
                new_selected_orders = self.best_solutions[best_idx][0].copy()
                
                # Pequeña mutación: añadir o quitar algunas órdenes
                if random.random() < 0.5:
                    # Añadir órdenes adicionales
                    unselected = [i for i in range(len(self.orders)) if i not in new_selected_orders]
                    if unselected:
                        candidates = random.sample(unselected, min(neighborhood_size, len(unselected)))
                        total_units = sum(self.order_units[i] for i in new_selected_orders)
                        for i in candidates:
                            if total_units + self.order_units[i] <= self.wave_size_ub:
                                new_selected_orders.append(i)
                                total_units += self.order_units[i]
                else:
                    # Quitar algunas órdenes
                    if len(new_selected_orders) > 1:
                        to_remove = random.sample(new_selected_orders, min(neighborhood_size, len(new_selected_orders) // 4))
                        remaining_units = sum(self.order_units[i] for i in new_selected_orders) - sum(self.order_units[i] for i in to_remove)
                        if remaining_units >= self.wave_size_lb:
                            new_selected_orders = [i for i in new_selected_orders if i not in to_remove]
            else:
                # Generar una solución completamente nueva
                # Seleccionar órdenes basándonos en eficiencia
                order_efficiency = []
                for i in range(len(self.orders)):
                    # Calcular los pasillos necesarios para esta orden
                    required_items = set(self.orders[i].keys())
                    required_aisles = set()
                    for item in required_items:
                        if item in self.item_locations:
                            for aisle in self.item_locations[item]:
                                required_aisles.add(aisle)
                    
                    # Calcular eficiencia = unidades / pasillos
                    if required_aisles:
                        efficiency = self.order_units[i] / len(required_aisles)
                    else:
                        efficiency = 0
                    
                    order_efficiency.append((i, efficiency, self.order_units[i]))
                
                # Ordenar por eficiencia (de mayor a menor)
                order_efficiency.sort(key=lambda x: x[1], reverse=True)
                
                # Agregar perturbación al orden
                if random.random() < 0.5:
                    # Hacer shuffle parcial
                    n = len(order_efficiency)
                    for i in range(min(n, 100)):  # Limitar a 100 intercambios para eficiencia
                        idx1, idx2 = random.randint(0, n-1), random.randint(0, n-1)
                        order_efficiency[idx1], order_efficiency[idx2] = order_efficiency[idx2], order_efficiency[idx1]
                
                # Reiniciar órdenes seleccionadas
                new_selected_orders = []
                total_units = 0
                
                # Agregar órdenes hasta alcanzar el límite inferior
                for i, _, units in order_efficiency:
                    if total_units + units <= self.wave_size_ub:
                        new_selected_orders.append(i)
                        total_units += units
                        
                        if total_units >= self.wave_size_lb and random.random() < 0.3:
                            # 30% de probabilidad de detenerse si ya alcanzamos el mínimo
                            break
        
        # Determinar los pasillos necesarios para la nueva selección de órdenes
        new_visited_aisles = self._get_required_aisles(new_selected_orders)
        
        return new_selected_orders, new_visited_aisles
    
    def local_search(self, selected_orders, visited_aisles, max_iterations=1000):
        """
        Realiza una búsqueda local para intentar mejorar la solución actual.
        """
        current_orders = selected_orders.copy()
        current_aisles = visited_aisles.copy()
        current_value = self.evaluate_solution(current_orders, current_aisles)
        
        best_orders = current_orders.copy()
        best_aisles = current_aisles.copy()
        best_value = current_value
        
        # Intentar diferentes estrategias de vecindario
        strategies = ["add", "remove", "swap"]
        iterations_without_improvement = 0
        
        for _ in range(max_iterations):
            # Seleccionar estrategia
            strategy = random.choice(strategies)
            
            # Generar vecino
            new_orders, new_aisles = self.get_neighbor_solution(
                current_orders, current_aisles, 
                neighborhood_size=1, 
                strategy=strategy
            )
            
            # Evaluar vecino
            new_value = self.evaluate_solution(new_orders, new_aisles)
            
            # Actualizar si mejora
            if new_value > current_value:
                current_orders = new_orders
                current_aisles = new_aisles
                current_value = new_value
                iterations_without_improvement = 0
                
                # Actualizar mejor solución
                if new_value > best_value:
                    best_orders = new_orders.copy()
                    best_aisles = new_aisles.copy()
                    best_value = new_value
            else:
                iterations_without_improvement += 1
                
            # Salir si no hay mejora por un tiempo
            if iterations_without_improvement >= 100:
                break
        
        return best_orders, best_aisles, best_value
    
    def solve_with_multi_start(self, time_limit=600, max_starts=10, verbose=True):
        """
        Resuelve el problema utilizando múltiples inicios con diferentes objetivos
        y estrategias, y luego mejora las soluciones usando búsqueda local.
        """
        # Iniciar la barra de progreso
        progress = ProgressBar(time_limit)
        progress.start()
        
        start_time = time.time()
        
        # Preprocesar datos
        if verbose:
            print("\nPreprocesando datos...")
        self.preprocess_data()
        
        # Determinar el tipo de instancia
        is_large_instance = len(self.orders) > 100
        
        # Configurar enfoques de inicio según tamaño
        if is_large_instance:
            # Para instancias grandes, diferentes objetivos y tiempos
            initial_approaches = [
                {"objective": "ratio", "time_limit": 60, "focus_on_units": True},
                {"objective": "units", "time_limit": 30, "focus_on_units": True},
                {"objective": "aisles", "time_limit": 30, "focus_on_units": False}
            ]
        else:
            # Para instancias pequeñas
            initial_approaches = [
                {"objective": "ratio", "time_limit": 30, "focus_on_units": False},
                {"objective": "units", "time_limit": 15, "focus_on_units": False},
                {"objective": "aisles", "time_limit": 15, "focus_on_units": False}
            ]
        
        # Soluciones encontradas
        all_solutions = []
        best_solution = None
        best_objective = 0
        
        # Generar soluciones iniciales con diferentes enfoques
        for i, approach in enumerate(initial_approaches):
            if time.time() - start_time >= time_limit:
                break
                
            if verbose:
                elapsed = time.time() - start_time
                remaining = time_limit - elapsed
                print(f"\n[{elapsed:.1f}s] Generando solución inicial {i+1}/{len(initial_approaches)} "
                      f"(Objetivo: {approach['objective']}, Tiempo: {approach['time_limit']}s)")
            
            # Ajustar tiempo límite basado en tiempo restante
            adjusted_time = min(approach["time_limit"], remaining * 0.3)  # Máx 30% del tiempo restante
            
            # Generar solución
            solution = self.generate_initial_solution_mip(
                time_limit=adjusted_time,
                verbose=verbose,
                focus_on_units=approach["focus_on_units"],
                objective_type=approach["objective"]
            )
            
            if solution:
                selected_orders, visited_aisles = solution
                objective = self.evaluate_solution(selected_orders, visited_aisles)
                
                # Guardar solución
                all_solutions.append((solution, objective))
                
                # Actualizar mejor solución
                if objective > best_objective:
                    best_solution = solution
                    best_objective = objective
                    
                    if verbose:
                        print(f"Nueva mejor solución: Valor={objective:.2f}, "
                              f"Órdenes={len(selected_orders)}, Pasillos={len(visited_aisles)}")
        
        # Tiempo para búsqueda local
        elapsed = time.time() - start_time
        remaining_time = time_limit - elapsed
        
        if verbose:
            print(f"\n[{elapsed:.1f}s] Tiempo restante para búsqueda local: {remaining_time:.2f}s")
        
        if not all_solutions:
            progress.stop()
            if verbose:
                print("No se pudo generar ninguna solución inicial factible.")
            return None, 0
            
        # Ordenar soluciones por valor objetivo (de mayor a menor)
        all_solutions.sort(key=lambda x: x[1], reverse=True)
        
        # Tomar las mejores soluciones para refinar
        top_solutions = all_solutions[:min(len(all_solutions), max_starts)]
        refined_solutions = []
        
        # Tiempo a asignar a cada solución
        time_per_solution = remaining_time / len(top_solutions)
        
        # Refinar cada solución usando búsqueda local
        for i, (solution, _) in enumerate(top_solutions):
            if time.time() - start_time >= time_limit:
                break
                
            selected_orders, visited_aisles = solution
            
            if verbose:
                current_time = time.time() - start_time
                print(f"\n[{current_time:.1f}s] Refinando solución {i+1}/{len(top_solutions)} "
                      f"(Órdenes: {len(selected_orders)}, Pasillos: {len(visited_aisles)})")
            
            # Optimizar pasillos primero
            optimized_aisles = self.optimize_aisles(selected_orders, visited_aisles)
            
            # Calcular el nuevo valor objetivo
            current_objective = self.evaluate_solution(selected_orders, optimized_aisles)
            
            # Guardar en mejores soluciones para diversificación futura
            self.best_solutions.append((selected_orders, optimized_aisles))
            
            # Tiempo para esta solución
            solution_end_time = min(time.time() + time_per_solution, start_time + time_limit)
            
            # Aplicar búsqueda local hasta alcanzar el tiempo límite
            iteration = 0
            while time.time() < solution_end_time:
                iteration += 1
                
                # Buscar localmente por un número limitado de iteraciones
                max_local_iterations = 1000
                improved_orders, improved_aisles, improved_objective = self.local_search(
                    selected_orders, optimized_aisles, max_iterations=max_local_iterations
                )
                
                # Actualizar si hay mejora
                if improved_objective > current_objective:
                    selected_orders = improved_orders
                    optimized_aisles = improved_aisles
                    current_objective = improved_objective
                    
                    # Guardar en mejores soluciones
                    self.best_solutions.append((selected_orders, optimized_aisles))
                    
                    if verbose:
                        current_time = time.time() - start_time
                        print(f"[{current_time:.1f}s] Solución mejorada: Valor={current_objective:.2f}, "
                              f"Órdenes={len(selected_orders)}, Pasillos={len(optimized_aisles)}")
                
                # Actualizar barra de progreso
                progress.update_info(f"Mejor={best_objective:.2f}, Sol={i+1}/{len(top_solutions)}, "
                                    f"Iter={iteration}, Ord={len(selected_orders)}, Pas={len(optimized_aisles)}")
                
                # Aplicar perturbación y continuar búsqueda
                if iteration % 5 == 0:  # Cada 5 iteraciones
                    # Intentar una perturbación más grande
                    perturbed_orders, perturbed_aisles = self.get_neighbor_solution(
                        selected_orders, optimized_aisles, 
                        neighborhood_size=max(5, len(selected_orders) // 10),
                        strategy="perturb"
                    )
                    
                    # Evaluar solución perturbada
                    perturbed_objective = self.evaluate_solution(perturbed_orders, perturbed_aisles)
                    
                    # Aceptar con cierta probabilidad incluso si es peor
                    if perturbed_objective > current_objective * 0.9:  # Aceptar si está dentro del 90%
                        selected_orders = perturbed_orders
                        optimized_aisles = perturbed_aisles
                        current_objective = perturbed_objective
            
            # Guardar solución refinada
            refined_solutions.append(((selected_orders, optimized_aisles), current_objective))
            
            # Actualizar mejor solución global
            if current_objective > best_objective:
                best_solution = (selected_orders, optimized_aisles)
                best_objective = current_objective
        
        # Detener barra de progreso
        progress.stop()
        
        if verbose:
            print("\nMejores soluciones encontradas:")
            refined_solutions.sort(key=lambda x: x[1], reverse=True)
            for i, ((orders, aisles), obj) in enumerate(refined_solutions[:3]):  # Top 3
                print(f"{i+1}. Valor={obj:.2f}, Órdenes={len(orders)}, Pasillos={len(aisles)}")
        
        # Retornar la mejor solución encontrada
        return best_solution, best_objective
    
    def write_solution(self, selected_orders, visited_aisles, output_file_path):
        """
        Escribe la solución en el formato requerido
        """
        with open(output_file_path, 'w') as file:
            file.write(f"{len(selected_orders)}\n")
            for order in selected_orders:
                file.write(f"{order}\n")
            file.write(f"{len(visited_aisles)}\n")
            for aisle in visited_aisles:
                file.write(f"{aisle}\n")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python improved_solver.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    print(f"\n{'='*50}")
    print(f"RESOLVIENDO INSTANCIA (ENFOQUE MEJORADO): {input_file}")
    print(f"{'='*50}")
    
    solver = ImprovedSolver()
    solver.read_input(input_file)
    
    start_time = time.time()
    # Utilizar el enfoque multi-start
    result = solver.solve_with_multi_start(
        time_limit=600,  # 10 minutos
        max_starts=5,    # Máximo 5 inicios
        verbose=True
    )
    end_time = time.time()
    
    print(f"\n{'='*50}")
    print(f"RESULTADOS:")
    print(f"{'='*50}")
    
    if result is not None and result[0] is not None:
        (selected_orders, visited_aisles), objective = result
        print(f"Solución factible encontrada:")
        print(f"- Valor objetivo: {objective}")
        print(f"- Órdenes seleccionadas: {len(selected_orders)}")
        print(f"- Pasillos visitados: {len(visited_aisles)}")
        print(f"- Ratio (unidades/pasillos): {objective}")
        print(f"- Tiempo total: {end_time - start_time:.2f} segundos")
        solver.write_solution(selected_orders, visited_aisles, output_file)
        print(f"Solución guardada en: {output_file}")
    else:
        print("No se encontró una solución factible")
        
    print(f"{'='*50}\n") 