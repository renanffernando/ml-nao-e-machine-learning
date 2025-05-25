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
        Generates an initial solution using mixed integer programming
        
        Args:
            time_limit: Time limit in seconds
            verbose: Show detailed information
            focus_on_units: If True, model will focus on maximizing units
            objective_type: Type of objective function ("ratio", "units", "aisles")
            min_orders: Minimum number of orders to select
        """
        start_time = time.time()
        
        # Create solver
        if verbose:
            print(f"Creating SCIP solver for initial solution (objective: {objective_type})...")
        solver = pywraplp.Solver.CreateSolver('SCIP')
        if not solver:
            print("Could not create SCIP solver")
            return None
            
        # Set time limit (in seconds)
        solver_time_limit = int((time_limit - (time.time() - start_time)) * 1000)
        solver.set_time_limit(solver_time_limit)
        if verbose:
            print(f"Time limit for initial solver: {solver_time_limit/1000:.2f} seconds")
        
        # Variables: order and aisle selection
        order_vars = {}
        for i in range(len(self.orders)):
            order_vars[i] = solver.BoolVar(f'order_{i}')
            
        aisle_vars = {}
        for j in range(len(self.aisles)):
            aisle_vars[j] = solver.BoolVar(f'aisle_{j}')
        
        # Constraint: wave size limits
        total_units = solver.Sum([self.order_units[i] * order_vars[i] for i in range(len(self.orders))])
        solver.Add(total_units >= self.wave_size_lb)
        solver.Add(total_units <= self.wave_size_ub)
        
        # Constraint: if an order is selected, we need to visit aisles containing its items
        for i in range(len(self.orders)):
            for item in self.orders[i]:
                # Aisles containing this item
                relevant_aisles = [j for j in range(len(self.aisles)) if item in self.aisles[j]]
                if relevant_aisles:
                    solver.Add(
                        solver.Sum([aisle_vars[j] for j in relevant_aisles]) >= order_vars[i]
                    )
        
        # Constraint: minimum number of orders if specified
        if min_orders is not None:
            solver.Add(solver.Sum([order_vars[i] for i in range(len(self.orders))]) >= min_orders)
        
        # Objective function based on type
        objective = solver.Objective()
        
        if objective_type == "units":
            # Maximize units
            for i in range(len(self.orders)):
                objective.SetCoefficient(order_vars[i], self.order_units[i])
            objective.SetMaximization()
            
        elif objective_type == "aisles":
            # Minimize aisles
            for j in range(len(self.aisles)):
                objective.SetCoefficient(aisle_vars[j], -1)
            objective.SetMaximization()
            
        else:  # "ratio" (default)
            # Maximize units/aisles ratio
            if focus_on_units:
                # For large instances, focus on getting orders
                K = sum(self.order_units) / (len(self.aisles) * 2)  # Smaller factor
            else:
                # For small instances, balance between orders and aisles
                K = sum(self.order_units) / len(self.aisles)
            
            for i in range(len(self.orders)):
                objective.SetCoefficient(order_vars[i], self.order_units[i])
            for j in range(len(self.aisles)):
                objective.SetCoefficient(aisle_vars[j], -K)
            objective.SetMaximization()
        
        # Solve the model
        if verbose:
            print("Solving MIP model for initial solution...")
        status = solver.Solve()
        
        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            # Get selected orders
            selected_orders = [i for i in range(len(self.orders)) if order_vars[i].solution_value() > 0.5]
            
            # Get visited aisles
            visited_aisles = [j for j in range(len(self.aisles)) if aisle_vars[j].solution_value() > 0.5]
            
            # Calculate ratio
            total_units = sum(self.order_units[i] for i in selected_orders)
            ratio = total_units / len(visited_aisles) if visited_aisles else 0
            
            if verbose:
                print(f"Initial MIP solution found: {len(selected_orders)} orders, {len(visited_aisles)} aisles")
                print(f"Units: {total_units}, Ratio: {ratio:.2f}")
                print(f"MIP solution time: {solver.WallTime()/1000:.2f} seconds")
            
            return selected_orders, visited_aisles
        else:
            if verbose:
                print("Could not find initial solution with MIP")
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
        Solves the problem using multiple starting points and local search.
        
        Args:
            time_limit: Time limit in seconds
            max_starts: Maximum number of starting points to try
            verbose: If True, show detailed information
            
        Returns:
            (selected_orders, visited_aisles, best_ratio) if solution found, None otherwise
        """
        start_time = time.time()
        best_solution = None
        best_ratio = 0
        all_solutions = []
        
        # For progress reporting
        if verbose:
            progress = ProgressBar(max_iterations=max_starts, 
                                max_time=time_limit,
                                start_time=start_time,
                                description="Generating initial solutions")
            
        # Generate initial solutions with different objectives
        objectives = [
            ("ratio", False),  # Default ratio optimization
            ("ratio", True),   # Ratio optimization focusing on units
            ("units", False),  # Pure units maximization
            ("aisles", False)  # Pure aisles minimization
        ]
        
        # Add some random minimum order requirements
        min_orders_list = [None]  # Start with no minimum
        avg_orders = len(self.orders) // 2
        min_orders_list.extend([
            max(1, int(avg_orders * 0.3)),  # 30% of average
            max(1, int(avg_orders * 0.5)),  # 50% of average
            max(1, int(avg_orders * 0.7))   # 70% of average
        ])
        
        # Try different combinations
        iteration = 0
        for obj_type, focus in objectives:
            for min_orders in min_orders_list:
                # Check remaining time
                elapsed = time.time() - start_time
                remaining_time = time_limit - elapsed
                
                if remaining_time < 30:  # Need at least 30 seconds
                    if verbose:
                        print(f"Insufficient remaining time ({remaining_time:.2f}s). Stopping generation.")
                    break
                    
                # Try to generate solution
                solution = self.generate_initial_solution_mip(
                    time_limit=min(remaining_time * 0.2, 60),  # Use at most 20% of remaining time
                    verbose=verbose,
                    focus_on_units=focus,
                    objective_type=obj_type,
                    min_orders=min_orders
                )
                
                if solution:
                    selected_orders, visited_aisles = solution
                    ratio = sum(self.order_units[i] for i in selected_orders) / len(visited_aisles)
                    all_solutions.append((solution, ratio))
                    
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_solution = solution
                        
                iteration += 1
                if verbose:
                    progress.update(iteration)
                    
                if iteration >= max_starts:
                    break
                    
            if iteration >= max_starts:
                break
                
        if verbose:
            print(f"\n[{elapsed:.1f}s] Remaining time for local search: {remaining_time:.2f}s")
        
        if not all_solutions:
            progress.stop()
            if verbose:
                print("Could not generate any feasible initial solution.")
            return None, 0
            
        # Sort solutions by objective value (highest to lowest)
        all_solutions.sort(key=lambda x: x[1], reverse=True)
        
        # Take best solutions to refine
        top_solutions = all_solutions[:min(len(all_solutions), max_starts)]
        refined_solutions = []
        
        # Time to allocate to each solution
        time_per_solution = remaining_time / len(top_solutions)
        
        # Refine each solution using local search
        for i, (solution, _) in enumerate(top_solutions):
            # Check remaining time
            elapsed = time.time() - start_time
            remaining_time = time_limit - elapsed
            
            if remaining_time < 10:  # Need at least 10 seconds
                if verbose:
                    print(f"Insufficient remaining time ({remaining_time:.2f}s). Stopping refinement.")
                break
                
            # Try to improve solution
            improved = self.improve_solution(
                solution[0], solution[1],
                time_limit=min(time_per_solution, remaining_time - 5),
                verbose=verbose
            )
            
            if improved:
                selected_orders, visited_aisles = improved
                ratio = sum(self.order_units[i] for i in selected_orders) / len(visited_aisles)
                refined_solutions.append((improved, ratio))
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_solution = improved
                    
        if verbose:
            progress.stop()
            
        # Return best solution found
        if best_solution:
            return best_solution[0], best_solution[1], best_ratio
        else:
            return None
    
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