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

class WaveOrderPickingHybridSolver(WaveOrderPicking):
    def __init__(self):
        super().__init__()
        self.order_units = None  # Unidades por orden
        self.item_locations = None  # Pasillos que contienen cada item
        self.cache = {}  # Caché para evaluaciones de soluciones
        
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
    
    def generate_initial_solution_mip(self, time_limit=30, verbose=True, focus_on_units=False):
        """
        Genera una solución inicial usando programación lineal entera mixta
        
        Args:
            time_limit: Límite de tiempo en segundos
            verbose: Mostrar información detallada
            focus_on_units: Si es True, el modelo enfocará en maximizar unidades
                           sobre minimizar pasillos (útil para instancias grandes)
        """
        start_time = time.time()
        
        # Crear el solver
        if verbose:
            print("Creando solver SCIP para solución inicial...")
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
        
        # Optimización: usar modelo simple para solución inicial rápida
        # Balancear maximizar unidades y minimizar pasillos
        if focus_on_units:
            # Para instancias grandes, enfocamos en conseguir órdenes
            # con un factor de penalización menor por pasillo
            K = sum(self.order_units) / (len(self.aisles) * 2)  # Factor de escala reducido
        else:
            # Para instancias pequeñas, balance entre órdenes y pasillos
            K = sum(self.order_units) / len(self.aisles)  # Factor de escala estándar
        
        objective = solver.Objective()
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
            
            if verbose:
                print(f"Solución MIP inicial encontrada: {len(selected_orders)} órdenes, {len(visited_aisles)} pasillos")
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
        Genera una solución vecina realizando cambios:
        
        Args:
            selected_orders: Lista de órdenes seleccionadas
            visited_aisles: Lista de pasillos visitados
            neighborhood_size: Tamaño del vecindario (número de elementos a modificar)
            strategy: Estrategia específica para generar el vecino:
                     - None: aleatorio entre todas las estrategias
                     - "add": agregar órdenes
                     - "remove": quitar órdenes
                     - "swap": intercambiar órdenes
                     - "perturb": perturbación grande (reinicio parcial)
        """
        # Hacer una copia para no modificar la original
        new_selected_orders = selected_orders.copy()
        
        # Elegir operación si no se especificó
        if strategy is None:
            strategies = ["add", "remove", "swap"]
            # Ocasionalmente hacer una perturbación grande
            if random.random() < 0.05:  # 5% de probabilidad
                strategy = "perturb"
            else:
                strategy = random.choice(strategies)
        
        if strategy == "add":  # Agregar
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
                
        elif strategy == "remove":  # Quitar
            if len(new_selected_orders) > 1:  # Asegurar que quede al menos una orden
                # Elegir aleatoriamente órdenes para quitar
                to_remove = random.sample(new_selected_orders, min(neighborhood_size, max(1, len(new_selected_orders) // 4)))
                
                # Verificar si podemos quitar sin incumplir el límite inferior
                remaining_units = sum(self.order_units[i] for i in new_selected_orders) - sum(self.order_units[i] for i in to_remove)
                if remaining_units >= self.wave_size_lb:
                    new_selected_orders = [i for i in new_selected_orders if i not in to_remove]
                
        elif strategy == "swap":  # Intercambiar
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
        
        elif strategy == "perturb":  # Perturbación grande (reinicio parcial)
            # Mantener solo una parte de las órdenes actuales
            keep_ratio = random.uniform(0.3, 0.7)  # Mantener entre 30% y 70%
            keep_count = max(1, int(len(new_selected_orders) * keep_ratio))
            
            # Órdenes a mantener
            new_selected_orders = random.sample(new_selected_orders, keep_count)
            
            # Calcular unidades restantes y cuántas necesitamos
            current_units = sum(self.order_units[i] for i in new_selected_orders)
            remaining_capacity = self.wave_size_ub - current_units
            min_additional = max(0, self.wave_size_lb - current_units)
            
            # Encontrar órdenes que no están seleccionadas
            unselected_orders = [i for i in range(len(self.orders)) if i not in new_selected_orders]
            
            # Ordenar por eficiencia (unidades por pasillo)
            order_efficiency = []
            for i in unselected_orders:
                # Estimar pasillos para esta orden
                required_items = set(self.orders[i].keys())
                required_aisles = set()
                for item in required_items:
                    if item in self.item_locations:
                        for aisle in self.item_locations[item]:
                            required_aisles.add(aisle)
                
                # Cálculo de eficiencia
                efficiency = self.order_units[i] / max(1, len(required_aisles))
                order_efficiency.append((i, efficiency, self.order_units[i]))
            
            # Ordenar por eficiencia descendente
            order_efficiency.sort(key=lambda x: x[1], reverse=True)
            
            # Agregar órdenes hasta alcanzar el mínimo requerido
            for i, _, units in order_efficiency:
                if min_additional <= 0 and random.random() < 0.5:
                    # Si ya alcanzamos el mínimo, hay 50% de probabilidad de parar
                    break
                
                if units <= remaining_capacity:
                    new_selected_orders.append(i)
                    remaining_capacity -= units
                    min_additional = max(0, min_additional - units)
        
        # Determinar los pasillos necesarios para la nueva selección de órdenes
        new_visited_aisles = self._get_required_aisles(new_selected_orders)
        
        return new_selected_orders, new_visited_aisles
    
    def solve_with_vns(self, time_limit=600, initial_temp=100.0, cooling_rate=0.95,
                      iterations_per_temp=100, mip_time_limit=60, focus_on_units=False, verbose=True):
        """
        Resuelve el problema usando Variable Neighborhood Search combinado con Simulated Annealing
        
        Args:
            time_limit: Límite de tiempo total en segundos
            mip_time_limit: Tiempo asignado para la fase MIP inicial
            initial_temp: Temperatura inicial para Recocido Simulado
            cooling_rate: Tasa de enfriamiento
            iterations_per_temp: Iteraciones por nivel de temperatura
            focus_on_units: Si es True, el modelo MIP se enfocará más en maximizar unidades
            verbose: Mostrar información detallada
        """
        # Iniciar la barra de progreso
        progress = ProgressBar(time_limit)
        progress.start()
        
        start_time = time.time()
        
        # Preprocesar datos
        if verbose:
            print("\nPreprocesando datos...")
        self.preprocess_data()
        
        # Generar solución inicial con MIP (asignar una parte del tiempo total)
        adjusted_mip_time = min(mip_time_limit, time_limit * 0.2)  # Máx. 20% del tiempo total
        solution = self.generate_initial_solution_mip(time_limit=adjusted_mip_time, 
                                                     verbose=verbose,
                                                     focus_on_units=focus_on_units)
        
        if not solution:
            progress.stop()
            if verbose:
                print("No se pudo generar una solución inicial factible.")
            return None, 0
        
        # Tiempo restante para la búsqueda
        elapsed = time.time() - start_time
        remaining_time = time_limit - elapsed
        
        if verbose:
            print(f"Tiempo restante para Variable Neighborhood Search: {remaining_time:.2f} segundos")
        
        if remaining_time <= 0:
            progress.stop()
            selected_orders, visited_aisles = solution
            objective = self.compute_objective_function(selected_orders, visited_aisles)
            return (selected_orders, visited_aisles), objective
        
        current_solution = solution
        best_solution = solution
        
        selected_orders, visited_aisles = current_solution
        current_value = self.evaluate_solution(selected_orders, visited_aisles)
        best_value = current_value
        
        # Calcular temperatura inicial adaptativa basada en el valor actual
        if initial_temp <= 0:
            # Calculamos la temperatura inicial basada en un porcentaje del valor actual
            # para permitir movimientos que empeoran hasta en un 10%
            initial_temp = abs(current_value * 0.1)
            if verbose:
                print(f"Temperatura inicial adaptativa: {initial_temp}")
        
        if verbose:
            print(f"Solución inicial: Valor={current_value}, Órdenes={len(selected_orders)}, Pasillos={len(visited_aisles)}")
            
        # Inicializar temperatura y estructuras
        temp = initial_temp
        neighborhood_strategies = ["add", "remove", "swap", "perturb"]
        current_strategy_idx = 0
        
        # Estadísticas
        iterations = 0
        accepted_moves = 0
        improving_moves = 0
        strategy_stats = {s: {"attempts": 0, "accepted": 0, "improved": 0} for s in neighborhood_strategies}
        
        # Para reinicios
        non_improving_iterations = 0
        max_non_improving = 5000
        
        # Para intensificación
        intensification_phase = False
        
        while time.time() - start_time < time_limit:
            # Seleccionar estrategia de vecindario
            current_strategy = neighborhood_strategies[current_strategy_idx]
            
            # Realizar iteraciones con esta estrategia
            strategy_iterations = 0
            strategy_improvements = 0
            
            for _ in range(iterations_per_temp):
                iterations += 1
                non_improving_iterations += 1
                strategy_stats[current_strategy]["attempts"] += 1
                
                # Tamaño de vecindario adaptativo
                if current_strategy == "perturb":
                    # Perturbación siempre usa un vecindario grande
                    neighborhood_size = max(5, len(selected_orders) // 10)
                elif intensification_phase:
                    # En fase de intensificación, movimientos más pequeños
                    neighborhood_size = 1
                else:
                    # En fase normal, tamaño variable
                    neighborhood_size = random.randint(1, min(5, max(1, len(selected_orders) // 20)))
                
                # Generar vecino usando la estrategia actual
                neighbor_solution = self.get_neighbor_solution(
                    selected_orders, visited_aisles, 
                    neighborhood_size=neighborhood_size,
                    strategy=current_strategy
                )
                
                # Evaluar vecino
                new_selected_orders, new_visited_aisles = neighbor_solution
                neighbor_value = self.evaluate_solution(new_selected_orders, new_visited_aisles)
                
                # Calcular delta (cambio en la función objetivo)
                delta = neighbor_value - current_value
                
                # Decidir si aceptar el movimiento
                accept = False
                if delta > 0:  # Mejora
                    accept = True
                    improving_moves += 1
                    strategy_improvements += 1
                    strategy_stats[current_strategy]["improved"] += 1
                    non_improving_iterations = 0
                else:  # No mejora, aceptar con probabilidad
                    # La probabilidad depende de la fase y estrategia
                    if intensification_phase:
                        # En intensificación, menos probabilidad
                        probability = math.exp(delta / (temp * 0.5))
                    elif current_strategy == "perturb":
                        # Para perturbaciones, mayor probabilidad de aceptar
                        probability = math.exp(delta / (temp * 2.0))
                    else:
                        # Caso normal
                        probability = math.exp(delta / temp)
                        
                    if random.random() < probability:
                        accept = True
                
                if accept:
                    accepted_moves += 1
                    strategy_stats[current_strategy]["accepted"] += 1
                    current_solution = neighbor_solution
                    selected_orders, visited_aisles = current_solution
                    current_value = neighbor_value
                    strategy_iterations += 1
                    
                    # Actualizar mejor solución si corresponde
                    if current_value > best_value:
                        best_solution = current_solution
                        best_value = current_value
                        
                        if verbose:
                            elapsed = time.time() - start_time
                            print(f"[{elapsed:.1f}s] Nueva mejor solución: Valor={best_value}, "
                                  f"Órdenes={len(selected_orders)}, Pasillos={len(visited_aisles)}, "
                                  f"Estrategia={current_strategy}")
                
                # Verificar si es tiempo de cambiar de estrategia
                if strategy_iterations >= 100 or iterations % 50 == 0:
                    break
                    
                # Verificar si es tiempo de reiniciar (diversificación)
                if non_improving_iterations > max_non_improving:
                    if verbose:
                        print(f"Reiniciando búsqueda después de {non_improving_iterations} iteraciones sin mejora")
                    
                    # Reiniciar desde la mejor solución
                    current_solution = best_solution
                    selected_orders, visited_aisles = current_solution
                    current_value = best_value
                    
                    # Reiniciar temperatura y contadores
                    temp = initial_temp
                    non_improving_iterations = 0
                    intensification_phase = False
                    
                    # Ir directamente a la perturbación
                    current_strategy_idx = neighborhood_strategies.index("perturb")
                    break
                
                # Verificar límite de tiempo
                if time.time() - start_time >= time_limit:
                    break
            
            # Cambiar a la siguiente estrategia
            current_strategy_idx = (current_strategy_idx + 1) % len(neighborhood_strategies)
            
            # Si ninguna estrategia mejora, alternar entre exploración e intensificación
            if iterations % 500 == 0:
                intensification_phase = not intensification_phase
                if verbose:
                    phase_name = "intensificación" if intensification_phase else "exploración"
                    print(f"Cambiando a fase de {phase_name}")
            
            # Enfriar
            if temp > initial_temp * 0.1:
                temp *= cooling_rate
            else:
                temp *= cooling_rate * 0.9
            
            # Actualizar barra de progreso
            acceptance_rate = accepted_moves / max(1, iterations)
            improvement_rate = improving_moves / max(1, iterations)
            
            phase_name = "I" if intensification_phase else "E"
            progress.update_info(f"Mejor={best_value:.2f}, Temp={temp:.2f}, "
                                f"Fase={phase_name}, Est={current_strategy[0:3]}, "
                                f"Acep={acceptance_rate:.2f}")
            
            # Detener si la temperatura es muy baja y estamos en fase de intensificación
            if temp < 0.001 and intensification_phase:
                if verbose:
                    print("Temperatura muy baja, terminando búsqueda.")
                break
        
        # Detener barra de progreso
        progress.stop()
        
        if verbose:
            print("\nEstadísticas de Variable Neighborhood Search:")
            print(f"- Iteraciones totales: {iterations}")
            print(f"- Movimientos aceptados: {accepted_moves} ({accepted_moves/max(1, iterations):.2%})")
            print(f"- Movimientos de mejora: {improving_moves} ({improving_moves/max(1, iterations):.2%})")
            print(f"- Temperatura final: {temp:.4f}")
            print("\nEstadísticas por estrategia:")
            for s in neighborhood_strategies:
                attempts = strategy_stats[s]["attempts"]
                if attempts > 0:
                    accepted = strategy_stats[s]["accepted"]
                    improved = strategy_stats[s]["improved"]
                    print(f"- {s}: {attempts} intentos, {accepted/attempts:.1%} aceptados, {improved/attempts:.1%} mejoras")
        
        selected_orders, visited_aisles = best_solution
        return (selected_orders, visited_aisles), best_value
        
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
        print("Usage: python hybrid_solver.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    print(f"\n{'='*50}")
    print(f"RESOLVIENDO INSTANCIA (ENFOQUE HÍBRIDO): {input_file}")
    print(f"{'='*50}")
    
    solver = WaveOrderPickingHybridSolver()
    solver.read_input(input_file)
    
    # Determinar tipo de instancia basado en el número de órdenes
    is_large_instance = len(solver.orders) > 100
    
    # Ajustar parámetros según tamaño de la instancia
    if is_large_instance:
        print("Detectada instancia grande. Ajustando parámetros...")
        mip_time = 120  # 2 minutos para MIP en instancias grandes
        focus_on_units = True
        sa_temp = -1  # Usar temperatura adaptativa
    else:
        mip_time = 60  # 1 minuto para MIP en instancias pequeñas
        focus_on_units = False
        sa_temp = 100.0
    
    start_time = time.time()
    # Usar el nuevo algoritmo de Variable Neighborhood Search
    result = solver.solve_with_vns(
        time_limit=600,
        mip_time_limit=mip_time,
        initial_temp=sa_temp,
        cooling_rate=0.95,
        iterations_per_temp=100,
        focus_on_units=focus_on_units,
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