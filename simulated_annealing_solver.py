import sys
import time
import random
import math
import threading
import numpy as np
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

class WaveOrderPickingSASolver(WaveOrderPicking):
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
    
    def generate_initial_solution(self, strategy="greedy"):
        """
        Genera una solución inicial
        
        Estrategias disponibles:
        - "random": Selección aleatoria de órdenes hasta cumplir con los límites
        - "greedy": Selección de órdenes basada en la densidad de items (unidades/pasillos)
        """
        if strategy == "random":
            return self._generate_random_solution()
        else:  # greedy
            return self._generate_greedy_solution()
    
    def _generate_random_solution(self):
        """Genera una solución inicial aleatoria"""
        selected_orders = []
        total_units = 0
        
        # Ordenar las órdenes aleatoriamente
        order_candidates = list(range(len(self.orders)))
        random.shuffle(order_candidates)
        
        # Agregar órdenes hasta alcanzar el mínimo (LB)
        for i in order_candidates:
            if total_units + self.order_units[i] <= self.wave_size_ub:
                selected_orders.append(i)
                total_units += self.order_units[i]
                
                if total_units >= self.wave_size_lb:
                    break
                    
        # Si no alcanzamos el límite inferior, no hay solución factible
        if total_units < self.wave_size_lb:
            return None
            
        # Determinar los pasillos necesarios
        visited_aisles = self._get_required_aisles(selected_orders)
        
        # Verificar si la solución es factible
        if not self.is_solution_feasible(selected_orders, visited_aisles):
            return None
            
        return selected_orders, visited_aisles
    
    def _generate_greedy_solution(self):
        """
        Genera una solución inicial utilizando un enfoque greedy:
        1. Ordenar las órdenes por eficiencia (unidades / pasillos necesarios)
        2. Seleccionar órdenes hasta alcanzar el límite inferior
        3. Refinar la solución
        """
        # Calcular la "eficiencia" de cada orden (unidades / pasillos necesarios)
        order_efficiency = []
        
        for i in range(len(self.orders)):
            # Calcular los pasillos necesarios para esta orden
            required_aisles = set()
            for item_id in self.orders[i]:
                if item_id in self.item_locations:
                    # Tomar el primer pasillo que tiene este item (se podría optimizar más)
                    if self.item_locations[item_id]:
                        required_aisles.add(self.item_locations[item_id][0])
            
            # Calcular eficiencia
            if required_aisles:
                efficiency = self.order_units[i] / len(required_aisles)
            else:
                efficiency = 0
                
            order_efficiency.append((i, efficiency, self.order_units[i]))
        
        # Ordenar por eficiencia (de mayor a menor)
        order_efficiency.sort(key=lambda x: (x[1], x[2]), reverse=True)
        
        # Seleccionar órdenes
        selected_orders = []
        total_units = 0
        
        for i, _, units in order_efficiency:
            if total_units + units <= self.wave_size_ub:
                selected_orders.append(i)
                total_units += units
                
                # Si ya alcanzamos el mínimo, podemos parar o seguir añadiendo
                if total_units >= self.wave_size_lb:
                    # Añadir más si mejora la ratio (esto depende del problema)
                    continue
        
        # Si no alcanzamos el límite inferior, intentar con random
        if total_units < self.wave_size_lb:
            return self._generate_random_solution()
            
        # Determinar los pasillos necesarios
        visited_aisles = self._get_required_aisles(selected_orders)
        
        # Verificar si la solución es factible
        if not self.is_solution_feasible(selected_orders, visited_aisles):
            return self._generate_random_solution()
            
        return selected_orders, visited_aisles
    
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
    
    def get_neighbor_solution(self, selected_orders, visited_aisles, neighborhood_size=1):
        """
        Genera una solución vecina realizando pequeños cambios:
        1. Agregar/quitar una orden
        2. Intercambiar órdenes
        """
        # Hacer una copia para no modificar la original
        new_selected_orders = selected_orders.copy()
        
        # Elegir operación:
        # 1: Agregar una orden
        # 2: Quitar una orden
        # 3: Intercambiar órdenes
        operation = random.randint(1, 3)
        
        if operation == 1:  # Agregar
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
                
        elif operation == 2:  # Quitar
            if len(new_selected_orders) > 1:  # Asegurar que quede al menos una orden
                # Elegir aleatoriamente órdenes para quitar
                to_remove = random.sample(new_selected_orders, min(neighborhood_size, len(new_selected_orders) // 2))
                
                # Verificar si podemos quitar sin incumplir el límite inferior
                remaining_units = sum(self.order_units[i] for i in new_selected_orders) - sum(self.order_units[i] for i in to_remove)
                if remaining_units >= self.wave_size_lb:
                    new_selected_orders = [i for i in new_selected_orders if i not in to_remove]
                
        elif operation == 3:  # Intercambiar
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
        
        # Determinar los pasillos necesarios para la nueva selección de órdenes
        new_visited_aisles = self._get_required_aisles(new_selected_orders)
        
        return new_selected_orders, new_visited_aisles
    
    def solve_with_simulated_annealing(self, time_limit=600, initial_temp=100.0, cooling_rate=0.95, 
                                     iterations_per_temp=100, verbose=True):
        """
        Resuelve el problema usando el algoritmo de Recocido Simulado (Simulated Annealing)
        
        Args:
            time_limit: Límite de tiempo en segundos
            initial_temp: Temperatura inicial
            cooling_rate: Tasa de enfriamiento
            iterations_per_temp: Iteraciones por nivel de temperatura
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
        
        # Generar solución inicial
        if verbose:
            print("Generando solución inicial greedy...")
        solution = self.generate_initial_solution(strategy="greedy")
        
        if not solution:
            if verbose:
                print("No se pudo generar una solución inicial factible. Intentando con random...")
            solution = self.generate_initial_solution(strategy="random")
            
            if not solution:
                progress.stop()
                if verbose:
                    print("No se pudo generar una solución inicial factible.")
                return None, 0
        
        current_solution = solution
        best_solution = solution
        
        selected_orders, visited_aisles = current_solution
        current_value = self.evaluate_solution(selected_orders, visited_aisles)
        best_value = current_value
        
        if verbose:
            print(f"Solución inicial: Valor={current_value}, Órdenes={len(selected_orders)}, Pasillos={len(visited_aisles)}")
            
        # Inicializar temperatura
        temp = initial_temp
        
        # Estadísticas
        iterations = 0
        accepted_moves = 0
        improving_moves = 0
        
        while time.time() - start_time < time_limit:
            # Realizar iteraciones a esta temperatura
            for _ in range(iterations_per_temp):
                iterations += 1
                
                # Generar vecino
                neighbor_solution = self.get_neighbor_solution(selected_orders, visited_aisles)
                
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
                else:  # No mejora, aceptar con probabilidad
                    probability = math.exp(delta / temp)
                    if random.random() < probability:
                        accept = True
                
                if accept:
                    accepted_moves += 1
                    current_solution = neighbor_solution
                    selected_orders, visited_aisles = current_solution
                    current_value = neighbor_value
                    
                    # Actualizar mejor solución si corresponde
                    if current_value > best_value:
                        best_solution = current_solution
                        best_value = current_value
                        
                        if verbose:
                            elapsed = time.time() - start_time
                            print(f"[{elapsed:.1f}s] Nueva mejor solución: Valor={best_value}, "
                                  f"Órdenes={len(selected_orders)}, Pasillos={len(visited_aisles)}")
                
                # Verificar límite de tiempo
                if time.time() - start_time >= time_limit:
                    break
            
            # Enfriar
            temp *= cooling_rate
            
            # Actualizar barra de progreso
            acceptance_rate = accepted_moves / max(1, iterations)
            improvement_rate = improving_moves / max(1, iterations)
            
            progress.update_info(f"Mejor={best_value:.2f}, Temp={temp:.2f}, "
                                f"Acep={acceptance_rate:.2f}, Mej={improvement_rate:.2f}")
            
            # Detener si la temperatura es muy baja
            if temp < 0.01:
                if verbose:
                    print("Temperatura muy baja, terminando búsqueda.")
                break
        
        # Detener barra de progreso
        progress.stop()
        
        if verbose:
            print("\nEstadísticas del Recocido Simulado:")
            print(f"- Iteraciones totales: {iterations}")
            print(f"- Movimientos aceptados: {accepted_moves} ({accepted_moves/max(1, iterations):.2%})")
            print(f"- Movimientos de mejora: {improving_moves} ({improving_moves/max(1, iterations):.2%})")
            print(f"- Temperatura final: {temp:.4f}")
        
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
        print("Usage: python simulated_annealing_solver.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    print(f"\n{'='*50}")
    print(f"RESOLVIENDO INSTANCIA (RECOCIDO SIMULADO): {input_file}")
    print(f"{'='*50}")
    
    solver = WaveOrderPickingSASolver()
    solver.read_input(input_file)
    
    start_time = time.time()
    result = solver.solve_with_simulated_annealing(
        time_limit=600,
        initial_temp=100.0,
        cooling_rate=0.95,
        iterations_per_temp=100,
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