import sys
import time
import random
import math
import threading
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
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

class DecompositionSolver(WaveOrderPicking):
    def __init__(self):
        super().__init__()
        self.order_units = None       # Unidades por orden
        self.order_features = None    # Características para clustering
        self.aisle_features = None    # Características para clustering de pasillos
        self.item_locations = None    # Pasillos que contienen cada item
        self.order_clusters = None    # Asignación de órdenes a clusters
        self.aisle_clusters = None    # Asignación de pasillos a clusters
        self.cache = {}               # Caché para evaluaciones de soluciones
        
    def preprocess_data(self):
        """Preprocesa los datos para acelerar los cálculos y crear features para clustering"""
        # Calcular unidades por orden
        self.order_units = [sum(order.values()) for order in self.orders]
        
        # Para cada item, encontrar en qué pasillos está disponible
        self.item_locations = {}
        for item_id in set().union(*[set(order.keys()) for order in self.orders]):
            self.item_locations[item_id] = [
                j for j, aisle in enumerate(self.aisles) if item_id in aisle
            ]
        
        # Crear características para órdenes basadas en pasillos donde están sus items
        num_orders = len(self.orders)
        num_aisles = len(self.aisles)
        
        # Matriz para órdenes: cada fila es una orden, cada columna es un pasillo
        # El valor indica si la orden requiere items de ese pasillo
        self.order_features = np.zeros((num_orders, num_aisles))
        
        for i, order in enumerate(self.orders):
            for item_id in order:
                for aisle_id in self.item_locations.get(item_id, []):
                    self.order_features[i, aisle_id] = 1
                    
        # Matriz para pasillos: cada fila es un pasillo, cada columna es un tipo de item
        # El valor indica si el pasillo contiene ese tipo de item
        all_items = set().union(*[set(order.keys()) for order in self.orders])
        item_to_index = {item_id: idx for idx, item_id in enumerate(all_items)}
        num_items = len(all_items)
        
        self.aisle_features = np.zeros((num_aisles, num_items))
        for j, aisle in enumerate(self.aisles):
            for item_id in aisle:
                if item_id in item_to_index:
                    item_idx = item_to_index[item_id]
                    self.aisle_features[j, item_idx] = aisle[item_id]  # Usar cantidad como peso
    
    def cluster_orders(self, num_clusters=10, verbose=True):
        """
        Agrupa órdenes similares en clusters para su procesamiento por separado
        """
        if verbose:
            print(f"Agrupando órdenes en {num_clusters} clusters...")
        
        # Asegurar que tengamos un número razonable de clusters
        num_clusters = min(num_clusters, len(self.orders) // 10 + 1)
        
        # Aplicar K-means para agrupar órdenes similares
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        self.order_clusters = kmeans.fit_predict(self.order_features)
        
        if verbose:
            # Contar órdenes por cluster
            cluster_counts = defaultdict(int)
            for cluster_id in self.order_clusters:
                cluster_counts[cluster_id] += 1
            
            for cluster_id, count in sorted(cluster_counts.items()):
                print(f"  Cluster {cluster_id}: {count} órdenes")
    
    def cluster_aisles(self, num_clusters=5, verbose=True):
        """
        Agrupa pasillos similares en clusters
        """
        if verbose:
            print(f"Agrupando pasillos en {num_clusters} clusters...")
        
        # Asegurar que tengamos un número razonable de clusters
        num_clusters = min(num_clusters, len(self.aisles) // 2 + 1)
        
        # Aplicar K-means para agrupar pasillos similares
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        self.aisle_clusters = kmeans.fit_predict(self.aisle_features)
        
        if verbose:
            # Contar pasillos por cluster
            cluster_counts = defaultdict(int)
            for cluster_id in self.aisle_clusters:
                cluster_counts[cluster_id] += 1
            
            for cluster_id, count in sorted(cluster_counts.items()):
                print(f"  Cluster {cluster_id}: {count} pasillos")
    
    def solve_subproblem(self, order_indices, aisle_indices, 
                         time_limit=60, verbose=True, min_objective=None):
        """
        Resuelve un subproblema para un subconjunto de órdenes y pasillos
        
        Args:
            order_indices: Índices de órdenes a considerar
            aisle_indices: Índices de pasillos a considerar
            time_limit: Límite de tiempo en segundos
            verbose: Si es True, muestra información detallada
            min_objective: Valor mínimo objetivo que debe alcanzar la solución
            
        Returns:
            (selected_orders, visited_aisles, objective) o None si no hay solución factible
        """
        start_time = time.time()
        
        # Crear el solver
        if verbose:
            print(f"Creando solver SCIP para subproblema ({len(order_indices)} órdenes, {len(aisle_indices)} pasillos)...")
        solver = pywraplp.Solver.CreateSolver('SCIP')
        if not solver:
            print("No se pudo crear el solver SCIP")
            return None
            
        # Establecer límite de tiempo (en segundos)
        solver_time_limit = int((time_limit - (time.time() - start_time)) * 1000)
        solver.set_time_limit(solver_time_limit)
        
        # Variables: selección de órdenes y pasillos
        order_vars = {}
        for i in order_indices:
            order_vars[i] = solver.BoolVar(f'order_{i}')
            
        aisle_vars = {}
        for j in aisle_indices:
            aisle_vars[j] = solver.BoolVar(f'aisle_{j}')
        
        # Restricción: límites de tamaño de wave
        total_units = solver.Sum([self.order_units[i] * order_vars[i] for i in order_indices])
        solver.Add(total_units >= self.wave_size_lb)
        solver.Add(total_units <= self.wave_size_ub)
        
        # Restricción: todos los items requeridos deben estar disponibles
        for i in order_indices:
            for item_id, qty in self.orders[i].items():
                # Encontrar pasillos del subconjunto que tienen este item
                available_aisles = [j for j in aisle_indices if item_id in self.aisles[j]]
                
                if not available_aisles:
                    # Si ningún pasillo en el subconjunto tiene este item, forzar a no seleccionar esta orden
                    solver.Add(order_vars[i] == 0)
                    break
                
                # Total disponible para este item en los pasillos seleccionados
                available_expr = solver.Sum([
                    self.aisles[j].get(item_id, 0) * aisle_vars[j]
                    for j in available_aisles
                ])
                
                # Restricción: disponible >= requerido
                solver.Add(available_expr >= qty * order_vars[i])
        
        # Asegurar que al menos un pasillo sea visitado si hay órdenes seleccionadas
        total_orders = solver.Sum([order_vars[i] for i in order_indices])
        solver.Add(solver.Sum([aisle_vars[j] for j in aisle_indices]) >= total_orders / len(order_indices))
        
        # Restricción: Valor objetivo mínimo (opcional)
        if min_objective is not None:
            # Estimación del valor objetivo
            estimated_objective = (total_units / 
                                  solver.Sum([aisle_vars[j] for j in aisle_indices]))
            solver.Add(estimated_objective >= min_objective)
            
        # Función objetivo: maximizar ratio unidades/pasillos
        objective = solver.Objective()
        objective_scale = max(1, sum(self.order_units) // len(self.aisles))
        
        for i in order_indices:
            objective.SetCoefficient(order_vars[i], self.order_units[i])
        for j in aisle_indices:
            objective.SetCoefficient(aisle_vars[j], -objective_scale)
        objective.SetMaximization()
        
        # Resolver el modelo
        status = solver.Solve()
        
        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            # Obtener órdenes seleccionadas
            selected_orders = [i for i in order_indices if order_vars[i].solution_value() > 0.5]
            
            # Obtener pasillos visitados
            visited_aisles = [j for j in aisle_indices if aisle_vars[j].solution_value() > 0.5]
            
            # Calcular valor objetivo real
            total_units = sum(self.order_units[i] for i in selected_orders)
            objective_value = total_units / len(visited_aisles) if visited_aisles else 0
            
            if verbose:
                print(f"Solución para subproblema: Valor={objective_value:.2f}, "
                      f"Órdenes={len(selected_orders)}, Pasillos={len(visited_aisles)}")
                      
            return selected_orders, visited_aisles, objective_value
        else:
            if verbose:
                print("No se pudo encontrar una solución factible para el subproblema")
            return None
    
    def solve_with_decomposition(self, time_limit=600, num_order_clusters=10, 
                               num_aisle_clusters=5, verbose=True):
        """
        Resuelve el problema usando un enfoque de descomposición:
        1. Agrupa órdenes y pasillos en clusters
        2. Resuelve subproblemas para combinaciones de clusters
        3. Combina las soluciones manteniendo la factibilidad
        
        Args:
            time_limit: Límite de tiempo total
            num_order_clusters: Número de clusters para órdenes
            num_aisle_clusters: Número de clusters para pasillos
            verbose: Si es True, muestra información detallada
        """
        # Iniciar barra de progreso
        progress = ProgressBar(time_limit)
        progress.start()
        
        start_time = time.time()
        
        # Preprocesar datos
        if verbose:
            print("\nPreprocesando datos...")
        self.preprocess_data()
        
        # Agrupar órdenes y pasillos en clusters
        self.cluster_orders(num_clusters=num_order_clusters, verbose=verbose)
        self.cluster_aisles(num_clusters=num_aisle_clusters, verbose=verbose)
        
        # Las mejores soluciones encontradas
        best_solution = None
        best_objective = 0
        
        # Para cada combinación de clusters, resolver un subproblema
        order_cluster_ids = sorted(set(self.order_clusters))
        aisle_cluster_ids = sorted(set(self.aisle_clusters))
        
        # Enfoque 1: Resolver subproblemas para cada cluster de órdenes con todos los pasillos
        if verbose:
            print("\nEnfoque 1: Resolución por clusters de órdenes...")
            
        for oc_id in order_cluster_ids:
            if time.time() - start_time >= time_limit:
                break
                
            # Obtener órdenes de este cluster
            order_indices = [i for i, cluster in enumerate(self.order_clusters) if cluster == oc_id]
            
            # Usar todos los pasillos
            aisle_indices = list(range(len(self.aisles)))
            
            # Resolver subproblema
            time_remaining = time_limit - (time.time() - start_time)
            subproblem_time = min(60, time_remaining * 0.1)  # Máx 10% del tiempo restante
            
            if verbose:
                print(f"\nResolviendo para cluster de órdenes {oc_id} "
                      f"({len(order_indices)} órdenes, {len(aisle_indices)} pasillos)")
                
            result = self.solve_subproblem(
                order_indices, 
                aisle_indices,
                time_limit=subproblem_time,
                verbose=verbose
            )
            
            if result:
                selected_orders, visited_aisles, objective = result
                
                # Actualizar mejor solución
                if objective > best_objective:
                    best_solution = (selected_orders, visited_aisles)
                    best_objective = objective
                    
                    if verbose:
                        print(f"Nueva mejor solución: Valor={objective:.2f}, "
                              f"Órdenes={len(selected_orders)}, Pasillos={len(visited_aisles)}")
                              
            # Actualizar barra de progreso
            elapsed = time.time() - start_time
            progress.update_info(f"Mejor={best_objective:.2f}, Enfoque=1, "
                                f"OCluster={oc_id}/{len(order_cluster_ids)}")
        
        # Enfoque 2: Resolver subproblemas combinando clusters de órdenes con clusters de pasillos
        if verbose:
            print("\nEnfoque 2: Combinación de clusters de órdenes y pasillos...")
            
        for oc_id in order_cluster_ids:
            if time.time() - start_time >= time_limit:
                break
                
            # Obtener órdenes de este cluster
            order_indices = [i for i, cluster in enumerate(self.order_clusters) if cluster == oc_id]
            
            # Para cada cluster de pasillos
            for ac_id in aisle_cluster_ids:
                if time.time() - start_time >= time_limit:
                    break
                    
                # Obtener pasillos de este cluster
                aisle_indices = [j for j, cluster in enumerate(self.aisle_clusters) if cluster == ac_id]
                
                # Resolver subproblema
                time_remaining = time_limit - (time.time() - start_time)
                subproblem_time = min(30, time_remaining * 0.05)  # Máx 5% del tiempo restante
                
                if verbose:
                    print(f"\nResolviendo para cluster de órdenes {oc_id} y pasillos {ac_id} "
                          f"({len(order_indices)} órdenes, {len(aisle_indices)} pasillos)")
                    
                result = self.solve_subproblem(
                    order_indices, 
                    aisle_indices,
                    time_limit=subproblem_time,
                    verbose=verbose
                )
                
                if result:
                    selected_orders, visited_aisles, objective = result
                    
                    # Actualizar mejor solución
                    if objective > best_objective:
                        best_solution = (selected_orders, visited_aisles)
                        best_objective = objective
                        
                        if verbose:
                            print(f"Nueva mejor solución: Valor={objective:.2f}, "
                                  f"Órdenes={len(selected_orders)}, Pasillos={len(visited_aisles)}")
                                  
                # Actualizar barra de progreso
                elapsed = time.time() - start_time
                progress.update_info(f"Mejor={best_objective:.2f}, Enfoque=2, "
                                    f"OCluster={oc_id}/{len(order_cluster_ids)}, "
                                    f"ACluster={ac_id}/{len(aisle_cluster_ids)}")
        
        # Enfoque 3: Resolver con combinaciones de múltiples clusters de órdenes
        if verbose:
            print("\nEnfoque 3: Combinación de múltiples clusters de órdenes...")
        
        # Elegir las mejores combinaciones de clusters de órdenes
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time > 60 and len(order_cluster_ids) > 1:
            # Combinar 2 clusters de órdenes
            for i, oc_id1 in enumerate(order_cluster_ids[:-1]):
                for oc_id2 in order_cluster_ids[i+1:]:
                    if time.time() - start_time >= time_limit:
                        break
                        
                    # Obtener órdenes de ambos clusters
                    order_indices = [
                        i for i, cluster in enumerate(self.order_clusters) 
                        if cluster == oc_id1 or cluster == oc_id2
                    ]
                    
                    # Usar todos los pasillos
                    aisle_indices = list(range(len(self.aisles)))
                    
                    # Resolver subproblema
                    time_remaining = time_limit - (time.time() - start_time)
                    subproblem_time = min(90, time_remaining * 0.15)  # Máx 15% del tiempo restante
                    
                    if verbose:
                        print(f"\nResolviendo para clusters de órdenes {oc_id1}+{oc_id2} "
                              f"({len(order_indices)} órdenes, {len(aisle_indices)} pasillos)")
                        
                    result = self.solve_subproblem(
                        order_indices, 
                        aisle_indices,
                        time_limit=subproblem_time,
                        verbose=verbose
                    )
                    
                    if result:
                        selected_orders, visited_aisles, objective = result
                        
                        # Actualizar mejor solución
                        if objective > best_objective:
                            best_solution = (selected_orders, visited_aisles)
                            best_objective = objective
                            
                            if verbose:
                                print(f"Nueva mejor solución: Valor={objective:.2f}, "
                                      f"Órdenes={len(selected_orders)}, Pasillos={len(visited_aisles)}")
                                      
                    # Actualizar barra de progreso
                    elapsed = time.time() - start_time
                    progress.update_info(f"Mejor={best_objective:.2f}, Enfoque=3, "
                                        f"Combinando clusters {oc_id1}+{oc_id2}")
        
        # Enfoque 4: Intento con clusters específicos y optimizaciones híbridas
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time > 120 and best_solution:
            if verbose:
                print("\nEnfoque 4: Optimizaciones híbridas específicas...")
            
            # Usar la mejor solución encontrada hasta ahora como base
            selected_orders, visited_aisles = best_solution
            
            # Intentar mejorar la solución
            time_remaining = time_limit - (time.time() - start_time)
            
            # Identificar órdenes que podrían agregarse
            current_units = sum(self.order_units[i] for i in selected_orders)
            remaining_capacity = self.wave_size_ub - current_units
            
            # Órdenes candidatas para agregar (no seleccionadas actualmente)
            unselected_orders = [i for i in range(len(self.orders)) if i not in selected_orders]
            
            # Ordenar por eficiencia (unidades por pasillo adicional requerido)
            order_efficiency = []
            for i in unselected_orders:
                if self.order_units[i] <= remaining_capacity:
                    # Calcular pasillos adicionales necesarios
                    required_aisles = set()
                    for item_id in self.orders[i]:
                        for aisle_id in self.item_locations.get(item_id, []):
                            if aisle_id not in visited_aisles:
                                required_aisles.add(aisle_id)
                    
                    # Eficiencia = unidades / pasillos adicionales requeridos
                    if required_aisles:
                        efficiency = self.order_units[i] / len(required_aisles)
                    else:
                        # Muy alta eficiencia si no requiere pasillos adicionales
                        efficiency = self.order_units[i] * 10
                    
                    order_efficiency.append((i, efficiency, len(required_aisles)))
            
            # Ordenar por eficiencia decreciente
            order_efficiency.sort(key=lambda x: x[1], reverse=True)
            
            # Intentar agregar las órdenes más eficientes
            candidate_orders = selected_orders.copy()
            candidate_aisles = visited_aisles.copy()
            
            orders_added = False
            for order_id, _, _ in order_efficiency[:min(100, len(order_efficiency))]:
                # Verificar si aún tenemos capacidad
                candidate_units = sum(self.order_units[i] for i in candidate_orders)
                if candidate_units + self.order_units[order_id] <= self.wave_size_ub:
                    # Agregar orden
                    candidate_orders.append(order_id)
                    
                    # Calcular los pasillos necesarios
                    new_aisles = self._get_required_aisles(candidate_orders)
                    
                    # Verificar si la solución es factible
                    if self.is_solution_feasible(candidate_orders, new_aisles):
                        candidate_aisles = new_aisles
                        orders_added = True
                        
                        if verbose:
                            print(f"Orden {order_id} agregada a la solución")
            
            if orders_added:
                # Calcular nuevo valor objetivo
                candidate_units = sum(self.order_units[i] for i in candidate_orders)
                candidate_objective = candidate_units / len(candidate_aisles) if candidate_aisles else 0
                
                if candidate_objective > best_objective:
                    best_solution = (candidate_orders, candidate_aisles)
                    best_objective = candidate_objective
                    
                    if verbose:
                        print(f"Solución mejorada con enfoque 4: Valor={best_objective:.2f}, "
                              f"Órdenes={len(candidate_orders)}, Pasillos={len(candidate_aisles)}")
        
        # Detener barra de progreso
        progress.stop()
        
        # Retornar la mejor solución encontrada
        if best_solution:
            return best_solution, best_objective
        else:
            return None, 0
    
    def optimize_aisles(self, selected_orders, visited_aisles):
        """
        Intenta reducir el número de pasillos visitados manteniendo las órdenes.
        Usa un algoritmo greedy para seleccionar pasillos más eficientes.
        """
        # Recolectar los items requeridos y sus cantidades
        required_items = {}
        for i in selected_orders:
            for item_id, qty in self.orders[i].items():
                required_items[item_id] = required_items.get(item_id, 0) + qty
        
        # Inicializar pasillos visitados y items pendientes
        optimized_aisles = []
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
                optimized_aisles.append(j)
                
        # Verificar si la solución es factible con los pasillos optimizados
        if self.is_solution_feasible(selected_orders, optimized_aisles):
            # Si encontramos una selección de pasillos más pequeña, usarla
            if len(optimized_aisles) < len(visited_aisles):
                return optimized_aisles
        
        # Si la optimización no mejoró o no es factible, mantener los pasillos originales
        return visited_aisles
    
    def _get_required_aisles(self, selected_orders):
        """
        Determina los pasillos necesarios para cubrir todos los items
        de las órdenes seleccionadas.
        """
        return self.optimize_aisles(selected_orders, [])  # Optimizar desde cero
    
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
        print("Usage: python decomposition_solver.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    print(f"\n{'='*50}")
    print(f"RESOLVIENDO INSTANCIA (ENFOQUE DESCOMPOSICIÓN): {input_file}")
    print(f"{'='*50}")
    
    solver = DecompositionSolver()
    solver.read_input(input_file)
    
    # Determinar número de clusters basado en el tamaño de la instancia
    num_orders = len(solver.orders)
    num_aisles = len(solver.aisles)
    
    # Más clusters para instancias más grandes
    num_order_clusters = max(5, min(20, num_orders // 100))
    num_aisle_clusters = max(3, min(10, num_aisles // 10))
    
    print(f"Instancia: {num_orders} órdenes, {num_aisles} pasillos")
    print(f"Configuración: {num_order_clusters} clusters de órdenes, {num_aisle_clusters} clusters de pasillos")
    
    start_time = time.time()
    # Utilizar enfoque de descomposición
    result = solver.solve_with_decomposition(
        time_limit=600,  # 10 minutos
        num_order_clusters=num_order_clusters,
        num_aisle_clusters=num_aisle_clusters,
        verbose=True
    )
    end_time = time.time()
    
    print(f"\n{'='*50}")
    print(f"RESULTADOS:")
    print(f"{'='*50}")
    
    if result[0] is not None:
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