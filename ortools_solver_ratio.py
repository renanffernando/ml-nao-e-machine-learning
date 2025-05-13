import sys
import time
import threading
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

class WaveOrderPickingRatioSolver(WaveOrderPicking):
    def __init__(self):
        super().__init__()
        
    def solve_with_ratio_mip(self, time_limit=600, verbose=True):
        """
        Resuelve el problema usando programación lineal entera mixta con una formulación directa,
        sin usar la técnica de transformación de ratios.
        """
        # Iniciar la barra de progreso
        progress = ProgressBar(time_limit)
        progress.start()
        
        start_time = time.time()
        
        # Crear el solver
        if verbose:
            print("\nCreando solver SCIP...")
        solver = pywraplp.Solver.CreateSolver('SCIP')
        if not solver:
            progress.stop()
            print("No se pudo crear el solver SCIP")
            return None, 0
            
        # Establecer límite de tiempo (en segundos)
        solver_time_limit = int((time_limit - (time.time() - start_time)) * 1000)
        solver.set_time_limit(solver_time_limit)
        if verbose:
            print(f"Límite de tiempo para el solver: {solver_time_limit/1000:.2f} segundos")
        
        # Variables: selección de órdenes y pasillos
        if verbose:
            print(f"Creando variables para {len(self.orders)} órdenes y {len(self.aisles)} pasillos...")
        
        order_vars = {}
        for i in range(len(self.orders)):
            order_vars[i] = solver.BoolVar(f'order_{i}')
            
        aisle_vars = {}
        for j in range(len(self.aisles)):
            aisle_vars[j] = solver.BoolVar(f'aisle_{j}')
        
        progress.update_info(f"Variables creadas: {len(order_vars) + len(aisle_vars)}")
            
        # Restricción: límites de tamaño de wave
        if verbose:
            print(f"Límites de tamaño de wave: [{self.wave_size_lb}, {self.wave_size_ub}]")
        
        # Calcular unidades por orden
        order_units = [sum(self.orders[i].values()) for i in range(len(self.orders))]
        if verbose:
            print(f"Unidades promedio por orden: {sum(order_units)/len(order_units):.2f}")
        
        # Restricción: total de unidades = suma de unidades de órdenes seleccionadas
        if verbose:
            print("Añadiendo restricción de unidades totales...")
        
        total_units = solver.Sum([order_units[i] * order_vars[i] for i in range(len(self.orders))])
        solver.Add(total_units >= self.wave_size_lb)
        solver.Add(total_units <= self.wave_size_ub)
        
        # Restricción: todos los items requeridos deben estar disponibles
        all_items = set()
        for order in self.orders:
            all_items.update(order.keys())
        
        if verbose:
            print(f"Añadiendo restricciones para {len(all_items)} items diferentes...")
            
        num_constraints = 2  # Ya tenemos 2 restricciones (LB, UB)
            
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
            num_constraints += 1
            
        progress.update_info(f"Restricciones base: {num_constraints}")
            
        # Restricción: si una orden está seleccionada, necesitamos visitar al menos un pasillo que tenga cada item
        if verbose:
            print("Añadiendo restricciones de conectividad entre órdenes y pasillos...")
            
        for i in range(len(self.orders)):
            for item in self.orders[i]:
                # Pasillos que contienen este item
                relevant_aisles = [j for j in range(len(self.aisles)) if item in self.aisles[j]]
                if relevant_aisles:
                    solver.Add(
                        solver.Sum([aisle_vars[j] for j in relevant_aisles]) >= order_vars[i]
                    )
                    num_constraints += 1
                    
        # Asegurar que al menos un pasillo sea visitado si hay órdenes seleccionadas
        solver.Add(solver.Sum([aisle_vars[j] for j in range(len(self.aisles))]) >= 
                  solver.Sum([order_vars[i] for i in range(len(self.orders))]) / len(self.orders))
        num_constraints += 1
                    
        # Restricción: un pasillo solo se visita si tiene items requeridos por las órdenes seleccionadas
        for j in range(len(self.aisles)):
            items_in_aisle = set(self.aisles[j].keys())
            relevant_orders = []
            for i in range(len(self.orders)):
                items_in_order = set(self.orders[i].keys())
                if items_in_aisle.intersection(items_in_order):
                    relevant_orders.append(i)
            
            if relevant_orders:
                # El pasillo j solo puede ser visitado si al menos una orden relevante está seleccionada
                solver.Add(
                    aisle_vars[j] <= solver.Sum([order_vars[i] for i in relevant_orders])
                )
                num_constraints += 1
        
        if verbose:
            print(f"Total de restricciones: {num_constraints}")
        
        progress.update_info(f"Total restricciones: {num_constraints}")
        
        # Definición de la variable para el número de pasillos visitados
        num_aisles = solver.Sum([aisle_vars[j] for j in range(len(self.aisles))])
        
        # Definición para el uso con el modelo de Charnes-Cooper
        # (https://lpsolve.sourceforge.net/5.5/ratio.htm)
        # Objetivo: maximizar total_units / num_aisles
        # Implementamos una aproximación: 
        # max total_units - K * num_aisles
        # donde K es un factor de escala para balancear los términos
        K = sum(order_units) / len(self.aisles)  # Valor promedio de las unidades por orden / pasillos
        
        if verbose:
            print(f"Configurando función objetivo: max total_units - {K} * num_aisles")
            
        objective = solver.Objective()
        for i in range(len(self.orders)):
            objective.SetCoefficient(order_vars[i], order_units[i])
        for j in range(len(self.aisles)):
            objective.SetCoefficient(aisle_vars[j], -K)
        objective.SetMaximization()
        
        # Resolver el modelo
        if verbose:
            print("\nResolviendo el modelo...")
        
        progress.update_info("Resolviendo...")
        status = solver.Solve()
        
        # Detener la barra de progreso
        progress.stop()
        
        if status == pywraplp.Solver.OPTIMAL:
            status_str = "ÓPTIMA"
        elif status == pywraplp.Solver.FEASIBLE:
            status_str = "FACTIBLE"
        elif status == pywraplp.Solver.INFEASIBLE:
            status_str = "INFACTIBLE"
        elif status == pywraplp.Solver.UNBOUNDED:
            status_str = "ILIMITADA"
        elif status == pywraplp.Solver.ABNORMAL:
            status_str = "ANORMAL"
        elif status == pywraplp.Solver.MODEL_INVALID:
            status_str = "MODELO INVÁLIDO"
        elif status == pywraplp.Solver.NOT_SOLVED:
            status_str = "NO RESUELTO"
        else:
            status_str = f"DESCONOCIDO ({status})"
            
        if verbose:
            print(f"Estado de la solución: {status_str}")
            print(f"Tiempo de solución: {solver.WallTime()/1000:.2f} segundos")
        
        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            # Obtener órdenes seleccionadas
            selected_orders = [i for i in range(len(self.orders)) if order_vars[i].solution_value() > 0.5]
            
            # Obtener pasillos visitados
            visited_aisles = [j for j in range(len(self.aisles)) if aisle_vars[j].solution_value() > 0.5]
            
            if verbose:
                total_selected_units = sum(order_units[i] for i in selected_orders)
                print(f"Órdenes seleccionadas: {len(selected_orders)}")
                print(f"Pasillos visitados: {len(visited_aisles)}")
                print(f"Unidades totales: {total_selected_units}")
                
            # Si por alguna razón no hay pasillos visitados pero hay órdenes seleccionadas,
            # incluir los pasillos necesarios
            if not visited_aisles and selected_orders:
                if verbose:
                    print("Aviso: No hay pasillos visitados, añadiendo los necesarios...")
                
                # Añadir todos los pasillos que tienen items requeridos por las órdenes seleccionadas
                required_items = {}
                for i in selected_orders:
                    for item, qty in self.orders[i].items():
                        required_items[item] = required_items.get(item, 0) + qty
                
                # Añadir pasillos de forma greedy hasta satisfacer todos los items
                remaining_items = required_items.copy()
                potential_aisles = [(j, sum(self.aisles[j].get(item, 0) for item in remaining_items)) 
                                   for j in range(len(self.aisles))]
                potential_aisles.sort(key=lambda x: x[1], reverse=True)
                
                for j, _ in potential_aisles:
                    if not remaining_items:
                        break
                        
                    # Ver si este pasillo proporciona algún item requerido
                    items_provided = False
                    for item in list(remaining_items.keys()):
                        if item in self.aisles[j]:
                            qty_available = self.aisles[j][item]
                            if qty_available > 0:
                                items_provided = True
                                remaining_items[item] = max(0, remaining_items[item] - qty_available)
                                if remaining_items[item] == 0:
                                    del remaining_items[item]
                    
                    if items_provided:
                        visited_aisles.append(j)
                
                if verbose:
                    print(f"Pasillos añadidos manualmente: {len(visited_aisles)}")
            
            # Calcular el valor objetivo verdadero
            if visited_aisles:
                objective_value = self.compute_objective_function(selected_orders, visited_aisles)
                if verbose:
                    print(f"Valor objetivo: {objective_value}")
                return (selected_orders, visited_aisles), objective_value
        else:
            if verbose:
                print("No se encontró una solución factible")
        
        return None, 0
        
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
        print("Usage: python ortools_solver_ratio.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    print(f"\n{'='*50}")
    print(f"RESOLVIENDO INSTANCIA (MÉTODO RATIO): {input_file}")
    print(f"{'='*50}")
    
    solver = WaveOrderPickingRatioSolver()
    solver.read_input(input_file)
    
    start_time = time.time()
    result = solver.solve_with_ratio_mip(time_limit=600, verbose=True)
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