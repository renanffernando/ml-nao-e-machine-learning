#!/usr/bin/env python3

import time
import json
from dinkelbach_solver import DinkelbachSolver

def main():
    print("Ejecutando algoritmo de Dinkelbach con warm start en instancia 10...")
    
    start_time = time.time()
    
    try:
        solver = DinkelbachSolver('instance_0010.txt')
        x, y, ratio = solver.solve()
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Preparar resultados
        results = {
            "instance": "instance_0010.txt",
            "optimal_ratio": float(ratio),
            "execution_time_seconds": execution_time,
            "selected_orders": [i for i, xi in enumerate(x) if xi > 0.5] if x else [],
            "used_aisles": [j for j, yj in enumerate(y) if yj > 0.5] if y else [],
            "solution_x": x if x else [],
            "solution_y": y if y else [],
            "num_selected_orders": len([i for i, xi in enumerate(x) if xi > 0.5]) if x else 0,
            "num_used_aisles": len([j for j, yj in enumerate(y) if yj > 0.5]) if y else 0
        }
        
        # Guardar en archivo JSON
        with open('results_instance10.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Guardar resumen en archivo de texto
        with open('results_instance10.txt', 'w') as f:
            f.write(f"RESULTADOS ALGORITMO DINKELBACH - INSTANCIA 10\n")
            f.write(f"=" * 50 + "\n\n")
            f.write(f"Ratio óptimo: {ratio:.6f}\n")
            f.write(f"Tiempo de ejecución: {execution_time:.2f} segundos\n")
            f.write(f"Órdenes seleccionadas: {len([i for i, xi in enumerate(x) if xi > 0.5]) if x else 0}\n")
            f.write(f"Pasillos usados: {len([j for j, yj in enumerate(y) if yj > 0.5]) if y else 0}\n\n")
            
            if x:
                f.write(f"Órdenes seleccionadas: {[i for i, xi in enumerate(x) if xi > 0.5]}\n\n")
                f.write(f"Pasillos usados: {[j for j, yj in enumerate(y) if yj > 0.5]}\n\n")
        
        print(f"\nRESULTADOS FINALES:")
        print(f"Ratio óptimo: {ratio:.6f}")
        print(f"Tiempo: {execution_time:.2f} segundos")
        print(f"Órdenes seleccionadas: {len([i for i, xi in enumerate(x) if xi > 0.5]) if x else 0}")
        print(f"Pasillos usados: {len([j for j, yj in enumerate(y) if yj > 0.5]) if y else 0}")
        print(f"\nResultados guardados en:")
        print(f"- results_instance10.json (datos completos)")
        print(f"- results_instance10.txt (resumen)")
        
    except Exception as e:
        end_time = time.time()
        execution_time = end_time - start_time
        
        error_results = {
            "instance": "instance_0010.txt",
            "error": str(e),
            "execution_time_seconds": execution_time
        }
        
        with open('results_instance10_error.json', 'w') as f:
            json.dump(error_results, f, indent=2)
        
        print(f"ERROR: {e}")
        print(f"Tiempo hasta error: {execution_time:.2f} segundos")
        print(f"Error guardado en results_instance10_error.json")

if __name__ == '__main__':
    main() 