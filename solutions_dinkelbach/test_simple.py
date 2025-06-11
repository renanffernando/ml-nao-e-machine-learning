#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dinkelbach_solver import DinkelbachSolver

def main():
    try:
        print("Testing Dinkelbach solver...")
        solver = DinkelbachSolver('instance_0002.txt')
        print("Solver created successfully")
        x, y, ratio = solver.solve()
        print(f"Solution found: ratio = {ratio}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 