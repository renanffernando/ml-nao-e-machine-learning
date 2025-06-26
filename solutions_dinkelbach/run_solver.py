import sys
sys.path.append('.')

from src.solvers.exact.dinkelbach_solver import DinkelbachSolver

def main():
    solver = DinkelbachSolver('instance_0001.txt')
    solver.solve()

if __name__ == '__main__':
    main() 