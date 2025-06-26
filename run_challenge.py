import os
import subprocess
import sys
import platform
import time

# Add CPLEX Python path
CPLEX_PYTHON_PATH = r"C:\Program Files\IBM\ILOG\CPLEX_Studio_Community2212\cplex\python\3.7\x64_win32"
if CPLEX_PYTHON_PATH not in sys.path:
    sys.path.append(CPLEX_PYTHON_PATH)

# Paths to the libraries
CPLEX_PATH = r"C:\Program Files\IBM\ILOG\CPLEX_Studio_Community2212\cplex\bin\x64_win32"
OR_TOOLS_PATH = "$HOME/Documents/or-tools/build/lib/"

USE_CPLEX = True
USE_OR_TOOLS = True

MAX_RUNNING_TIME = 605  # seconds

def compile_java_code(source_folder):
    print(f"Compiling Java code in {source_folder}...")
    # Change to the source folder
    os.chdir(source_folder)

    # Run Maven compile
    result = subprocess.run(["mvn", "clean", "package"], capture_output=True, text=True)

    if result.returncode != 0:
        print("Maven compilation failed:")
        print(result.stderr)
        return False

    print("Maven compilation successful.")
    return True

def run_java_solver(source_folder, input_file, output_file):
    # Change to the source folder
    os.chdir(source_folder)

    # Set the library path (if needed)
    if USE_CPLEX and USE_OR_TOOLS:
        libraries = f"{OR_TOOLS_PATH}:{CPLEX_PATH}"
    elif USE_CPLEX:
        libraries = CPLEX_PATH
    elif USE_OR_TOOLS:
        libraries = OR_TOOLS_PATH

    if platform.system() == "Darwin":
        timeout_command = "gtimeout"
    else:
        timeout_command = "timeout"

    # Main Java command
    cmd = [timeout_command, f"{MAX_RUNNING_TIME}s", "java", "-Xmx16g", "-jar", "target/ChallengeSBPO2025-1.0.jar",
           input_file,
           output_file]
    if USE_CPLEX or USE_OR_TOOLS:
        cmd.insert(3, f"-Djava.library.path={libraries}")

    result = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)
    return result.returncode == 0

def run_python_solver(solver_path, input_file, output_file):
    print(f"Running Python solver on {os.path.basename(input_file)}")
    start_time = time.time()
    
    # Run the Python solver with timeout
    cmd = [sys.executable, solver_path, input_file, output_file]
    try:
        result = subprocess.run(cmd, stderr=subprocess.PIPE, text=True, timeout=MAX_RUNNING_TIME)
        
        elapsed_time = time.time() - start_time
        if result.returncode == 0:
            print(f"Solved in {elapsed_time:.2f} seconds")
            return True
        else:
            print(f"Failed to solve {os.path.basename(input_file)}:")
            print(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print(f"Timeout after {MAX_RUNNING_TIME} seconds")
        return False

def run_benchmark(source_folder, input_folder, output_folder, solver_type="java"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Detect solver file for Python
    if solver_type == "python":
        solver_path = os.path.join(source_folder, "src", "solvers", "exact", "dinkelbach_solver.py")
        if not os.path.exists(solver_path):
            print(f"Error: Solver not found at {solver_path}")
            return False

    successful_instances = 0
    total_instances = 0

    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            total_instances += 1
            print(f"\nProcessing instance {filename}")
            input_file = os.path.join(input_folder, filename)
            output_file = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.txt")
            
            if solver_type == "java":
                success = run_java_solver(source_folder, input_file, output_file)
            else:  # python
                success = run_python_solver(solver_path, input_file, output_file)
            
            if success:
                successful_instances += 1
                print(f"Successfully solved {filename}")
            else:
                print(f"Failed to solve {filename}")

    print(f"\nBenchmark completed:")
    print(f"Successfully solved {successful_instances} out of {total_instances} instances")
    return successful_instances == total_instances

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python run_challenge.py <source_folder> <input_folder> <output_folder> [solver_type]")
        print("solver_type can be 'java' or 'python' (default: java)")
        sys.exit(1)

    source_folder = sys.argv[1]
    input_folder = sys.argv[2]
    output_folder = sys.argv[3]
    solver_type = sys.argv[4] if len(sys.argv) > 4 else "java"

    if solver_type not in ["java", "python"]:
        print("Error: solver_type must be 'java' or 'python'")
        sys.exit(1)

    if solver_type == "java" and not compile_java_code(source_folder):
        sys.exit(1)

    run_benchmark(source_folder, input_folder, output_folder, solver_type)
