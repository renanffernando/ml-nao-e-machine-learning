import os
import subprocess
import sys
import platform

# Paths to the CPLEX library
CPLEX_PATH = "/opt/ibm/ILOG/CPLEX_Studio2211/cplex/bin/x86-64_linux/"

MAX_RUNNING_TIME = "605s"


class TimeoutError(Exception):
    """Custom exception for timeout errors."""
    pass


def compile_code(source_folder):
    print(f"Compiling code in {source_folder}...")

    # Run Maven compile without changing the directory
    result = subprocess.run(
        ["mvn", "clean", "package"],
        capture_output=True,
        text=True,
        cwd=source_folder
    )

    if result.returncode != 0:
        print("Maven compilation failed:")
        print(result.stderr)
        return False

    print("Maven compilation successful.")
    return True


def run_benchmark(source_folder, input_folder, output_folder):
    # Make sure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Set the CPLEX library path
    libraries = CPLEX_PATH

    if platform.system() == "Darwin":
        timeout_command = "gtimeout"
    else:
        timeout_command = "timeout"

    # Get the path to the shaded (fat) JAR file
    jar_path = os.path.join(source_folder, "target", "ChallengeSBPO2025-1.0.jar")

    for filename in sorted(os.listdir(input_folder)):
        if filename.endswith(".txt"):
            print(f"Running {filename}", 100 * "-")
            input_file = os.path.join(input_folder, filename)
            output_file = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.txt")

            # Main Java command
            cmd = [
                timeout_command,
                MAX_RUNNING_TIME,
                "java",
                "-Xmx16g",
                "-jar",
                jar_path,
                input_file,
                output_file
            ]

            cmd.insert(3, f"-Djava.library.path={libraries}")

            result = subprocess.run(
                cmd,
                stderr=subprocess.PIPE,
                text=True,
                cwd=source_folder  # Set a working directory directly
            )

            # Check for timeout (return code 124 is the standard timeout exit code)
            if result.returncode == 124:
                error_msg = f"Execution timed out after {MAX_RUNNING_TIME} for {input_file}"
                print(error_msg)
                raise TimeoutError(error_msg)
            elif result.returncode != 0:
                print(f"Execution failed for {input_file}:")
                print(result.stderr)
                raise RuntimeError(f"Execution failed for {input_file}: {result.stderr}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python run_challenge.py <source_folder> <input_folder> <output_folder>")
        sys.exit(1)

    source_folder = os.path.abspath(sys.argv[1])
    input_folder = os.path.abspath(sys.argv[2])
    output_folder = os.path.abspath(sys.argv[3])

    if compile_code(source_folder):
        run_benchmark(source_folder, input_folder, output_folder)
