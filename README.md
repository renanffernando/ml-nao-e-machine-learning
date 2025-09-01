# challenge-sbpo-2025

Welcome to the Mercado Libre First Optimization Challenge repository! This challenge is part of the [LVII Brazilian Symposium on Operations Research (SBPO 2025)](https://sbpo2025.galoa.com.br/sbpo-2025/page/5407-home). For further details, please read the post on Medium ([Portuguese version](https://medium.com/mercadolibre-tech/desafio-mercado-livre-de-otimiza%C3%A7%C3%A3o-3a4009607ee3); [Spanish version](https://medium.com/mercadolibre-tech/primer-desaf%C3%ADo-mercado-libre-de-optimizaci%C3%B3n-e8dad236054c)).
In this repository, you will find the base code for the framework, documentation, and other resources related to the challenge.

## Change Log

- **05-03-2025**: Updated the challenge rules to clarify that, due to allowing a multithread environment, no seed for random generation will be provided.
- **27-02-2025**: Updated the challenge rules to include specific details of the computer environment in which the challenge will be run.
- **21-02-2025**: Corrected OR-Tools version to 9.11.
- **17-01-2025**: Base framework code, documentation and dataset `A`.

## Challenge Rules and Problem Description

Spanish and Portuguese versions of the challenge rules and problem description can be found in the `docs` directory:

- **Spanish**:
  - [Problem description](docs/es_problem_description.pdf)
  - [Challenge rules](docs/es_challenge_rules.pdf)


- **Portuguese**:
  - [Problem description](docs/pt_problem_description.pdf)
  - [Challenge rules](docs/pt_challenge_rules.pdf)

## Project Structure

- `src/main/java/org/sbpo2025/challenge`
  - `Challenge.java` ⟶ Main Java class for reading an input, solving the challenge, and writing the output.
  - `ChallengeSolver.java` ⟶ Java class responsible for solving the wave order picking problem. Most of the solving logic should be implemented here.
  - `ChallengeSolution.java` ⟶ Java class representing the solution to the wave order picking problem.
- `datasets/` ⟶ Directory containing input instance files.
- `run_challenge.py` ⟶ Python script to compile code, run benchmarks, and evaluate solutions.
- `checker.py` ⟶ Python script for evaluating the feasibility and objective value of solutions.

## Prerequisites

- Java 11
- Maven
- Python 3.8 or higher
- CPLEX 22.11 (optional)
- OR-Tools 9.11 (optional)

## Setup

1. Clone the repository:
    ```sh
    git clone https://github.com/mercadolibre/challenge-sbpo-2025
    ```
2. Set the paths to CPLEX and OR-Tools libraries in `run_challenge.py` if needed, e.g.:
    ```sh
    cplex_path = "$HOME/CPLEX_Studio2211/opl/bin/arm64_osx/"
    or_tools_path = "$HOME/Documents/or-tools/build/lib/"
    ```

## Usage

### Running the challenge

To compile the code and run benchmarks, use the following command:
```sh
python run_challenge.py <source_folder> <input_folder> <output_folder>
```
Where `<source_folder>` is the path to the Java source code, more specifically, where the `pom.xml` file is located.

In order to run this script you will need the `timeout` (or `gtimeout` on macOS) command installed. You can install it using `apt-get install coreutils` (or equivalent) on Linux or `brew install coreutils` on macOS.

### Checking solution viability

To check the feasibility and objective value of a solution, use the following command:
```sh
python checker.py <input_file> <solution_file>
```

## Examples

1. Compile and run benchmarks:
    ```sh
    python run_challenge.py . datasets/a output/a
    ```
   
2. Check solution viability:
    ```sh
    python checker.py src/main/resources/instances/instance_001.txt output/instance_001.txt
    ```
