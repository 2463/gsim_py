# gsim_py

`gsim_py` is a Python library that provides a high-performance interface to the `gsim` quantum simulation engine written in Rust. It is designed for efficient simulation of quantum systems, leveraging the speed of Rust for the underlying computations while providing a convenient Python API.

## Important Notice: Dependency on `gsim`

Before you can install or use `gsim_py`, you must have the [`gsim`](https://github.com/2463/gsim/) project available on your local machine. `gsim_py` depends on `gsim` as a local path dependency.

Please ensure that you have cloned the `gsim` repository and that its directory is located at the same level as the `gsim_py` directory. The expected directory structure is as follows:

```
/your/workspace/
├── gsim/
│   ├── Cargo.toml
│   └── src/
└── gsim_py/
    ├── Cargo.toml
    └── src/
```

## Installation

This project uses [Maturin](https://www.maturin.rs/) to build the Python package from the Rust source code. Maturin is a tool for building and publishing Python packages written in Rust. It compiles the Rust code into a native Python extension module, allowing you to call Rust functions from Python with minimal overhead.

The `pyproject.toml` file in this repository is configured to use Maturin as the build backend. When you run `pip install`, it will automatically use Maturin to compile the Rust code and install the resulting package.

To install `gsim_py`, follow these steps:

1.  Ensure you have a Rust toolchain installed. If not, you can install it from [rust-lang.org](https://www.rust-lang.org/tools/install).

2.  Make sure you have satisfied the dependency on `gsim` as described above.

3.  Navigate to the `gsim_py` project directory:
    ```bash
    cd gsim_py
    ```

4.  Install the package using pip. This command will invoke Maturin to build the Rust extension and install it into your Python environment.
    ```bash
    pip install .
    ```

## Usage

Here is an example of how to use `gsim_py` to run a 4-qubit simulation, based on `benchmark_rs_py.py`:

```python
import gsim_py
import numpy as np
import time

# 4-qubit benchmark based on gsimpy/tests/test_gsim4qubit.py
pauli_i = np.array([[1, 0], [0, 1]], dtype=np.complex128)
pauli_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
pauli_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
pauli_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

# 4-qubit operators
xixx = np.kron(pauli_x, np.kron(pauli_i, np.kron(pauli_x, pauli_x)))
iiyi = np.kron(pauli_i, np.kron(pauli_i, np.kron(pauli_y, pauli_i)))
zzzz = np.kron(pauli_z, np.kron(pauli_z, np.kron(pauli_z, pauli_z)))

# Generators for DLA
hams_for_dla = [xixx, iiyi, zzzz]

# Initial state: rho = (1/16) * ZZZZ
init_density_matrix = zzzz * (1.0 / 16.0)

# Observable
observable = zzzz

# Gate Hamiltonian
gate_ham = xixx + 0.5 * iiyi
gate_hamiltonians = [gate_ham]

# Create and prepare the gsim dictionary
gsim_dict = gsim_py.make_gsim(init_density_matrix, observable, hams_for_dla)
gsim_dict["ad_gates"] = gsim_py.get_ad_rep_gate_hams(gate_hamiltonians, gsim_dict["dla"])

params_and_gate_nums = [(np.pi / 2.0 * 0.2, 0)]

start_time = time.time()
result = gsim_py.simulate(gsim_dict, params_and_gate_nums)
duration = time.time() - start_time

print(f"gsim_py (Rust/Python, 4-qubit) simulation result: {result}")
print(f"Time elapsed in gsim_py (Rust/Python, 4-qubit) is: {duration:.6f} seconds")
```

This script demonstrates how to set up a simulation with initial states, operators, and Hamiltonians, and then execute it using `gsim_py`.
