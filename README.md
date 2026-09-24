# Molecular Ground State

A quantum-computing project for estimating the ground-state energy of the **beryllium hydride (BeH₂)** molecule using different quantum chemistry and quantum algorithms.

The project combines classical quantum-chemistry methods with quantum-circuit-based approaches, providing a common workflow to compare reference energies with variational and sampling-based methods.

## Overview

The current implementation uses **BeH₂** as the molecular system and evaluates its ground-state energy using:

* **CASCI** — Complete Active Space Configuration Interaction
* **CCSD** — Coupled Cluster Singles and Doubles
* **VQE** — Variational Quantum Eigensolver
* **SQD** — Sample-based Quantum Diagonalization
* **SKQD** — Sampling-based Krylov Quantum Diagonalization

The VQE and sampling-based methods operate on a quantum circuit initialized from a Hartree–Fock state and constructed using a **Unitary Cluster Jastrow (UCJ)** ansatz.

The main execution pipeline is implemented in [`src/main.py`](src/main.py).

## Architecture

The project is organized into the following main components:

```text
molecular-ground-state/
├── src/
│   ├── algorithms/
│   │   ├── vqe.py
│   │   ├── sqd.py
│   │   └── skqd.py
│   │
│   ├── ansatze/
│   │   └── ucj.py
│   │
│   ├── methods/
│   │   └── ccsd.py
│   │
│   ├── molecules/
│   │   └── beh2_molecule.py
│   │
│   └── main.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

### Main components

#### Molecule

`BeH2Molecule` encapsulates the molecular system and exposes the quantities required by the different algorithms, including:

* molecular and active-space information
* number of electrons
* qubit Hamiltonian
* one- and two-body integrals
* core energy
* CASCI reference energy
* SCF data

#### CCSD

The `CCSD` implementation is used to obtain coupled-cluster amplitudes and an energy reference.

The resulting `T1` and `T2` amplitudes are subsequently used to initialize the UCJ ansatz.

#### UCJ ansatz

The `UCJ` ansatz is constructed from the molecular active-space size and the number of repetitions.

The CCSD amplitudes are used to initialize the ansatz parameters:

```python
ansatz = UCJ(num_orbitals, n_reps=3)
ansatz.from_t_amplitudes(t1, t2)
```

#### VQE

The Variational Quantum Eigensolver optimizes the parameters of the quantum circuit with respect to the molecular qubit Hamiltonian.

#### SQD

Sample-based Quantum Diagonalization uses samples obtained from the quantum circuit to estimate the molecular ground-state energy.

#### SKQD

Sampling-based Krylov Quantum Diagonalization extends the sampling approach by constructing a Krylov subspace from the molecular Hamiltonian and sampled quantum states.

## Requirements

The project currently depends on:

* Python
* Qiskit
* Qiskit Aer
* Qiskit Addon SQD
* Qiskit IBM Runtime
* NumPy
* PySCF
* SciPy
* ffsim
* Matplotlib

The complete dependency list is available in [`requirements.txt`](requirements.txt).

## Installation

Clone the repository:

```bash
git clone https://github.com/GiveMeACent/molecular-ground-state.git
cd molecular-ground-state
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```powershell
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Running the project

From the repository root, run:

```bash
python src/main.py
```

The current implementation uses Qiskit Aer as the simulation backend.

The workflow:

1. Initializes an `AerSimulator`.
2. Creates the Qiskit transpilation pass manager.
3. Builds the BeH₂ molecular system.
4. Computes the CCSD reference and obtains its amplitudes.
5. Builds and initializes a UCJ ansatz.
6. Prepares a Hartree–Fock initial state using the Jordan–Wigner representation.
7. Runs VQE.
8. Runs SQD.
9. Runs SKQD.
10. Prints the resulting energies for comparison.

## Output

At the end of the execution, the program prints a summary similar to:

```text
=== SUMMARY ===
E(CASCI): ...
E(CCSD):  ...
E(VQE):   ...
E(SQD):   ...
E(SKQD):  ...
```

The numerical values depend on the molecular setup, simulation configuration and algorithmic parameters.

## Computational workflow

The overall workflow can be summarized as:

```text
                    ┌───────────────┐
                    │     BeH₂      │
                    │   molecule    │
                    └───────┬───────┘
                            │
                 ┌──────────┴──────────┐
                 │                     │
              CASCI                  CCSD
                 │                     │
          reference energy       T1 / T2 amplitudes
                 │                     │
                 │              ┌──────▼──────┐
                 │              │     UCJ     │
                 │              │   ansatz    │
                 │              └──────┬──────┘
                 │                     │
                 │              Hartree–Fock
                 │                initial state
                 │                     │
                 │          ┌──────────┼──────────┐
                 │          │          │          │
                 │         VQE        SQD       SKQD
                 │          │          │          │
                 └──────────┴──────────┴──────────┘
                            │
                     Energy comparison
```

## Methods

### CASCI

CASCI is used as a classical quantum-chemistry reference for the active-space problem.

### CCSD

CCSD provides both an energy estimate and the cluster amplitudes used to initialize the UCJ circuit.

### VQE

VQE estimates the ground-state energy by minimizing the expectation value

```text
E(θ) = ⟨ψ(θ)|H|ψ(θ)⟩
```

where `H` is the molecular qubit Hamiltonian and `θ` represents the parameters of the UCJ circuit.

### SQD

SQD uses samples generated by a quantum circuit to reconstruct and solve a reduced representation of the molecular problem.

### SKQD

SKQD builds upon the sampling approach by exploiting a Krylov subspace generated from the Hamiltonian, providing an alternative route to estimating the low-energy spectrum.

## Quantum simulation

The current entry point uses:

```python
backend = AerSimulator()
```

so the project can be executed locally without requiring access to a physical quantum processor.

The Qiskit transpilation pipeline is configured with optimization level `3`.

## Configuration

The main algorithmic parameters are currently defined directly in the source code.

For example, the UCJ ansatz uses three repetitions:

```python
ansatz = UCJ(num_orbitals, n_reps=3)
```

The SQD and SKQD routines also define their sampling configuration in `src/main.py`.

These parameters can be modified to experiment with different circuit depths and sampling strategies.

## Goals

This project is intended as an experimental framework for studying molecular ground-state estimation with quantum algorithms.

In particular, it provides a common environment for comparing:

* classical quantum-chemistry reference methods;
* variational quantum algorithms;
* sampling-based quantum algorithms;
* Krylov-subspace-based approaches.

## References

The implementation relies on the following software ecosystems:

* [Qiskit](https://qiskit.org/)
* [Qiskit Aer](https://github.com/Qiskit/qiskit-aer)
* [Qiskit Addon SQD](https://github.com/Qiskit/qiskit-addon-sqd)
* [Qiskit IBM Runtime](https://github.com/Qiskit/qiskit-ibm-runtime)
* [PySCF](https://pyscf.org/)
* [ffsim](https://github.com/qiskit-community/ffsim)

## License

No license is currently specified in the repository.

If this project is intended for public reuse, consider adding an explicit open-source license.

## Author

**GiveMeACent**

Repository:
https://github.com/GiveMeACent/molecular-ground-state
