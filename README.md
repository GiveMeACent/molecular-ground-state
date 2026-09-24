# molecular-ground-state

Ground-state energy estimation of **BeH₂** using three quantum-centric diagonalization techniques (**VQE**, **SQD**, and **SKQD**) compared against classical references (**RHF**, **CASCI**, and **CCSD**).

## Why

This is a personal project to understand the physics, mathematics, and implementation details behind VQE, SQD, and SKQD.

It is **not** intended to replace existing libraries such as `ffsim` or `qiskit-addon-sqd`. The goal was instead to implement a meaningful part of the stack from scratch and verify it against trusted implementations.

In particular, the **UCJ ansatz is implemented from scratch as a parameterized Qiskit circuit** and verified against `ffsim.UCJOpSpinBalancedJW`, reaching a state fidelity of **1**.

## Problem setup

The system is:

* **BeH₂**
* **STO-3G**
* 1 frozen core orbital
* active space: **6 orbitals / 4 electrons**

The quantum methods are:

* **VQE**: variational optimization of a parameterized UCJ circuit;
* **SQD**: sampling from a fixed circuit followed by classical diagonalization in the sampled subspace;
* **SKQD**: SQD extended with a Krylov sequence generated through Trotterized time evolution.

The classical references are RHF, CASCI, and CCSD.

## Structure

```text
molecular-ground-state/
├── README.md
├── requirements.txt
└── src/
    ├── main.py
    ├── algorithms/
    │   ├── vqe.py
    │   ├── sqd.py
    │   └── skqd.py
    ├── ansatze/
    │   └── ucj.py
    ├── methods/
    │   └── ccsd.py
    └── molecules/
        └── beh2_molecule.py
```

`BeH2Molecule` owns the molecular data and exposes both the fermionic integrals and the Jordan-Wigner qubit Hamiltonian.

## UCJ

The UCJ ansatz is built as a parameterized Qiskit circuit using:

* Givens rotations;
* a diagonal Jastrow term;
* the inverse Givens network;
* multiple configurable layers.

CCSD `t1` and `t2` amplitudes provide the initial parameters.

The implementation was cross-checked against `ffsim` and reached:

```text
state fidelity = 1
```

## Results

**BeH₂ — STO-3G — 6 orbitals / 4 electrons**

| Method | Energy (Ha) | ΔE from CCSD (Ha) |
| ------ | ----------: | ----------------: |
| RHF    |  -15.561278 |                 — |
| CASCI  |  -15.594710 |         reference |
| CCSD   |  -15.594350 |                 — |
| VQE    |  -15.588573 |            0.0058 |
| SQD    |  -15.589308 |            0.0050 |
| SKQD   |  -15.591839 |            0.0025 |

Configuration:

* UCJ: `n_reps = 3`
* VQE: COBYLA, 10 iterations
* SQD/SKQD: classical fermionic subspace diagonalization

CASCI is exact **within the selected active space**.

## Running

```bash
git clone https://github.com/GiveMeACent/molecular-ground-state.git
cd molecular-ground-state

pip install -r requirements.txt
python src/main.py
```

## Main dependencies

* `pyscf`: molecular integrals, RHF, CASCI, CCSD
* `ffsim`: fermionic operators, decompositions, Jordan-Wigner mapping
* `qiskit` / `qiskit-aer`: quantum circuits and simulation
* `qiskit-addon-sqd`: SQD/SKQD post-processing and diagonalization

## References

* IBM Quantum — [Sample-based quantum diagonalization of a chemistry Hamiltonian](https://quantum.cloud.ibm.com/docs/en/tutorials/sample-based-quantum-diagonalization)
* IBM Quantum — [Sample-based Krylov quantum diagonalization of a fermionic lattice model](https://quantum.cloud.ibm.com/docs/en/tutorials/sample-based-krylov-quantum-diagonalization)
* Motta et al. — [Quantum-Centric Algorithm for Sample-Based Krylov Diagonalization](https://arxiv.org/abs/2501.09702)
* [ffsim](https://qiskit-community.github.io/ffsim/)
