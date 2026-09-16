from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp


class VQE:
  def __init__(
      self,
      hamiltonian: SparsePauliOp,
      ansatz: QuantumCircuit,
      estimator,
      pass_manager=None,
      initial_state: QuantumCircuit | None = None,
      optimizer: str = "cobyla",
      max_iterations: int = 100,
      tolerance: float = 1e-6,
  ):

    self._hamiltonian = hamiltonian
    self._ansatz = ansatz
    self._estimator = estimator
    self._pass_manager = pass_manager
    self._initial_state = initial_state
    self._optimizer = optimizer
    self._max_iterations = max_iterations
    self._tolerance = tolerance

    self._cost_history: dict[
        str, int | float | np.ndarray
    ] = {}

  def prepare_circuit(self):
    return

  def evaluate_energy(self):
    return

  def optimize(self):
    return

  def run(self):
    return
