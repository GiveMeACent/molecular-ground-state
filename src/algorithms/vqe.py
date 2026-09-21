from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import minimize
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit.transpiler.preset_passmanagers import StagedPassManager


class VQE:
  def __init__(
      self,
      hamiltonian: SparsePauliOp,
      ansatz: QuantumCircuit,
      parameters: list[np.ndarray],
      estimator: Estimator,
      pass_manager: StagedPassManager,
      initial_state: QuantumCircuit | None = None,
      optimizer: str = "cobyla",
      max_iterations: int = 100,
      tolerance: float = 1e-6,
  ):

    self._hamiltonian = hamiltonian
    self._ansatz = ansatz
    self._parameters = parameters
    self._estimator = estimator
    self._pass_manager = pass_manager
    self._initial_state = initial_state
    self._optimizer = optimizer
    self._max_iterations = max_iterations
    self._tolerance = tolerance

    self._variational_circuit: QuantumCircuit = QuantumCircuit(
        hamiltonian.num_qubits, hamiltonian.num_qubits)
    self._variational_circuit_isa: QuantumCircuit | None = None
    self._cost_history: dict[
        str, int | float | np.ndarray
    ] = {
        "prev_vector": None,
        "iters": 0,
        "cost_history": [],
    }

  def _prepare_variational_circuit(self):
    if self._variational_circuit is not None:
      self._variational_circuit = QuantumCircuit(
          self._hamiltonian.num_qubits, self._hamiltonian.num_qubits)
    if self._initial_state is not None:
      self._variational_circuit.compose(self._initial_state, inplace=True)
    self._variational_circuit.compose(self._ansatz, inplace=True)
    self._variational_circuit.measure_all()

    self._variational_circuit_isa = self._pass_manager.run(
        self._variational_circuit)
    self._hamiltonian_isa = self._hamiltonian.apply_layout(
        layout=self._variational_circuit_isa.layout)

  def _evaluate_energy(self, parameters):
    pub = (self._variational_circuit_isa, [
           self._hamiltonian_isa], [parameters])
    result = self._estimator.run(pubs=[pub]).result()
    energy = result[0].data.evs[0]

    self._cost_history["iters"] += 1
    self._cost_history["prev_vector"] = parameters
    self._cost_history["cost_history"].append(energy)
    print(
        f"Iters. done: {self._cost_history['iters']} [Current cost: {energy}]")

    return energy

  def _optimize(self):
    x0 = np.concatenate([p.ravel() for p in self._parameters])
    result = minimize(
        self._evaluate_energy,
        x0,
        method=self._optimizer,
        tol=self._tolerance,
        options={"maxiter": self._max_iterations},
    )
    return result

  def run(self):
    self._prepare_variational_circuit()
    result = self._optimize()
    return result
