from __future__ import annotations

from enum import Enum
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import XXPlusYYGate, PhaseGate, CPhaseGate
import numpy as np
import ffsim


class SpinType(Enum):
  Spinless = "Spinless"
  SpinUnbalanced = "SpinUnbalanced"
  SpinBalanced = "SpinBalanced"


class UCJ:
  def __init__(self, num_spatial_orbitals: int, n_reps: int, spin_type: SpinType = SpinType.SpinBalanced):
    if spin_type is not SpinType.SpinBalanced:
      raise NotImplementedError(
          f"{spin_type} is not implemented, only SpinBalanced is implemented for now."
      )
    self._norb = num_spatial_orbitals
    self._n_reps = n_reps
    self._spin_type = spin_type

    self._circuit: QuantumCircuit | None = None
    self._diag_coulomb_mats: np.ndarray | None = None
    self._orbital_rotations: np.ndarray | None = None
    self._final_orbital_rotation: np.ndarray | None = None

    self._orbital_givens_indices_per_layer: list[list[tuple[int, int, float, float, Parameter, Parameter]]] = [
        [] for _ in range(self._n_reps + 1)
    ]
    self._diagonal_phases_indices_per_layer: list[list[tuple[int, float, Parameter]]] = [
        [] for _ in range(self._n_reps + 1)
    ]
    self._jastrow_indices_per_layer: list[list[tuple[int, int, float, Parameter]]] = [
        [] for _ in range(self._n_reps)
    ]
    self._parameters: dict[Parameter, float] = {}

  def from_t_amplitudes(self, t1: np.ndarray, t2: np.ndarray):
    self._diag_coulomb_mats, self._orbital_rotations = ffsim.linalg.double_factorized_t2(
        t2, max_terms=self._n_reps)
    self._final_orbital_rotation = ffsim.variational.util.orbital_rotation_from_t1_amplitudes(
        t1)

  def _initialize_givens_parameters(self):
    for k in range(self._n_reps + 1):
      if k < self._n_reps:
        rotations, diagonal_phases = ffsim.linalg.givens_decomposition(
            self._orbital_rotations[k]
        )
      else:
        rotations, diagonal_phases = ffsim.linalg.givens_decomposition(
            self._final_orbital_rotation
        )
      for p, (c, s, i, j) in enumerate(rotations):
        theta_parameter = Parameter(f"givens_{k}_{p}_theta")
        beta_parameter = Parameter(f"givens_{k}_{p}_beta")

        theta = 2 * np.arccos(c)
        beta = np.angle(s) - np.pi/2

        self._orbital_givens_indices_per_layer[k].append(
            (i, j, theta, beta, theta_parameter, beta_parameter))

      for l, phase in enumerate(diagonal_phases):
        parameter = Parameter(f"phase_{k}_{l}")
        phi = np.angle(phase)
        self._diagonal_phases_indices_per_layer[k].append(
            (l, phi, parameter))

  def _initialize_jastrow_parameters(self):
    for k in range(self._n_reps):
      for i in range(self._norb):
        for j in range(i, self._norb):
          parameter = Parameter(f"jastrow_{k}_{i}_{j}")
          value = float(self._diag_coulomb_mats[k, i, j])
          if i == j:
            value *= 0.5
          self._jastrow_indices_per_layer[k].append(
              (i, j, value, parameter))

  def _build_givens_subcircuit(self, layer_idx: int) -> QuantumCircuit:
    sub = QuantumCircuit(self._norb)
    self._append_givens_network(sub, layer_idx, qubit_offset=0)
    return sub

  def _append_givens_network(self, circuit: QuantumCircuit, layer_idx: int, qubit_offset: int) -> None:
    for i, j, _, _, theta_parameter, beta_parameter in self._orbital_givens_indices_per_layer[layer_idx]:
      circuit.append(XXPlusYYGate(theta_parameter, beta_parameter), [
          i + qubit_offset, j + qubit_offset])
    for l, _, parameter in self._diagonal_phases_indices_per_layer[layer_idx]:
      circuit.append(PhaseGate(parameter), [l + qubit_offset])

  def _append_jastrow(self, circuit: QuantumCircuit, layer_idx: int, qubit_offset: int) -> None:
    for i, j, _, parameter in self._jastrow_indices_per_layer[layer_idx]:
      if i == j:
        circuit.append(PhaseGate(parameter), [i + qubit_offset])
      else:
        circuit.append(CPhaseGate(parameter), [
            i + qubit_offset, j + qubit_offset])

  def _register_layer_values(self, layer_idx: int, include_jastrow: bool) -> None:
    for _, _, theta, beta, theta_parameter, beta_parameter in self._orbital_givens_indices_per_layer[layer_idx]:
      self._parameters[theta_parameter] = theta
      self._parameters[beta_parameter] = beta
    for _, phi, parameter in self._diagonal_phases_indices_per_layer[layer_idx]:
      self._parameters[parameter] = phi
    if include_jastrow:
      for _, _, value, parameter in self._jastrow_indices_per_layer[layer_idx]:
        self._parameters[parameter] = value

  def _build_circuit(self) -> QuantumCircuit:
    self._initialize_givens_parameters()
    self._initialize_jastrow_parameters()
    self._parameters = {}

    circuit = QuantumCircuit(self._norb * 2)
    alpha = list(range(0, self._norb))
    beta = list(range(self._norb, 2 * self._norb))

    for k in range(self._n_reps):
      forward = self._build_givens_subcircuit(k)
      inverse = forward.inverse()

      circuit.append(forward.to_instruction(), alpha)
      circuit.append(forward.to_instruction(), beta)

      self._append_jastrow(circuit, k, qubit_offset=0)
      self._append_jastrow(circuit, k, qubit_offset=self._norb)

      circuit.append(inverse.to_instruction(), alpha)
      circuit.append(inverse.to_instruction(), beta)

      self._register_layer_values(k, include_jastrow=True)

    final = self._build_givens_subcircuit(self._n_reps)
    circuit.append(final.to_instruction(), alpha)
    circuit.append(final.to_instruction(), beta)
    self._register_layer_values(self._n_reps, include_jastrow=False)

    self._circuit = circuit
    return circuit

  def set_parameters(self, parameters: dict[Parameter, float]):
    self._parameters = parameters.copy()

  def get_parameters(self) -> dict[Parameter, float]:
    return self._parameters.copy()

  def get_circuit(self) -> QuantumCircuit:
    if self._circuit is None:
      self._circuit = self._build_circuit()
    return self._circuit
