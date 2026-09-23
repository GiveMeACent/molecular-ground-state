from __future__ import annotations

from enum import Enum
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import XXPlusYYGate, PhaseGate, CPhaseGate
import numpy as np
import ffsim
import cmath
import math


class SpinType(Enum):
  Spinless = "Spinless"
  SpinUnbalanced = "SpinUnbalanced"
  SpinBalanced = "SpinBalanced"


class UCJ:
  def __init__(
      self,
      num_spatial_orbitals: int,
      n_reps: int,
      spin_type: SpinType = SpinType.SpinBalanced,
  ):
    if spin_type is not SpinType.SpinBalanced:
      raise NotImplementedError(
          f"{spin_type} is not implemented, only SpinBalanced is supported."
      )
    self._norb = num_spatial_orbitals
    self._n_reps = n_reps
    self._spin_type = spin_type

    self._circuit: QuantumCircuit | None = None
    self._diag_coulomb_mats: np.ndarray | None = None
    self._orbital_rotations: np.ndarray | None = None
    self._final_orbital_rotation: np.ndarray | None = None

    self._orbital_givens_per_layer: list[list[tuple]] = [
        [] for _ in range(self._n_reps + 1)
    ]
    self._diagonal_phases_per_layer: list[list[tuple]] = [
        [] for _ in range(self._n_reps + 1)
    ]
    self._jastrow_aa_per_layer: list[list[tuple]] = [
        [] for _ in range(self._n_reps)]
    self._jastrow_ab_per_layer: list[list[tuple]] = [
        [] for _ in range(self._n_reps)]

    self._parameters: dict[Parameter, float] = {}

  def from_t_amplitudes(self, t1: np.ndarray, t2: np.ndarray):
    ucj_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
        t2, t1=t1, n_reps=self._n_reps
    )
    self._diag_coulomb_mats = ucj_op.diag_coulomb_mats
    self._orbital_rotations = ucj_op.orbital_rotations
    self._final_orbital_rotation = ucj_op.final_orbital_rotation

  def _initialize_givens_parameters(self):
    for k in range(self._n_reps + 1):
      if k < self._n_reps:
        mat = self._orbital_rotations[k]
      else:
        mat = (
            self._final_orbital_rotation
            if self._final_orbital_rotation is not None
            else np.eye(self._norb, dtype=complex)
        )

      rotations, diagonal_phases = ffsim.linalg.givens_decomposition(mat)

      for p, (c, s, i, j) in enumerate(rotations):
        theta_param = Parameter(f"givens_{k}_{p}_theta")
        beta_param = Parameter(f"givens_{k}_{p}_beta")

        r, phi = cmath.polar(s)
        theta = 2 * math.atan2(r, c)
        beta = phi - 0.5 * math.pi

        self._orbital_givens_per_layer[k].append(
            (i, j, theta, beta, theta_param, beta_param)
        )

      for l, phase in enumerate(diagonal_phases):
        param = Parameter(f"phase_{k}_{l}")
        phi = cmath.phase(phase)
        self._diagonal_phases_per_layer[k].append((l, phi, param))

  def _initialize_jastrow_parameters(self):
    for k in range(self._n_reps):
      J_aa = self._diag_coulomb_mats[k, 0]
      J_ab = self._diag_coulomb_mats[k, 1]

      for i in range(self._norb):
        for j in range(i, self._norb):
          param = Parameter(f"jastrow_aa_{k}_{i}_{j}")
          value = float(J_aa[i, j])
          self._jastrow_aa_per_layer[k].append((i, j, value, param))

      for i in range(self._norb):
        for j in range(self._norb):
          param = Parameter(f"jastrow_ab_{k}_{i}_{j}")
          value = float(J_ab[i, j])
          self._jastrow_ab_per_layer[k].append((i, j, value, param))

  def _append_givens_network(
      self, circuit: QuantumCircuit, layer_idx: int, qubit_offset: int
  ) -> None:
    for i, j, _, _, theta_p, beta_p in self._orbital_givens_per_layer[layer_idx]:
      circuit.append(
          XXPlusYYGate(theta_p, beta_p),
          [i + qubit_offset, j + qubit_offset],
      )
    for l, _, param in self._diagonal_phases_per_layer[layer_idx]:
      circuit.append(PhaseGate(param), [l + qubit_offset])

  def _append_jastrow(
      self, circuit: QuantumCircuit, layer_idx: int, qubit_offset: int
  ) -> None:
    for i, j, _, param in self._jastrow_aa_per_layer[layer_idx]:
      if i == j:
        circuit.append(PhaseGate(0.5 * param), [i + qubit_offset])
      else:
        circuit.append(
            CPhaseGate(param), [i + qubit_offset, j + qubit_offset]
        )

  def _append_jastrow_cross(
      self,
      circuit: QuantumCircuit,
      layer_idx: int,
      alpha_offset: int,
      beta_offset: int,
  ) -> None:
    for i, j, _, param in self._jastrow_ab_per_layer[layer_idx]:
      circuit.append(
          CPhaseGate(param),
          [i + alpha_offset, j + beta_offset],
      )

  def _register_layer_values(self, layer_idx: int, include_jastrow: bool) -> None:
    for _, _, theta, beta, theta_p, beta_p in self._orbital_givens_per_layer[layer_idx]:
      self._parameters[theta_p] = theta
      self._parameters[beta_p] = beta
    for _, phi, param in self._diagonal_phases_per_layer[layer_idx]:
      self._parameters[param] = phi

    if include_jastrow:
      for _, _, value, param in self._jastrow_aa_per_layer[layer_idx]:
        self._parameters[param] = value
      for _, _, value, param in self._jastrow_ab_per_layer[layer_idx]:
        self._parameters[param] = value

  def _build_circuit(self) -> QuantumCircuit:
    if self._diag_coulomb_mats is None or self._orbital_rotations is None:
      raise RuntimeError(
          "Call from_t_amplitudes() before building the circuit.")

    self._initialize_givens_parameters()
    self._initialize_jastrow_parameters()
    self._parameters = {}

    circuit = QuantumCircuit(self._norb * 2)
    alpha = list(range(0, self._norb))
    beta = list(range(self._norb, 2 * self._norb))

    for k in range(self._n_reps):
      forward = QuantumCircuit(self._norb)
      self._append_givens_network(forward, k, qubit_offset=0)
      inverse = forward.inverse()

      circuit.append(inverse.to_instruction(), alpha)
      circuit.append(inverse.to_instruction(), beta)

      self._append_jastrow(circuit, k, qubit_offset=0)
      self._append_jastrow(circuit, k, qubit_offset=self._norb)
      self._append_jastrow_cross(
          circuit, k, alpha_offset=0, beta_offset=self._norb
      )

      circuit.append(forward.to_instruction(), alpha)
      circuit.append(forward.to_instruction(), beta)

      self._register_layer_values(k, include_jastrow=True)

    final = QuantumCircuit(self._norb)
    self._append_givens_network(final, self._n_reps, qubit_offset=0)
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
