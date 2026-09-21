from __future__ import annotations

from enum import Enum
from qiskit import QuantumCircuit
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
          f"{spin_type} only SpinBalanced is implemented for now."
      )
    self._norb = num_spatial_orbitals
    self._n_reps = n_reps
    self._spin_type = spin_type

    self._circuit: QuantumCircuit | None = None
    self._diag_coulomb_mats: np.ndarray | None = None
    self._orbital_rotations: np.ndarray | None = None
    self._final_orbital_rotation: np.ndarray | None = None

  def from_t_amplitudes(self, t1: np.ndarray, t2: np.ndarray):
    self._diag_coulomb_mats, self._orbital_rotations = ffsim.linalg.double_factorized_t2(
        t2, max_terms=self._n_reps)
    self._final_orbital_rotation = ffsim.variational.util.      orbital_rotation_from_t1_amplitudes(
        t1)

  def _build_givens_layers(self):
    for k in range(self._n_reps + 1):
      if k < self._n_reps:
        rotations, diagonal_phases = ffsim.linalg.givens_decomposition(
            self._orbital_rotations[k])
      elif k == self._n_reps:
        rotations, diagonal_phases = ffsim.linalg.givens_decomposition(
            self._final_orbital_rotation)
      for c, s, i, j in rotations:
        None
      for orb_idx, d in enumerate(diagonal_phases):
        None

  def _build_jastrow_layers(self):
    for k in range(self._n_reps):
      for i in range(self._norb):
        for j in range(i, self._norb):
          value = float(self._diag_coulomb_mats[k, i, j])
