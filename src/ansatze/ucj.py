from __future__ import annotations

from enum import Enum
import numpy as np
import ffsim


class SpinType(Enum):
  Spinless = "Spinless"
  SpinUnbalanced = "SpinUnbalanced"
  SpinBalanced = "SpinBalanced"


class UCJ:
  _n_occ: int
  _n_virt: int
  _n_reps: int
  _spin_type: SpinType
  _orbital_params: np.ndarray | None = None
  _jastrow_params: tuple[np.ndarray, ...] | None = None
  _operator: ffsim.UCJOpSpinless | ffsim.UCJOpSpinUnbalanced | ffsim.UCJOpSpinBalanced | None = None

  def __init__(self, n_occ: int, n_virt: int, n_reps: int, spin_type: SpinType):
    self._n_occ = n_occ
    self._n_virt = n_virt
    self._n_reps = n_reps
    self._spin_type = spin_type

  def set_params(self, orbital_parameters: np.ndarray, jastrow_params: tuple[np.ndarray, ...]):
    self._orbital_params = orbital_parameters
    self._jastrow_params = jastrow_params

  def build_operator(self):
    match self._spin_type:
      case SpinType.Spinless:
        self._operator = ffsim.UCJOpSpinless(
            self._jastrow_params, self._orbital_params)
      case SpinType.SpinUnbalanced:
        self._operator = ffsim.UCJOpSpinUnbalanced(
            self._jastrow_params, self._orbital_params)
      case SpinType.SpinBalanced:
        self._operator = ffsim.UCJOpSpinBalanced(
            self._jastrow_params, self._orbital_params)

  def get_params(self):
    return (self._orbital_params, self._jastrow_params)

  def get_operator(self):
    return self._operator
