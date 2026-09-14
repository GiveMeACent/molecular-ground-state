from __future__ import annotations

import numpy as np
import ffsim


class UCCSDAnsatz:
  def __init__(self, num_spatial_orbitals: int, num_occ: int) -> None:
    self._num_spatial_orbitals = num_spatial_orbitals
    self._num_occ = num_occ
    self._num_virt = num_spatial_orbitals - num_occ

    self._t1: np.ndarray | None = None
    self._t2: np.ndarray | None = None
    self._operator: ffsim.UCCSDOpRestricted | None = None

  def initialize_zero(self) -> None:
    self._t1 = np.zeros((self._num_occ, self._num_virt))
    self._t2 = np.zeros((self._num_occ, self._num_occ,
                        self._num_virt, self._num_virt))

  def set_parameters(
      self,
      t1: np.ndarray,
      t2: np.ndarray,
  ) -> None:
    self._t1 = t1
    self._t2 = t2

  def build_operator(self):
    self._operator = ffsim.UCCSDOpRestricted(
        self._t1,
        self._t2,
    )

  def get_operator(self) -> ffsim.UCCSDOpRestricted:
    if self._operator is None:
      raise RuntimeError(
          "UCCSD operator has not been initialized."
      )

    return self._operator

  def get_parameters(
      self,
  ) -> tuple[np.ndarray, np.ndarray]:
    if self._t1 is None or self._t2 is None:
      raise RuntimeError(
          "UCCSD parameters have not been initialized."
      )

    return self._t1, self._t2
