from __future__ import annotations

import numpy as np
import ffsim


class UCCSD:
  _n_occ: int
  _n_virt: int
  _num_elec: tuple[int, int]

  _t1: np.ndarray | None = None
  _t2: np.ndarray | None = None
  _operator: ffsim.UCCSDOpRestricted | None = None

  def __init__(self, n_occ: int, n_virt: int, num_elec: int):
    self._n_occ = n_occ
    self._n_virt = n_virt
    self._num_elec = num_elec

  def initialize_t1_t2(self):
    self._t1 = np.zeros(
        (self._num_occ, self._num_virt)
    )
    self._t2 = np.zeros(
        (self._num_occ, self._num_occ,
         self._num_virt, self._num_virt)
    )

  def build_operator(self):
    self._operator = ffsim.UCCSDOpRestricted(self._t1, self._t2)

  def set_t1_t2(self, t1: np.ndarray, t2: np.ndarray):
    self._t1 = t1
    self._t2 = t2

  def get_spatial_orbitals(self):
    return (self._n_occ, self._n_virt)

  def get_num_elec(self):
    return self._num_elec

  def get_t1_t2(self):
    return (self._t1, self._t2)

  def get_operator(self):
    return self._operator
