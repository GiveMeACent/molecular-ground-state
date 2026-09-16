from __future__ import annotations

import numpy as np
import pyscf


class CCSD:
  def __init__(
      self,
      scf: (
          pyscf.scf.rohf.HF1e
          | pyscf.scf.hf_symm.HF1e
          | pyscf.scf.hf.RHF
          | pyscf.scf.rohf.ROHF
          | pyscf.scf.hf_symm.RHF
          | pyscf.scf.hf_symm.ROHF
      ),
      mol: pyscf.gto.Mole,
      active_space: range,
  ) -> None:

    self._t1: np.ndarray | None = None
    self._t2: np.ndarray | None = None
    self._energy: float | None = None

    frozen = [
        i
        for i in range(mol.nao_nr())
        if i not in active_space
    ]

    self._ccsd = pyscf.cc.CCSD(
        scf,
        frozen=frozen,
    )

  def run(self) -> None:
    self._ccsd.run()

    self._t1 = self._ccsd.t1
    self._t2 = self._ccsd.t2
    self._energy = self._ccsd.e_tot

  def get_amplitudes(self) -> tuple[np.ndarray, np.ndarray]:
    if self._t1 is None or self._t2 is None:
      self.run()

    return self._t1, self._t2

  def get_energy(self) -> float:
    if self._energy is None:
      self.run()

    return self._energy
