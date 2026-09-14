import numpy as np
import pyscf


class BeH2Molecule:
  _mol: pyscf.gto.Mole
  _n_frozen: int
  _active_space: range
  _num_elec_a: int
  _num_elec_b: int

  _hcore: np.ndarray | None = None
  _eri: np.ndarray | None = None

  _nuclear_repulsion_energy: float | None = None
  _cas: object | None = None
  _casci_energy: float | None = None
  _scf: pyscf.scf.rohf.HF1e | pyscf.scf.hf_symm.HF1e | pyscf.scf.hf.RHF | pyscf.scf.rohf.ROHF | pyscf.scf.hf_symm.RHF | pyscf.scf.hf_symm.ROHF = None

  def __init__(self):
    self._mol = pyscf.gto.Mole()

    self._mol.build(
        atom=[
            ["H", (0, 0, 0)],
            ["Be", (1.30, 0, 0)],
            ["H", (2.60, 0, 0)],
        ],
        basis="sto-3g",
        symmetry="Dooh",
    )

    self._n_frozen = 1
    self._active_space = range(
        self._n_frozen,
        self._mol.nao_nr(),
    )

    n_electrons = int(
        sum(self._scf.mo_occ[self._active_space])
    )
    self._num_elec_a = (n_electrons + self._mol.spin) // 2
    self._num_elec_b = (n_electrons - self._mol.spin) // 2

  def get_body_integrals(self):
    if self._hcore is None or self._eri is None:
      self.compute_body_integrals()

    return self._hcore, self._eri

  def get_nuclear_repulsion_energy(self):
    if self._nuclear_repulsion_energy is None:
      self.compute_body_integrals()
    return self._nuclear_repulsion_energy

  def get_active_space(self):
    return self._active_space

  def get_casci_energy(self):
    if self._casci_energy is None:
      self.compute_casci_energy()

    return self._casci_energy

  def get_scf(self):
    return self._scf

  def get_molecule(self):
    return self._mol

  def compute_casci(self):
    self._scf = pyscf.scf.RHF(self._mol).run()
    num_orbitals = len(self._active_space)
    self._cas = pyscf.mcscf.CASCI(
        self._scf,
        num_orbitals,
        (self._num_elec_a, self._num_elec_b),
    )

  def compute_casci_energy(self):
    if self._cas is None:
      self.compute_casci()

    self._casci_energy = self._cas.run().e_tot

  def compute_body_integrals(self):
    if self._cas is None:
      self.compute_casci()

    mo = self._cas.sort_mo(self._active_space, base=0)

    (self._hcore, self._nuclear_repulsion_energy) = self._cas.get_h1cas(mo)

    num_orbitals = len(self._active_space)

    self._eri = pyscf.ao2mo.restore(1, self._cas.get_h2cas(mo), num_orbitals)

  def get_electrons_number(self):
    return (self._num_elec_a, self._num_elec_b)
