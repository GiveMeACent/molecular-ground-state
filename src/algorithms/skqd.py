import numpy as np
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import Sampler


class SKQD:
  def __init__(
      self,
      ansatz: QuantumCircuit,
      sampler: Sampler,
      pass_manager,
      body_integrals: tuple[np.ndarray, np.ndarray],
      core_energy: float,
      nelec: tuple[int, int],
      initial_state: QuantumCircuit | None = None,
  ):
    self._ansatz = ansatz
    self._sampler = sampler
    self._pass_manager = pass_manager
    self._body_integrals = body_integrals
    self._core_energy = core_energy
    self._nelec = nelec
    self._initial_state = initial_state
