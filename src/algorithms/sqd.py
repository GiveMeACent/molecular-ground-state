import numpy as np
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import Sampler
from qiskit_addon_sqd.fermion import diagonalize_fermionic_hamiltonian
from qiskit.primitives import BitArray


class SQD:
  def __init__(
      self,
      ansatz: QuantumCircuit,
      sampler: Sampler,
      pass_manager,
      body_integrals: tuple[np.ndarray, np.ndarray],
      core_energy: float,
      num_orbitals: int,
      nelec: tuple[int, int],
      initial_state: QuantumCircuit | None = None,
      shots=500,
      samples_per_batch=100,
      num_batches=3,
      max_iterations=5,
  ):
    self._ansatz = ansatz
    self._sampler = sampler
    self._pass_manager = pass_manager
    self._hcore, self._eri = body_integrals
    self._core_energy = core_energy
    self._num_orbitals = num_orbitals
    self._nelec = nelec
    self._initial_state = initial_state
    self._shots = shots
    self._samples_per_batch = samples_per_batch
    self._num_batches = num_batches
    self._max_iterations = max_iterations

    self._circuit_isa: QuantumCircuit | None = None
    self._result_history: list = []

  def _prepare_circuit(self) -> QuantumCircuit:
    circuit = QuantumCircuit(
        2 * self._num_orbitals, 2 * self._num_orbitals)
    if self._initial_state is not None:
      circuit.compose(self._initial_state, inplace=True)
    circuit.compose(self._ansatz, inplace=True)
    circuit.measure_all()

    self._circuit_isa = self._pass_manager.run(circuit)
    return self._circuit_isa

  def _sample(self) -> BitArray:
    job = self._sampler.run([self._circuit_isa], shots=self._shots)
    result = job.result()
    return result[0].data.meas

  def run(self) -> dict:
    self._prepare_circuit()
    bit_array = self._sample()

    def callback(results):
      self._result_history.append(results)

    result = diagonalize_fermionic_hamiltonian(
        self._hcore, self._eri, bit_array,
        samples_per_batch=self._samples_per_batch,
        norb=self._num_orbitals, nelec=self._nelec,
        num_batches=self._num_batches,
        max_iterations=self._max_iterations,
        symmetrize_spin=True, callback=callback,
    )

    return {
        "energy": result.energy + self._core_energy,
        "raw_result": result,
        "history": self._result_history,
    }
