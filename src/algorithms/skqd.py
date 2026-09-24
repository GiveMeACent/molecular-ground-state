import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import Sampler
from qiskit_addon_sqd.fermion import diagonalize_fermionic_hamiltonian
from qiskit.primitives import BitArray
from qiskit.circuit.library import PauliEvolutionGate


class SKQD:
  def __init__(
      self, qubit_hamiltonian: SparsePauliOp,
      ansatz: QuantumCircuit,
      body_integrals: tuple[np.ndarray, np.ndarray],
      core_energy: float,
      num_orbitals: int, nelec: tuple[int, int],
      time_step: float, krylov_dim: int,
      sampler: Sampler, pass_manager,
      initial_state: QuantumCircuit | None = None,
      shots=500, samples_per_batch=100, num_batches=3, max_iterations=5,
  ):
    self._qubit_hamiltonian = qubit_hamiltonian
    self._ansatz = ansatz
    self._hcore, self._eri = body_integrals
    self._core_energy = core_energy
    self._num_orbitals = num_orbitals
    self._nelec = nelec
    self._time_step = time_step
    self._krylov_dim = krylov_dim
    self._sampler = sampler
    self._pass_manager = pass_manager
    self._initial_state = initial_state
    self._shots = shots
    self._samples_per_batch = samples_per_batch
    self._num_batches = num_batches
    self._max_iterations = max_iterations

    self._result_history = []

  def _build_krylov_circuits(self) -> list[QuantumCircuit]:
    krylov_circuits: list[QuantumCircuit] = []
    time_evolution_gate = PauliEvolutionGate(
        self._qubit_hamiltonian, time=self._time_step)

    for _ in range(self._krylov_dim):
      krylov_circuit = QuantumCircuit(
          2 * self._num_orbitals, 2 * self._num_orbitals)
      if self._initial_state is not None:
        krylov_circuit.compose(self._initial_state, inplace=True)
      krylov_circuit.compose(self._ansatz, inplace=True)
      for _ in range(self._krylov_dim):
        krylov_circuit.compose(time_evolution_gate, inplace=True)
      krylov_circuit.measure_all()
      krylov_circuits.append(self._pass_manager.run(krylov_circuit))

    return krylov_circuits

  def _sample_all(self, circuits: list[QuantumCircuit]) -> BitArray:
    job = self._sampler.run(circuits, shots=self._shots)
    result = job.result()
    bit_arrays = [pub_result.data.meas for pub_result in result]
    return BitArray.concatenate_shots(bit_arrays)

  def run(self) -> dict:
    circuits = self._build_krylov_circuits()
    bit_array = self._sample_all(circuits)

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
