from qiskit.transpiler import generate_preset_pass_manager
import ffsim
from qiskit.primitives import BackendEstimatorV2 as Estimator
from qiskit.primitives import BackendSamplerV2 as Sampler
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
import random
import numpy as np

from molecules.beh2_molecule import BeH2Molecule
from methods.ccsd import CCSD
from ansatze.ucj import UCJ
from algorithms.vqe import VQE
from algorithms.sqd import SQD
from algorithms.skqd import SKQD


def main():
  random.seed(42)
  np.random.seed(42)

  backend = AerSimulator(seed_simulator=42)
  estimator = Estimator(
      backend=backend,
      options={
          "default_precision": 1 / (100000 ** 0.5),
          "seed_simulator": 42
      }
  )
  sampler = Sampler(
      backend=backend,
      options={
          "seed_simulator": 42
      }
  )
  pass_manager = generate_preset_pass_manager(
      backend=backend, seed_transpiler=42, optimization_level=1)

  mol = BeH2Molecule()
  num_orbitals = len(mol.get_active_space())
  n_electrons = mol.get_electrons_number()

  ccsd = CCSD(mol.get_scf(), mol.get_molecule(), mol.get_active_space())
  ccsd.run()
  t1, t2 = ccsd.get_amplitudes()

  ansatz = UCJ(num_orbitals, n_reps=3)
  ansatz.from_t_amplitudes(t1, t2)

  circuit = ansatz.get_circuit()
  parameters = ansatz.get_parameters()

  initial_state = ffsim.qiskit.PrepareHartreeFockJW(
      num_orbitals, n_electrons)

  vqe = VQE(
      mol.get_qubit_hamiltonian(),
      circuit,
      parameters,
      estimator,
      pass_manager,
      initial_state,
  )

  sqd = SQD(circuit.assign_parameters(parameters), sampler, pass_manager, mol.get_body_integrals(),
            mol.get_core_energy(), num_orbitals, n_electrons, initial_state, shots=10_000)

  skqd = SKQD(mol.get_qubit_hamiltonian(), sampler, pass_manager, circuit.assign_parameters(parameters), mol.get_body_integrals(
  ), mol.get_core_energy(), num_orbitals, n_electrons, 1, 2, initial_state, shots=100)

  vqe_result = vqe.run().fun
  sqd_result = sqd.run()["energy"]
  skqd_result = skqd.run()["energy"]

  print("=== SUMMARY ===")
  print(f"E(CASCI): {mol.get_casci_energy()}")
  print(f"E(CCSD):  {ccsd.get_energy()}")
  print(f"E(VQE):   {vqe_result}")
  print(f"E(SQD):   {sqd_result}")
  print(f"E(SKQD):  {skqd_result}")


if __name__ == "__main__":
  main()
