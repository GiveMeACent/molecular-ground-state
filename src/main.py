import ffsim
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from molecules.beh2_molecule import BeH2Molecule
from methods.ccsd import CCSD
from ansatze.ucj import UCJ
from algorithms.vqe import VQE


def main():
  backend = AerSimulator()
  pass_manager = generate_preset_pass_manager(
      target=backend.target, optimization_level=3
  )
  estimator = Estimator(backend)

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

  result = vqe.run()

  print("=== SUMMARY ===")
  print(f"E(CASCI): {mol.get_casci_energy()}")
  print(f"E(CCSD):  {ccsd.get_energy()}")
  print(f"E(VQE):   {result.fun}")


if __name__ == "__main__":
  main()
