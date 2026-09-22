from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
import ffsim

from molecules.beh2_molecule import BeH2Molecule
from methods.ccsd import CCSD
from ansatze.ucj import UCJ
from algorithms.vqe import VQE


def main():
  backend = AerSimulator()
  target = backend.target
  pm = generate_preset_pass_manager(target=target, optimization_level=3)
  estimator = Estimator(backend)

  mol = BeH2Molecule()

  print("=== MOLECULE ===")
  print("NAO:", mol.get_molecule().nao_nr())
  print("Active space:", list(mol.get_active_space()))
  print("Active orbitals:", len(mol.get_active_space()))
  print("Active electrons:", mol.get_electrons_number())

  print("\n=== RHF ===")
  print("E(RHF):", mol.get_scf().e_tot)

  print("\n=== CASCI ===")
  print("E(CASCI):", mol.get_casci_energy())

  cc = CCSD(
      mol.get_scf(),
      mol.get_molecule(),
      mol.get_active_space(),
  )
  cc.run()

  print("\n=== CCSD ===")
  print("E(CCSD):", cc.get_energy())   # se il tuo wrapper lo espone

  uc = UCJ(len(mol.get_active_space()), 4)

  uc.from_t_amplitudes(
      cc.get_amplitudes()[0],
      cc.get_amplitudes()[1],
  )

  qc = uc.get_circuit()
  params = uc.get_parameters()

  vq = VQE(
      mol.get_qubit_hamiltonian(),
      qc,
      params,
      estimator,
      pm,
      ffsim.qiskit.PrepareHartreeFockJW(
          len(mol.get_active_space()), mol.get_electrons_number())
  )

  print("\n=== VQE ===")
  print("E(VQE):", vq.run())


if __name__ == "__main__":
  main()
