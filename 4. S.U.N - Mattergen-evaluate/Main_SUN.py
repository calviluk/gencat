from ase.io import read, write
from ase.db import connect
import os
import subprocess
import json
import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from mattersim.forcefield import MatterSimCalculator

structures = connect('./combined_relax_structures.db')
#print(structures[0].info['total_energy'])
 
dbe = connect("extended_relax_structures.db")

calculator = MatterSimCalculator(pretrained_name="mattersim-v1.0.0-5M")
 
for i,row in enumerate(structures.select()):
        atoms = row.toatoms()
        atoms.calc = calculator
        energy = atoms.get_potential_energy()
        print(energy)
        write('structure.xyz', atoms)
        np.save('energy.npy', energy)
        subprocess.run(['python3', 'ev.py', '--structures_path=structure.xyz', '--energies_path="energy.npy"', '--relax=False', '--structure_matcher=disordered', '--save_as="metrics.json"'])
        with open('metrics.json', 'r') as file:
            data = json.load(file)
        result = {key: value['value'] for key, value in data.items()}
        for key, value in result.items():
            exec(f"{key} = {value}")
        dbe.write(atoms, avg_energy_above_hull_per_atom=avg_energy_above_hull_per_atom, avg_rmsd_from_relaxation=avg_rmsd_from_relaxation, frac_novel_unique_stable_structures=frac_novel_unique_stable_structures, frac_stable_structures=frac_stable_structures, frac_successful_jobs=frac_successful_jobs, avg_comp_validity=avg_comp_validity, avg_structure_comp_validity=avg_structure_comp_validity, avg_structure_validity=avg_structure_validity, frac_novel_structures=frac_novel_structures, frac_novel_systems=frac_novel_systems, frac_novel_unique_structures=frac_novel_unique_structures, frac_unique_structures=frac_unique_structures, frac_unique_systems=frac_unique_systems, precision=precision, recall=recall, c_id=i)
        os.remove('metrics.json')
        os.remove('structure.xyz')
        os.remove('energy.npy')
        del result