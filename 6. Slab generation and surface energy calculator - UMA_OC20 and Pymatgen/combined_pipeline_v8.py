"""
combined_pipeline_v7_HPC.py - Optimized for HPC submission
"""

import os
import sys
import csv
import numpy as np
import logging
from datetime import datetime
from collections import Counter

from ase.io import write
from ase import Atoms
from ase.constraints import FixAtoms
from ase.db import connect

from pymatgen.core import Structure
from pymatgen.core.surface import SlabGenerator

import torch

try:
    from huggingface_hub import login
    from fairchem.core import FAIRChemCalculator, pretrained_mlip
except ImportError as e:
    print(f"Error: FAIRChem not installed: {e}")
    sys.exit(1)

# Setup logging instead of prints
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('surface_energy_calculation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# BULK ENERGY LOADER
# ============================================================================

class BulkEnergyCsvReader:
    """Load pre-computed bulk energies from CSV file."""
    
    def __init__(self, csv_file='bulk_energies_summary.csv'):
        self.csv_file = csv_file
        self.bulk_energies = {}
    
    def load(self):
        """Load bulk energies from CSV."""
        if not os.path.exists(self.csv_file):
            raise FileNotFoundError(f"Bulk energies CSV not found: {self.csv_file}")
        
        try:
            with open(self.csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    element = row['Element'].strip()
                    energy_per_atom = float(row['Bulk_Energy_per_Atom_eV'])
                    self.bulk_energies[element] = energy_per_atom
            
            if not self.bulk_energies:
                raise ValueError(f"No data read from {self.csv_file}")
            
            logger.info(f"Loaded {len(self.bulk_energies)} bulk energies")
            return self.bulk_energies
        
        except Exception as e:
            logger.error(f"Error reading CSV: {e}")
            raise

# ============================================================================
# SURFACE ENERGY CALCULATOR
# ============================================================================

class CombinedSurfaceEnergyCalculator:
    """Surface formation energy calculator with UMA model."""
    
    def __init__(self,
                 input_db_file='S_filtered_structures.db',
                 bulk_csv_file='bulk_energies_summary.csv',
                 output_db='surface_formation_energies_uma.db',
                 output_traj_dir='grouped_slab_trajectories',
                 save_trajectories=True,
                 save_slab_geometries=True,
                 min_slab_size=10.0,
                 min_vacuum=20.0,
                 model_name="uma",  # ← Can change to "equiformer", "s2ef", etc.
                 max_structures=None):
        
        self.input_db_file = input_db_file
        self.bulk_csv_file = bulk_csv_file
        self.output_db = output_db
        self.output_traj_dir = output_traj_dir
        self.save_trajectories = save_trajectories
        self.save_slab_geometries = save_slab_geometries
        self.min_slab_size = min_slab_size
        self.min_vacuum = min_vacuum
        self.model_name = model_name
        self.max_structures = max_structures
        
        self.miller_indices = [(1, 0, 0), (1, 1, 0), (1, 1, 1)]
        self.z_tolerance = 0.5
        
        self.bulk_energies = {}
        self.calculator = None
        self.results = []
        self.trajectory_data = {str(m): [] for m in self.miller_indices}
        
        # Counter for progress tracking
        self.structures_processed = 0
        self.surfaces_computed = 0
    
    def _load_bulk_energies(self):
        """Load bulk energies from CSV."""
        reader = BulkEnergyCsvReader(self.bulk_csv_file)
        self.bulk_energies = reader.load()
    
    def _initialize_calculator(self):
        """Initialize UMA (or other) calculator."""
        logger.info(f"Initializing {self.model_name.upper()} calculator...")
        
        try:
            torch.utils.checkpoint.use_reentrant = False
            
            hf_token = os.environ.get('HF_TOKEN')
            if hf_token:
                login(token=hf_token)
            
            # Load specified model
            predictor = pretrained_mlip.get_predict_unit(f"{self.model_name}-s-1")
            self.calculator = FAIRChemCalculator(predictor, task_name="oc20")
            
            if hasattr(self.calculator, 'model'):
                self.calculator.model.eval()
            
            logger.info(f"✓ Calculator initialized")
        
        except Exception as e:
            logger.error(f"Failed to initialize calculator: {e}")
            raise
    
    def _get_all_structures(self):
        """Read structures from database."""
        if not os.path.exists(self.input_db_file):
            raise FileNotFoundError(f"Input database not found: {self.input_db_file}")
        
        try:
            input_db = connect(self.input_db_file)
            structures = []
            
            for idx, row in enumerate(input_db.select()):
                if self.max_structures is not None and idx >= self.max_structures:
                    break
                
                atoms = row.toatoms()
                atoms.pbc = True
                structures.append(atoms)
            
            total_available = len(list(input_db.select()))
            logger.info(f"Loaded {len(structures)} structures (total: {total_available})")
            
            return structures
        
        except Exception as e:
            logger.error(f"Error reading structures: {e}")
            raise
    
    def _compute_surface_area(self, atoms):
        """Compute surface area."""
        cell = atoms.cell
        a = cell[0]
        b = cell[1]
        area = np.linalg.norm(np.cross(a, b))
        return area
    
    def _compute_slab_energy(self, slab):
        """Compute total energy with error handling."""
        slab.calc = self.calculator
        
        try:
            with torch.no_grad():
                energy = slab.get_potential_energy()
            return energy
        
        except RuntimeError as e:
            error_str = str(e).lower()
            if "checkpoint" in error_str or "tensor" in error_str:
                logger.warning(f"Checkpoint error detected, retrying...")
                try:
                    with torch.no_grad():
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        energy = slab.get_potential_energy()
                    return energy
                except Exception as e2:
                    logger.error(f"Alternative attempt failed: {str(e2)[:50]}")
                    raise
            else:
                raise
    
    def _compute_bulk_reference_energy(self, slab):
        """Compute bulk reference energy."""
        composition = dict(Counter(slab.get_chemical_symbols()))
        total_energy = 0.0
        
        for elem, count in composition.items():
            if elem not in self.bulk_energies:
                raise KeyError(f"Element '{elem}' not in bulk energies")
            total_energy += self.bulk_energies[elem] * count
        
        return total_energy, composition
    
    def _compute_surface_formation_energy(self, slab_energy, bulk_ref_energy, area):
        """Compute surface formation energy."""
        return (slab_energy - bulk_ref_energy) / (2 * area)
    
    def _tag_layers_and_lock_bottom(self, ase_slab):
        """Assign layer tags and lock bottom layers."""
        z_coords = ase_slab.positions[:, 2]
        sorted_z = np.sort(np.unique(z_coords))
        layer_z_positions = []
        current_layer = [sorted_z[0]]
        
        for z in sorted_z[1:]:
            if z - current_layer[-1] <= self.z_tolerance:
                current_layer.append(z)
            else:
                layer_z_positions.append(current_layer)
                current_layer = [z]
        layer_z_positions.append(current_layer)
        
        tags = np.zeros(len(ase_slab), dtype=int)
        for layer_idx, layer_z_list in enumerate(layer_z_positions):
            for atom_idx, z_coord in enumerate(z_coords):
                for layer_z in layer_z_list:
                    if abs(z_coord - layer_z) <= self.z_tolerance:
                        tags[atom_idx] = layer_idx
                        break
        
        ase_slab.set_tags(tags)
        
        fixed_indices = [i for i, tag in enumerate(tags) if tag in [0, 1]]
        
        if fixed_indices:
            constraint = FixAtoms(indices=fixed_indices)
            ase_slab.set_constraint(constraint)
        
        return ase_slab, len(layer_z_positions), len(fixed_indices), fixed_indices
    
    def _generate_surfaces_from_structure(self, structure_idx, bulk_atoms, output_db_conn):
        """Generate surfaces from bulk structure."""
        import json
        
        try:
            pmg_structure = Structure(
                lattice=bulk_atoms.cell,
                species=bulk_atoms.get_chemical_symbols(),
                coords=bulk_atoms.get_positions(),
                coords_are_cartesian=True
            )
        except Exception as e:
            logger.error(f"Structure {structure_idx}: Conversion failed: {e}")
            return []
        
        structure_results = []
        
        for miller in self.miller_indices:
            try:
                slab_gen = SlabGenerator(
                    pmg_structure,
                    miller,
                    min_slab_size=self.min_slab_size,
                    min_vacuum_size=self.min_vacuum,
                    max_normal_search=2,
                    center_slab=True,
                    reorient_lattice=True
                )
                
                slabs = slab_gen.get_slabs(ftol=0.05, tol=0.1)
                
                if not slabs:
                    continue
                
                for term_idx, pmg_slab in enumerate(slabs):
                    try:
                        ase_slab = Atoms(
                            symbols=[str(site.specie) for site in pmg_slab.sites],
                            positions=[site.coords for site in pmg_slab.sites],
                            cell=pmg_slab.lattice.matrix,
                            pbc=True
                        )
                        
                        ase_slab, num_layers, num_fixed, fixed_indices = self._tag_layers_and_lock_bottom(ase_slab)
                        area = self._compute_surface_area(ase_slab)
                        bulk_ref_energy, composition = self._compute_bulk_reference_energy(ase_slab)
                        
                        slab_energy = self._compute_slab_energy(ase_slab)
                        gamma = self._compute_surface_formation_energy(slab_energy, bulk_ref_energy, area)
                        
                        result = {
                            'structure_idx': structure_idx,
                            'miller_index': str(miller),
                            'termination_index': term_idx,
                            'composition': json.dumps(composition),
                            'n_atoms': len(ase_slab),
                            'n_layers': num_layers,
                            'n_fixed_atoms': num_fixed,
                            'slab_area_ang2': float(area),
                            'bulk_reference_energy_ev': float(bulk_ref_energy),
                            'total_slab_energy_ev': float(slab_energy),
                            'surface_formation_energy_ev_per_ang2': float(gamma),
                            'calculation_date': datetime.now().isoformat()
                        }
                        
                        output_db_conn.write(ase_slab, **result)
                        structure_results.append(result)
                        self.surfaces_computed += 1
                        
                        if self.save_trajectories:
                            self.trajectory_data[str(miller)].append(ase_slab)
                        
                    except Exception as e:
                        logger.debug(f"Structure {structure_idx}, Miller {miller}, Term {term_idx}: {str(e)[:50]}")
                        continue
            
            except Exception as e:
                logger.debug(f"Structure {structure_idx}, Miller {miller}: {str(e)[:50]}")
                continue
        
        return structure_results
    
    def _save_trajectories(self):
        """Save trajectory files."""
        if not self.save_trajectories or not self.trajectory_data:
            return
        
        logger.info(f"Saving trajectories...")
        os.makedirs(self.output_traj_dir, exist_ok=True)
        
        for miller_str, slabs in self.trajectory_data.items():
            if not slabs:
                continue
            
            miller_formatted = miller_str.replace('(', '').replace(')', '').replace(' ', '').replace(',', '_')
            output_file = os.path.join(self.output_traj_dir, f'slabs_miller_{miller_formatted}.traj')
            
            try:
                write(output_file, slabs)
                logger.info(f"  {miller_formatted}: {len(slabs)} slabs")
            except Exception as e:
                logger.error(f"  {miller_formatted}: {e}")
    
    def compute_all(self):
        """Main computation pipeline."""
        logger.info("=" * 80)
        logger.info("SURFACE FORMATION ENERGY CALCULATION")
        logger.info(f"Model: {self.model_name.upper()}")
        logger.info("=" * 80)
        
        if not os.path.exists(self.input_db_file):
            raise FileNotFoundError(f"Input database not found: {self.input_db_file}")
        if not os.path.exists(self.bulk_csv_file):
            raise FileNotFoundError(f"Bulk CSV not found: {self.bulk_csv_file}")
        
        if os.path.exists(self.output_db):
            os.remove(self.output_db)
        
        self._load_bulk_energies()
        self._initialize_calculator()
        structures = self._get_all_structures()
        output_db_conn = connect(self.output_db)
        
        logger.info(f"Processing {len(structures)} structures...")
        logger.info(f"Miller indices: {self.miller_indices}")
        
        for struct_idx, bulk_atoms in enumerate(structures):
            try:
                structure_results = self._generate_surfaces_from_structure(
                    struct_idx, bulk_atoms, output_db_conn
                )
                self.results.extend(structure_results)
                self.structures_processed += 1
                
                # Log progress every 5 structures
                if (self.structures_processed % 5) == 0:
                    logger.info(f"Progress: {self.structures_processed}/{len(structures)} structures, {self.surfaces_computed} surfaces computed")
            
            except Exception as e:
                logger.error(f"Structure {struct_idx}: {e}")
                continue
        
        if self.save_trajectories:
            self._save_trajectories()
        
        logger.info("=" * 80)
        logger.info(f"COMPLETE")
        logger.info(f"  Structures processed: {self.structures_processed}")
        logger.info(f"  Total surfaces: {self.surfaces_computed}")
        logger.info(f"  Database: {self.output_db}")
        logger.info("=" * 80)
        
        return self.results

if __name__ == "__main__":
    calc = CombinedSurfaceEnergyCalculator(
        input_db_file='S_filtered_structures.db',
        bulk_csv_file='bulk_energies_summary.csv',
        output_db='surface_formation_energies_uma.db',
        output_traj_dir='grouped_slab_trajectories',
        save_trajectories=True,
        save_slab_geometries=True,
        model_name="uma",  # ← Can change models here
        max_structures=None  # None = process all
    )
    
    calc.compute_all()