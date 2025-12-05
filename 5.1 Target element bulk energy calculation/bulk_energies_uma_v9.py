"""
UMA-OC20 MLIP Bulk Energy Calculator - All Target Elements (Refactored v9 - WITH HG FIX)

Calculates bulk energy PER ATOM for all transition metals and selected p-block elements.
Uses Materials Project (pymatgen) for Mn, Ga, and In structures.
Saves results to SQLite database and CSV file.

FIXED: Added special handling for Hg (Mercury) which requires explicit lattice constants.

For use in Jupyter notebooks: load and execute cells sequentially.

SETUP INSTRUCTIONS:
===================
Before running this script in your Jupyter notebook, set up API keys by running:

    import os
    from getpass import getpass
    
    print("=" * 70)
    print("UMA-OC20 MLIP - API KEY SETUP")
    print("=" * 70)
    
    print("\n1. Materials Project API Key")
    print("   Get your key at: https://materialsproject.org/api")
    os.environ['MPRESTER_API_KEY'] = getpass("   Enter MPRESTER_API_KEY: ")
    
    print("\n2. HuggingFace API Token")
    print("   Get token at: https://huggingface.co/settings/tokens")
    print("   (Need access to 'fairchem/UMA-OC20' models)")
    os.environ['HF_TOKEN'] = getpass("   Enter HF_TOKEN: ")
    
    print("\n✓ API keys configured for this session!")
    print("=" * 70)

Then import and run this script.
"""

# ============================================================================
# SECTION 1: IMPORTS - All packages imported upfront
# ============================================================================

import os
import sys
import csv
import numpy as np
from datetime import datetime
from getpass import getpass

# ASE (Atomic Simulation Environment)
from ase.build import bulk
from ase.db import connect
from ase.io import write
from ase.atoms import Atoms

# FAIRChem - UMA-OC20 MLIP
try:
    from huggingface_hub import login
    from fairchem.core import FAIRChemCalculator, pretrained_mlip
except ImportError as e:
    print(f"Error: Required packages not installed: {e}")
    print("Please install: pip install huggingface-hub fairchem")
    sys.exit(1)

if os.path.exists('bulk_energies_uma_all_elements.db'):
    os.remove('bulk_energies_uma_all_elements.db')
    print("✓ Database deleted. Run main() again to regenerate.")

# PyMatGen - Materials Project API
try:
    from pymatgen.ext.matproj import MPRester
    HAS_MATPROJ_API = True
    print("✓ pymatgen available - will use Materials Project API for Mn, Ga, In")
except ImportError:
    HAS_MATPROJ_API = False
    print("⚠ Warning: pymatgen not found. Will use fallback structures for Mn, Ga, In.")
    print("   To use Materials Project: pip install pymatgen")


# ============================================================================
# SECTION 1A: INTERACTIVE API KEY LOGIN
# ============================================================================

def login_to_apis(interactive=True):
    """
    Set up API keys for Materials Project and HuggingFace.
    
    Args:
        interactive (bool): If True, prompt user for keys. If False, use env vars.
    """
    
    print("\n" + "=" * 80)
    print("API KEY CONFIGURATION")
    print("=" * 80)
    
    if interactive:
        print("\nPlease provide your API credentials:")
        print("(Leave blank to skip or use existing environment variables)")
        print()
        
        # Materials Project API Key
        print("1. Materials Project API Key")
        print("   Get your key at: https://materialsproject.org/api")
        mp_input = input("   Enter MPRESTER_API_KEY (or press Enter to skip): ").strip()
        if mp_input:
            os.environ['MPRESTER_API_KEY'] = mp_input
            print("   ✓ MPRESTER_API_KEY configured")
        else:
            print("   ⊘ Using existing MPRESTER_API_KEY or falling back to built-in structures")
        
        print()
        
        # HuggingFace API Token
        print("2. HuggingFace API Token")
        print("   Get token at: https://huggingface.co/settings/tokens")
        print("   (You need access to 'fairchem/UMA-OC20' models)")
        hf_input = input("   Enter HF_TOKEN (or press Enter to skip): ").strip()
        if hf_input:
            os.environ['HF_TOKEN'] = hf_input
            print("   ✓ HF_TOKEN configured")
        else:
            print("   ⊘ Using existing HF_TOKEN or attempting anonymous access")
    
    print()
    
    # Check what's set
    mp_key = os.environ.get('MPRESTER_API_KEY')
    hf_key = os.environ.get('HF_TOKEN')
    
    print("Current configuration:")
    print(f"  MPRESTER_API_KEY: {'✓ Set (' + ('*' * 8) + ')' if mp_key else '✗ Not set'}")
    print(f"  HF_TOKEN:         {'✓ Set (' + ('*' * 8) + ')' if hf_key else '✗ Not set'}")
    print("=" * 80)
    
    return mp_key is not None, hf_key is not None


# ============================================================================
# SECTION 2: CONFIGURATION & DATA
# ============================================================================

# Target elements for bulk energy calculation
TARGET_ELEMENTS = [
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Ru", "Rh", "Pd", "Ag", "Cd", "Hf", "Ta",
    "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Al", "Ga", "In"
]

# Known stable crystal structures (for ASE bulk() function)
CRYSTAL_STRUCTURES = {
    'Sc': 'hcp',  'Ti': 'hcp',  'V': 'bcc',  'Cr': 'bcc', 'Mn': 'bcc',
    'Fe': 'bcc',  'Co': 'hcp',  'Ni': 'fcc', 'Cu': 'fcc', 'Zn': 'hcp',
    'Y': 'hcp',   'Zr': 'hcp',  'Nb': 'bcc', 'Mo': 'bcc', 'Ru': 'hcp',
    'Rh': 'fcc',  'Pd': 'fcc',  'Ag': 'fcc', 'Cd': 'hcp', 'Hf': 'hcp',
    'Ta': 'bcc',  'W': 'bcc',   'Re': 'hcp', 'Os': 'hcp', 'Ir': 'fcc',
    'Pt': 'fcc',  'Au': 'fcc',  'Hg': 'hcp', 'Al': 'fcc',
    'Ga': 'orthorhombic',  'In': 'tetragonal'
}

# Materials Project IDs - PRIORITY DATA SOURCE FOR Mn, Ga, In
# Note: These elements have complex structures that should be retrieved from Materials Project
MATPROJ_IDS = {
    'Mn': 'mp-754',   # α-Mn (body-centered cubic-like structure)
    'Ga': 'mp-139',   # Ga orthorhombic (Cmca)
    'In': 'mp-124'    # In tetragonal (I4/mmm)
}

# Database and output file names
DB_NAME = 'bulk_energies_uma_all_elements.db'
CSV_NAME = 'bulk_energies_summary.csv'


# ============================================================================
# SECTION 3: HELPER FUNCTIONS - PyMatGen Conversion & Materials Project Access
# ============================================================================

def pymatgen_to_ase(pmg_structure):
    """Convert pymatgen Structure to ASE Atoms object."""
    try:
        symbols = [site.species.elements[0].symbol for site in pmg_structure]
        positions = pmg_structure.cart_coords
        cell = pmg_structure.lattice.matrix
        atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
        return atoms
    except Exception as e:
        print(f"   Error converting pymatgen structure: {e}")
        return None


def get_structure_from_matproj(element, mp_id):
    """
    Download structure from Materials Project using legacy MPRester API.
    
    Args:
        element (str): Element symbol
        mp_id (str): Materials Project ID (e.g., 'mp-754')
    
    Returns:
        tuple: (ASE Atoms object, structure_type_string) or (None, None) on failure
    """
    try:
        if not HAS_MATPROJ_API:
            return None, None
        
        # Get API key from environment variable
        api_key = os.environ.get('MPRESTER_API_KEY')
        if not api_key:
            print(f"   ⚠ MPRESTER_API_KEY not set. Cannot download {element} from Materials Project.")
            return None, None
        
        # Connect to Materials Project and retrieve structure
        mpr = MPRester(api_key=api_key)
        structure = mpr.get_structure_by_material_id(mp_id)
        
        # Convert to ASE format
        atoms = pymatgen_to_ase(structure)
        if atoms is None:
            return None, None
        
        print(f"   ✓ {element} downloaded from Materials Project (mp-id: {mp_id})")
        print(f"     • N atoms: {len(atoms)}, Cell volume: {atoms.get_volume():.2f} Ų")
        
        return atoms, f"materials_project_{mp_id}"
    
    except Exception as e:
        print(f"   ⚠ Error downloading {element} from Materials Project: {e}")
        return None, None


# ============================================================================
# SECTION 4: FALLBACK STRUCTURES - For when Materials Project unavailable
# ============================================================================

def get_mn_structure_fallback():
    """
    Fallback: α-Mn (body-centered cubic structure) with experimental lattice constant.
    Lattice constant: a = 8.918 Å
    """
    a = 8.918
    atoms = bulk('Mn', crystalstructure='bcc', a=a)
    return atoms


def get_ga_structure_fallback():
    """
    Fallback: Gallium orthorhombic structure (Cmca phase).
    Lattice parameters: a=4.5186, b=7.6594, c=4.5249 Å
    8 atoms per unit cell.
    """
    a, b, c = 4.5186, 7.6594, 4.5249
    positions = [
        [0.0,   0.0,     0.0],
        [0.5,   0.5,     0.0],
        [0.5,   0.0,     0.5],
        [0.0,   0.5,     0.5],
        [0.0,   0.3087,  0.0792],
        [0.5,   0.1913,  0.0792],
        [0.5,   0.3087,  0.5792],
        [0.0,   0.1913,  0.5792]
    ]
    symbols = ['Ga'] * 8
    cell = [[a, 0, 0], [0, b, 0], [0, 0, c]]
    atoms = Atoms(symbols=symbols, scaled_positions=positions, cell=cell, pbc=True)
    return atoms


def get_in_structure_fallback():
    """
    Fallback: Indium tetragonal structure (I4/mmm phase).
    Lattice parameters: a=3.2517, c=4.9465 Å
    2 atoms per unit cell.
    """
    a, c = 3.2517, 4.9465
    positions = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    symbols = ['In'] * 2
    cell = [[a, 0, 0], [0, a, 0], [0, 0, c]]
    atoms = Atoms(symbols=symbols, scaled_positions=positions, cell=cell, pbc=True)
    return atoms


def get_hg_structure_fallback():
    """
    Fallback: Mercury hcp structure with experimental lattice constants.
    Lattice parameters: a = 2.9994 Å, c = 5.6516 Å
    """
    a = 2.9994
    c = 5.6516
    atoms = bulk('Hg', crystalstructure='hcp', a=a, c=c)
    return atoms


# ============================================================================
# SECTION 5: MAIN STRUCTURE BUILDER
# ============================================================================

def build_structure(element):
    """
    Build crystal structure for given element.
    
    Priority:
    1. Materials Project (for Mn, Ga, In) if available
    2. Built-in fallback structures for Mn, Ga, In, Hg (elements requiring explicit lattice constants)
    3. ASE bulk() with known crystal structures for all other elements
    
    Args:
        element (str): Element symbol
    
    Returns:
        tuple: (ASE Atoms object, structure_type_string)
    """
    
    # PRIORITY 1: Materials Project for Mn, Ga, In (complex structures)
    if element in MATPROJ_IDS:
        print(f"\n   → Attempting Materials Project download for {element}...")
        atoms, structure_type = get_structure_from_matproj(element, MATPROJ_IDS[element])
        if atoms is not None:
            return atoms, structure_type
        print(f"   → Falling back to built-in structure for {element}")
    
    # PRIORITY 2: Built-in fallback structures (for elements needing explicit lattice constants)
    if element == 'Mn':
        return get_mn_structure_fallback(), 'mn_bcc_fallback'
    
    elif element == 'Ga':
        return get_ga_structure_fallback(), 'ga_orthorhombic_fallback'
    
    elif element == 'In':
        return get_in_structure_fallback(), 'in_tetragonal_fallback'
    
    elif element == 'Hg':
        return get_hg_structure_fallback(), 'hg_hcp_fallback'
    
    # PRIORITY 3: ASE bulk() with known crystal structures for all elements
    # This includes: Sc, Ti, V, Cr, Fe, Co, Ni, Cu, Zn, Y, Zr, Nb, Mo, Ru, Rh, Pd, Ag, Cd, Hf, Ta, W, Re, Os, Ir, Pt, Au, Al
    structure = CRYSTAL_STRUCTURES.get(element, 'fcc')
    atoms = bulk(element, crystalstructure=structure)
    return atoms, structure


# ============================================================================
# SECTION 6: DATABASE INITIALIZATION & SETUP
# ============================================================================

def initialize_calculator():
    """Initialize UMA-OC20 MLIP calculator and return FAIRChemCalculator object."""
    print("\nInitializing UMA-OC20 MLIP calculator...")
    
    try:
        # Authenticate with HuggingFace (uses HF_TOKEN env var)
        hf_token = os.environ.get('HF_TOKEN')
        if hf_token:
            login(token=hf_token)
        
        # Load pretrained model and create calculator
        predictor = pretrained_mlip.get_predict_unit("uma-s-1")
        calculator = FAIRChemCalculator(predictor, task_name="oc20")
        
        print("✓ Calculator initialized successfully")
        return calculator
    
    except Exception as e:
        print(f"✗ Failed to initialize calculator: {e}")
        print("  Check your HF_TOKEN environment variable and HuggingFace access")
        sys.exit(1)


def setup_database():
    """Connect to SQLite database and return connection object."""
    print(f"\nConnecting to database: {DB_NAME}")
    db = connect(DB_NAME)
    print(f"✓ Database connected")
    return db


# ============================================================================
# SECTION 7: MAIN CALCULATION LOOP
# ============================================================================

def run_calculations(calculator, db):
    """
    Run bulk energy calculations for all target elements.
    
    Args:
        calculator: FAIRChemCalculator object
        db: ASE database connection
    
    Returns:
        dict: Results dictionary with 'successful', 'failed', 'skipped' keys
    """
    
    results = {
        'successful': [],
        'skipped': [],
        'failed': []
    }
    
    print("\n" + "=" * 90)
    print("BULK ENERGY CALCULATIONS (PER ATOM)")
    print("=" * 90)
    
    # Process each element
    for idx, element in enumerate(TARGET_ELEMENTS, 1):
        try:
            # Build structure
            atoms, structure_type = build_structure(element)
            n_atoms = len(atoms)
            
            # Attach calculator
            atoms.calc = calculator
            
            # Calculate total energy
            bulk_energy_total = atoms.get_potential_energy()
            
            # Calculate energy per atom
            bulk_energy_per_atom = bulk_energy_total / n_atoms
            
            # Write to database
            db.write(atoms, data={
                'element': element,
                'bulk_energy_total': bulk_energy_total,
                'bulk_energy_per_atom': bulk_energy_per_atom,
                'n_atoms': n_atoms,
                'crystal_structure': structure_type,
                'calculation_date': datetime.now().isoformat(),
                'model': 'UMA-OC20'
            })
            
            # Track result
            results['successful'].append((element, structure_type, bulk_energy_per_atom, n_atoms))
            
            # Print progress
            print(f"[{idx:2d}/{len(TARGET_ELEMENTS)}] {element:3s} ({structure_type:32s}) | "
                  f"{bulk_energy_per_atom:>10.6f} eV/atom | {n_atoms:>3d} atoms ✓")
        
        except Exception as e:
            error_msg = str(e)
            results['failed'].append((element, 'unknown', error_msg))
            print(f"[{idx:2d}/{len(TARGET_ELEMENTS)}] {element:3s} - FAILED: {error_msg[:60]}")
    
    return results


# ============================================================================
# SECTION 8: RESULTS SUMMARY & EXPORT
# ============================================================================

def print_summary(results):
    """Print formatted summary of calculation results."""
    
    print("\n" + "=" * 90)
    print("CALCULATION SUMMARY")
    print("=" * 90)
    
    print(f"✓ Successful: {len(results['successful']):2d}/{len(TARGET_ELEMENTS)}")
    print(f"✗ Failed:     {len(results['failed']):2d}/{len(TARGET_ELEMENTS)}")
    if results['skipped']:
        print(f"⊘ Skipped:    {len(results['skipped']):2d}/{len(TARGET_ELEMENTS)}")
    
    # Successful calculations (sorted by energy per atom)
    if results['successful']:
        print("\n" + "-" * 90)
        print("SUCCESSFUL CALCULATIONS (sorted by bulk energy per atom):")
        print("-" * 90)
        print(f"{'Element':<10} {'Structure':<35} {'Energy/Atom (eV)':<20} {'N atoms':<10}")
        print("-" * 90)
        
        for element, structure, energy_per_atom, n_atoms in sorted(results['successful'], key=lambda x: x[2]):
            print(f"{element:<10} {structure:<35} {energy_per_atom:>18.6f} {n_atoms:>8}")
    
    # Failed calculations
    if results['failed']:
        print("\n" + "-" * 90)
        print("FAILED CALCULATIONS:")
        print("-" * 90)
        for element, _, error in results['failed']:
            print(f"  {element:<10} Error: {error[:65]}")
    
    # Skipped calculations
    if results['skipped']:
        print("\n" + "-" * 90)
        print("SKIPPED ELEMENTS:")
        print("-" * 90)
        for element, structure, reason in results['skipped']:
            print(f"  {element:<10} ({reason})")


def export_to_csv(results):
    """Export results to CSV file for further analysis."""
    
    print(f"\nExporting results to: {CSV_NAME}")
    
    with open(CSV_NAME, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Header row
        writer.writerow([
            'Element',
            'Crystal_Structure',
            'Total_Bulk_Energy_eV',
            'Bulk_Energy_per_Atom_eV',
            'N_Atoms',
            'Calculation_Date'
        ])
        
        # Data rows (sorted by energy per atom)
        for element, structure, energy_per_atom, n_atoms in sorted(results['successful'], key=lambda x: x[2]):
            bulk_energy_total = energy_per_atom * n_atoms
            writer.writerow([
                element,
                structure,
                f"{bulk_energy_total:.6f}",
                f"{energy_per_atom:.6f}",
                n_atoms,
                datetime.now().isoformat()
            ])
    
    print(f"✓ Results exported to: {CSV_NAME}")


# ============================================================================
# SECTION 9: MAIN EXECUTION
# ============================================================================

def main(interactive_login=True):
    """
    Main execution function.
    
    Args:
        interactive_login (bool): If True, prompt for API keys. If False, use env vars.
    """
    
    print("=" * 90)
    print("UMA-OC20 MLIP BULK ENERGY CALCULATOR - ALL TARGET ELEMENTS (v9 - HG FIXED)")
    print("=" * 90)
    print(f"Start time:                  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total elements to process:   {len(TARGET_ELEMENTS)}")
    print(f"Materials Project API:       {'✓ Available' if HAS_MATPROJ_API else '✗ Unavailable'}")
    print(f"Database:                    {DB_NAME}")
    print(f"CSV export:                  {CSV_NAME}")
    print("=" * 90)
    
    # Login to APIs
    login_to_apis(interactive=interactive_login)
    
    # Initialize calculator and database
    calculator = initialize_calculator()
    db = setup_database()
    
    # Run calculations
    results = run_calculations(calculator, db)
    
    # Print summary and export
    print_summary(results)
    export_to_csv(results)
    
    # Final message
    print("\n" + "=" * 90)
    print(f"✓ All results saved to: {DB_NAME}")
    print(f"✓ Summary exported to: {CSV_NAME}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Ready to use in surface thermodynamics calculations!")
    print("=" * 90)


# ============================================================================
# JUPYTER NOTEBOOK USAGE
# ============================================================================
# In a Jupyter notebook, run this as:
#
#   from bulk_energies_uma_v9 import main
#   
#   # Run with interactive login prompt:
#   main(interactive_login=True)
#
#   # Or if you already set environment variables:
#   main(interactive_login=False)
#
# Or import and use individual components:
#
#   from bulk_energies_uma_v9 import (
#       login_to_apis,
#       initialize_calculator, 
#       setup_database, 
#       run_calculations,
#       print_summary,
#       export_to_csv,
#       TARGET_ELEMENTS
#   )
#   
#   # Set up API keys first
#   login_to_apis(interactive=True)
#
#   # Then run calculations
#   calc = initialize_calculator()
#   db = setup_database()
#   results = run_calculations(calc, db)
#   print_summary(results)
#   export_to_csv(results)
#
# ============================================================================

if __name__ == "__main__":
    main(interactive_login=True)
