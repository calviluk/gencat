from ase.io import read, write
from ase import Atoms
from ase.constraints import FixAtoms
import numpy as np
import os
from pymatgen.core import Structure
from pymatgen.analysis.adsorption import AdsorbateSiteFinder

# Finding adsorption sites
def find_unique_surface_sites_pymatgen(slab, site_types=['ontop'], min_distance=0.5, 
                                      adsorbate_height=2.0, max_sites=None):
    """
    Find unique adsorption sites using pymatgen's AdsorbateSiteFinder.
    Much more robust and accurate than manual site detection.
    
    Parameters:
    - slab: ASE Atoms object
    - site_types: List of site types ['ontop', 'bridge', 'hollow'] 
    - min_distance: Minimum distance between sites (Å)
    - adsorbate_height: Height above surface for adsorbate placement (Å)
    - max_sites: Maximum number of sites per type to return (None = no limit)
    
    Returns:
    - List of (site_position, site_type) tuples
    """
    try:
        # Convert ASE slab to pymatgen Structure
        pmg_slab = Structure(
            lattice=slab.cell,
            species=slab.get_chemical_symbols(),
            coords=slab.get_positions(),
            coords_are_cartesian=True
        )
        
        # Debug: Print slab info
        print(f"    Slab composition: {pmg_slab.composition}")
        print(f"    Slab dimensions: {pmg_slab.lattice.abc}")
        print(f"    Number of atoms: {len(pmg_slab)}")
        
        # Create AdsorbateSiteFinder from Pymatgen
        asf = AdsorbateSiteFinder(pmg_slab)
        
        # Find adsorption sites with VERY PERMISSIVE parameters
        ads_sites_dict = asf.find_adsorption_sites(
            distance=adsorbate_height,     # Height above surface
            put_inside=True,               # Ensure sites are within unit cell
            symm_reduce=0.05,              # Quite permissive symmetry reduction
            near_reduce=0.05,              # VERY small minimum distance between sites
            positions=site_types,          # Site types to find
            no_obtuse_hollow=False         # Allow all hollow sites
        )
        
        print(f"    Raw sites found by pymatgen:")
        for site_type, sites in ads_sites_dict.items():
            print(f"      {site_type}: {len(sites)} sites")
        
        total_raw_sites = sum(len(sites) for sites in ads_sites_dict.values())/2  # Each site counted twice in dict
        print(f"    Total raw sites: {int(total_raw_sites)}")
        
        if total_raw_sites == 0:
            print(f"    WARNING: No sites found by pymatgen! Falling back to manual generation...")
            return generate_comprehensive_fallback_sites(slab, site_types, min_distance, adsorbate_height)
        
        # Collect ALL sites FIRST, then apply distance filtering ACROSS ALL TYPES
        all_candidate_sites = []
        for site_type in site_types:
            sites = ads_sites_dict.get(site_type, [])
            print(f"    Processing {len(sites)} {site_type} sites")
            
            if len(sites) == 0:
                print(f"    WARNING: No {site_type} sites found!")
                continue
            
            # Add all sites of this type to candidates (NO filtering yet)
            for site in sites:
                all_candidate_sites.append((site.tolist(), site_type))
                
        print(f"    Total candidate sites before distance filtering: {len(all_candidate_sites)}")
        
        # NOW apply distance filtering across ALL site types. Only if min dist. is specified
        if min_distance > 0:
            filtered_sites = []
            for site_pos, site_type in all_candidate_sites:
                # Check distance to ALL already selected sites (across all types)
                too_close = False
                min_dist_found = float('inf')
                
                for existing_site, existing_type in filtered_sites:
                    distance = np.linalg.norm(np.array(site_pos) - np.array(existing_site))
                    min_dist_found = min(min_dist_found, distance)
                    
                    if distance < min_distance:
                        too_close = True
                        break
                
                if not too_close:
                    filtered_sites.append((site_pos, site_type))
                    print(f"      Added {site_type} site: {site_pos[:2]} (min dist to others: {min_dist_found:.2f} Å)")
                else:
                    print(f"      Filtered out {site_type} site: {site_pos[:2]} (too close: {min_dist_found:.2f} Å)")
        else:
            # No distance filtering
            filtered_sites = all_candidate_sites
            print(f"    No distance filtering applied")
        
        print(f"    Sites after distance filtering: {len(filtered_sites)}")
        
        # Apply max_sites limit PER TYPE if specified. Irrelevant for this project
        if max_sites is not None:
            print(f"    Applying max_sites limit: {max_sites} per type")
            final_sites = []
            
            # Count sites per type and limit each type
            type_counts = {}
            for site_pos, site_type in filtered_sites:
                if site_type not in type_counts:
                    type_counts[site_type] = 0
                
                if type_counts[site_type] < max_sites:
                    final_sites.append((site_pos, site_type))
                    type_counts[site_type] += 1
                    print(f"      Kept {site_type} site #{type_counts[site_type]}")
                else:
                    print(f"      Reached limit for {site_type} sites ({max_sites})")
                    
            filtered_sites = final_sites
        
        print(f"    Total final sites: {len(filtered_sites)}")
        
        # Print final site distribution
        site_counts = {}
        for _, site_type in filtered_sites:
            site_counts[site_type] = site_counts.get(site_type, 0) + 1
        print(f"    Final site distribution: {site_counts}")
        
        # If still no sites, use fallback
        if len(filtered_sites) == 0:
            print(f"    No sites after filtering! Using fallback generation...")
            return generate_comprehensive_fallback_sites(slab, site_types, min_distance, adsorbate_height)
        
        return filtered_sites
        
    except Exception as e:
        print(f"    Error in pymatgen site finding: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback: Create comprehensive manual sites
        print(f"    Falling back to comprehensive manual site generation...")
        return generate_comprehensive_fallback_sites(slab, site_types, min_distance, adsorbate_height)

# Only used if Pymatgen should fail
def generate_comprehensive_fallback_sites(slab, site_types, min_distance, adsorbate_height):
    """
    Comprehensive fallback method to generate many adsorption sites manually.
    """
    print(f"    === COMPREHENSIVE FALLBACK SITE GENERATION ===")
    print(f"    Slab has {len(slab)} atoms")
    print(f"    Requested site types: {site_types}")
    print(f"    Cell dimensions: {slab.cell.lengths()}")
    
    # Get all atom positions and analyze the slab
    positions = slab.get_positions()
    z_coords = positions[:, 2]
    z_min, z_max = z_coords.min(), z_coords.max()
    slab_thickness = z_max - z_min
    
    print(f"    Z range: {z_min:.2f} to {z_max:.2f} Å (thickness: {slab_thickness:.2f} Å)")
    
    # Method 1: Use tags if available
    tags = slab.get_tags()
    if len(tags) > 0 and max(tags) > 0:
        print(f"    Using atom tags for surface detection")
        unique_tags = np.unique(tags)
        print(f"    Available tags: {unique_tags}")
        
        # Take atoms with the highest tags as surface atoms
        max_tags = sorted(unique_tags, reverse=True)[:2]  # Top 2 tag values
        surface_mask = np.isin(tags, max_tags)
        surface_positions = positions[surface_mask]
        print(f"    Surface atoms (tags {max_tags}): {len(surface_positions)} atoms")
    else:
        # Method 2: Use Z-coordinate based detection
        print(f"    Using Z-coordinate based surface detection")
        surface_tolerance = min(2.0, slab_thickness * 0.15)  # Adaptive tolerance
        surface_mask = z_coords > (z_max - surface_tolerance)
        surface_positions = positions[surface_mask]
        print(f"    Surface atoms (z > {z_max - surface_tolerance:.2f}): {len(surface_positions)} atoms")
    
    if len(surface_positions) == 0:
        print(f"    ERROR: No surface atoms found! Using top 25% of atoms by z-coordinate")
        sorted_indices = np.argsort(z_coords)
        top_25_percent = int(0.25 * len(slab))
        surface_indices = sorted_indices[-top_25_percent:]
        surface_positions = positions[surface_indices]
        print(f"    Fallback surface atoms: {len(surface_positions)} atoms")
    
    sites = []
    
    # Generate ONTOP sites - one above each surface atom
    if 'ontop' in site_types:
        print(f"    Generating ONTOP sites...")
        ontop_count = 0
        for i, surf_pos in enumerate(surface_positions):
            site_pos = [surf_pos[0], surf_pos[1], surf_pos[2] + adsorbate_height]
            sites.append((site_pos, 'ontop'))
            ontop_count += 1
        print(f"    Generated {ontop_count} ontop sites")
    
    # Generate BRIDGE sites - midpoints between nearby surface atom pairs
    if 'bridge' in site_types:
        print(f"    Generating BRIDGE sites...")
        bridge_count = 0
        max_bridge_distance = 4.0  # Maximum distance for bridge sites
        
        for i in range(len(surface_positions)):
            for j in range(i+1, len(surface_positions)):
                pos1 = surface_positions[i]
                pos2 = surface_positions[j]
                
                # Calculate 2D distance (ignore z for bridge calculation)
                distance_2d = np.linalg.norm(pos1[:2] - pos2[:2])
                
                if distance_2d < max_bridge_distance:
                    # Bridge site at midpoint, at height of higher atom + adsorbate_height
                    bridge_pos = [
                        (pos1[0] + pos2[0]) / 2,
                        (pos1[1] + pos2[1]) / 2, 
                        max(pos1[2], pos2[2]) + adsorbate_height
                    ]
                    sites.append((bridge_pos, 'bridge'))
                    bridge_count += 1
        
        print(f"    Generated {bridge_count} bridge sites")
    
    # Generate HOLLOW sites - grid-based approach
    if 'hollow' in site_types:
        print(f"    Generating HOLLOW sites...")
        hollow_count = 0
        
        # Create a regular grid over the surface
        if len(surface_positions) >= 3:
            # Get bounding box of surface atoms
            x_coords = surface_positions[:, 0]
            y_coords = surface_positions[:, 1]
            
            x_min, x_max = x_coords.min(), x_coords.max()
            y_min, y_max = y_coords.min(), y_coords.max()
            
            # Create a grid - start with 4x4, can be adjusted
            n_grid = 4
            for i in range(n_grid):
                for j in range(n_grid):
                    # Grid point positions (offset from edges)
                    x = x_min + (x_max - x_min) * (i + 0.3) / (n_grid - 0.4)
                    y = y_min + (y_max - y_min) * (j + 0.3) / (n_grid - 0.4)
                    z = z_max + adsorbate_height
                    
                    hollow_pos = [x, y, z]
                    sites.append((hollow_pos, 'hollow'))
                    hollow_count += 1
        
        print(f"    Generated {hollow_count} hollow sites")
    
    print(f"    Total sites before distance filtering: {len(sites)}")
    
    # Apply distance filtering ONLY if min_distance > 0
    if min_distance > 0:
        print(f"    Applying distance filtering (min_distance = {min_distance:.2f} Å)")
        filtered_sites = []
        
        for site_pos, site_type in sites:
            too_close = False
            min_dist_found = float('inf')
            
            for existing_pos, existing_type in filtered_sites:
                distance = np.linalg.norm(np.array(site_pos) - np.array(existing_pos))
                min_dist_found = min(min_dist_found, distance)
                
                if distance < min_distance:
                    too_close = True
                    break
            
            if not too_close:
                filtered_sites.append((site_pos, site_type))
        
        sites = filtered_sites
        print(f"    Sites after distance filtering: {len(sites)}")
    else:
        print(f"    Skipping distance filtering (min_distance = 0)")
    
    # Print final site distribution
    site_counts = {}
    for _, site_type in sites:
        site_counts[site_type] = site_counts.get(site_type, 0) + 1
    
    print(f"    === FINAL FALLBACK RESULTS ===")
    print(f"    Total sites generated: {len(sites)}")
    print(f"    Site distribution: {site_counts}")
    
    if len(sites) == 0:
        print(f"    ERROR: No sites generated even in fallback! Creating emergency ontop sites...")
        # Emergency: Create at least a few ontop sites
        for i in range(min(5, len(surface_positions))):
            pos = surface_positions[i]
            emergency_site = [pos[0], pos[1], pos[2] + adsorbate_height]
            sites.append((emergency_site, 'ontop'))
        print(f"    Emergency sites created: {len(sites)}")
    
    return sites

# Now adding the adsorbates to the specified adsorption sites.
def add_adsorbates_to_slabs_pymatgen(input_files, output_dir, adsorbate, site_types, 
                                    min_site_distance=1.0, max_sites_per_type=None, 
                                    adsorbate_height=2.0): 
    """
    Add adsorbates to unique surface sites using pymatgen's AdsorbateSiteFinder.
    
    Parameters:
    - input_files: List of trajectory file paths
    - output_dir: Output directory for adsorbate-covered slabs
    - adsorbate: Adsorbate molecule ('CO', 'O', 'H', 'OH', etc.)
    - site_types: List of site types to consider ['ontop', 'bridge', 'hollow']
    - min_site_distance: Minimum distance between adsorption sites (Å)
    - max_sites_per_type: Maximum number of sites per type to consider (None = no limit)
    - adsorbate_height: Height above surface for adsorbate placement (Å)
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Define common adsorbates
    adsorbate_molecules = {
        'CO': Atoms('CO', positions=[[0, 0, 0], [0, 0, 1.15]]),
        'O': Atoms('O', positions=[[0, 0, 0]]),
        'H': Atoms('H', positions=[[0, 0, 0]]),
        'OH': Atoms('OH', positions=[[0, 0, 0], [0, 0, 0.97]]),
        'NO': Atoms('NO', positions=[[0, 0, 0], [0, 0, 1.15]]),
        'CH3': Atoms('CH3', positions=[[0, 0, 0], [0, 0, 1.09], [0.94, 0, -0.36], [-0.47, 0.82, -0.36]]),
        'OOH': Atoms('OOH', positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.45], [0.0, 0.0, 2.35]])
    }
    
    if adsorbate not in adsorbate_molecules:
        print(f"Warning: Adsorbate '{adsorbate}' not defined. Using 'CO' as default.")
        adsorbate = 'CO'
    
    adsorbate_mol = adsorbate_molecules[adsorbate]
    
    total_structures_processed = 0
    total_structures_saved = 0
    
    for input_file in input_files:
        if not os.path.exists(input_file):
            print(f"Warning: File {input_file} not found. Skipping...")
            continue
            
        print(f"\nProcessing: {os.path.basename(input_file)}")
        
        # Read all slabs from trajectory file
        slabs = read(input_file, index=':')
        print(f"  Loaded {len(slabs)} slabs")
        
        all_adsorbate_slabs = []
        file_structures_processed = 0
        file_structures_saved = 0
        
        # Process each slab
        for slab_idx, slab in enumerate(slabs):
            try:
                print(f"  Processing slab {slab_idx + 1}/{len(slabs)}")
                
                # Find adsorption sites using pymatgen AdsorbateSiteFinder
                all_sites = find_unique_surface_sites_pymatgen(
                    slab, 
                    site_types=site_types,
                    min_distance=min_site_distance,
                    adsorbate_height=adsorbate_height,
                    max_sites=max_sites_per_type  # This can be None for unlimited sites
                )
                
                # Print site summary
                site_counts_by_type = {}
                for _, site_type in all_sites:
                    site_counts_by_type[site_type] = site_counts_by_type.get(site_type, 0) + 1
                
                print(f"    Final site counts using pymatgen AdsorbateSiteFinder:")
                for site_type, count in site_counts_by_type.items():
                    print(f"      {site_type}: {count} sites")
                print(f"    Total sites: {len(all_sites)}")

                # Create adsorbate-covered slabs for each site
                for site_idx, (site_pos, site_type) in enumerate(all_sites):
                    # Copy the original slab
                    adsorbate_slab = slab.copy()
                    
                    # Add adsorbate at the site (site_pos is already in Cartesian coordinates)
                    adsorbate_copy = adsorbate_mol.copy()
                    adsorbate_copy.translate(site_pos)
                    adsorbate_slab.extend(adsorbate_copy)
                    
                    total_structures_processed += 1
                    file_structures_processed += 1
                    
                    # Preserve constraints from original slab
                    if slab.constraints:
                        for constraint in slab.constraints:
                            if isinstance(constraint, FixAtoms):
                                adsorbate_slab.set_constraint(FixAtoms(indices=constraint.index))
                    
                    # Add metadata
                    adsorbate_slab.info.update(slab.info)
                    adsorbate_slab.info['adsorbate'] = adsorbate
                    adsorbate_slab.info['adsorption_site'] = site_type
                    adsorbate_slab.info['site_position'] = site_pos
                    adsorbate_slab.info['site_index'] = site_idx + 1
                    adsorbate_slab.info['original_slab_index'] = slab_idx + 1
                    adsorbate_slab.info['site_finder'] = 'pymatgen_AdsorbateSiteFinder'
                    
                    all_adsorbate_slabs.append(adsorbate_slab)
                    total_structures_saved += 1
                    file_structures_saved += 1
                    
                    print(f"      Site {site_idx + 1} ({site_type}): Added adsorbate at {site_pos}")
                
            except Exception as e:
                print(f"  Error processing slab {slab_idx + 1}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        ##################################
        # There needs to be a calculation of the energy of the structure with the adsorbate here.
        # Then it is to be subtracted by the energy of the clean slab and the energy of the adsorbate in gas phase.
        # This is not implemented yet.

        ##################################

        # Save all adsorbate-covered slabs
        if all_adsorbate_slabs:
            base_name = os.path.basename(input_file).replace('.traj', '')
            output_filename = os.path.join(output_dir, f"{base_name}_{adsorbate}_adsorbates_pymatgen.traj")
            
            write(output_filename, all_adsorbate_slabs)
            
            print(f"  Saved {len(all_adsorbate_slabs)} adsorbate-covered slabs to: {os.path.basename(output_filename)}")
            print(f"  File processing: {file_structures_saved} structures saved from {file_structures_processed} sites")
            
            # Print summary statistics
            total_atoms = sum(len(slab) for slab in all_adsorbate_slabs)
            original_atoms = sum(len(slab) - len(adsorbate_mol) for slab in all_adsorbate_slabs)
            adsorbate_atoms = total_atoms - original_atoms
            
            print(f"    Total atoms: {total_atoms} (original: {original_atoms}, adsorbate: {adsorbate_atoms})")
            
            # Count sites by type
            site_counts = {}
            for slab in all_adsorbate_slabs:
                site_type = slab.info.get('adsorption_site', 'unknown')
                site_counts[site_type] = site_counts.get(site_type, 0) + 1
            
            print(f"    Site distribution: {site_counts}")
        else:
            print(f"  No adsorbate-covered slabs generated for {input_file}")
    
    # Print overall statistics - THIS WAS MISSING!
    print(f"\n" + "="*60)
    print(f"OVERALL STATISTICS (using pymatgen AdsorbateSiteFinder)")
    print(f"="*60)
    print(f"Total sites processed: {total_structures_processed}")
    print(f"Total structures saved: {total_structures_saved}")
    print(f"\nAdsorbate addition complete! Files saved in: {output_dir}")

# Define your input files
input_files = [
    r"./grouped_slab_trajectories/slabs_miller_1_0_0.traj",
    r"./grouped_slab_trajectories/slabs_miller_1_1_0.traj",
    r"./grouped_slab_trajectories/slabs_miller_1_1_1.traj"
]
# Add OH adsorbates using pymatgen AdsorbateSiteFinder - GET ALL SITES
add_adsorbates_to_slabs_pymatgen(
    input_files=input_files,
    output_dir="adsorbate_covered_slabs_pymatgen_all_sites_included",
    adsorbate='OH',
    site_types=['ontop', 'bridge', 'hollow'],  # All site types
    min_site_distance=0.3,      # VERY small minimum distance to keep more sites
    max_sites_per_type=None,    # NO LIMIT on sites per type
    adsorbate_height=1.5
)

