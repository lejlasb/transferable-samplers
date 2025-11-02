# src/data/bioemu_datamodule.py
import logging
import os
import glob
import random
from typing import Optional, List, Union

import mdtraj as md
import numpy as np
import openmm
import openmm.app
import torch
import torchvision

from src.data.base_datamodule import BaseDataModule
from src.data.datasets.tensor_dataset import TensorDataset
from src.data.energy.openmm import OpenMMBridge, OpenMMEnergy
from src.data.preprocessing.tica import get_tica_model
from src.data.transforms.center_of_mass import CenterOfMassTransform
from src.data.transforms.rotation import Random3DRotationTransform
from src.data.transforms.standardize import StandardizeTransform


class BioEmuDataModule(BaseDataModule):
    def __init__(
        self,
        data_dir: str,
        topology_file: str,
        data_files: Union[str, List[str]],
        sequence: str,
        temperature: float = 300,
        num_dimensions: int = 3,
        num_atoms: int = -1,  # CHANGED: -1 means auto-detect from topology
        com_augmentation: bool = False,
        num_eval_samples: int = 10_000,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        train_split: float = 0.8,
        val_split: float = 0.1,
        # BioEmu specific parameters
        use_positions: bool = True,
        positions_key: str = "pos",
        orientations_key: str = "node_orientations",
        shuffle_files: bool = True,
        file_type: str = "auto",
        stride: int = 1,
        # NEW: Atom selection parameters
        atom_selection: str = "all",  # "all", "backbone", "ca" (alpha carbons), "heavy" (no hydrogens)
        use_angstroms: bool = True,  # Convert XTC from nm to angstroms
    ):
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

        # Store additional parameters
        self.train_split = train_split
        self.val_split = val_split
        self.use_positions = use_positions
        self.positions_key = positions_key
        self.orientations_key = orientations_key
        self.shuffle_files = shuffle_files
        self.file_type = file_type
        self.stride = stride
        self.atom_selection = atom_selection
        self.use_angstroms = use_angstroms

        self.trajectory_name = f"{sequence}_{temperature}K"

        # Setup paths
        self.topology_path = os.path.join(data_dir, topology_file)
        
        # Handle data files
        self.data_files = self._resolve_data_files(data_dir, data_files)
        
        if not self.data_files:
            raise ValueError(f"No data files found matching: {data_files}")

        # Create output paths for processed data
        file_ext = "xtc" if self._is_xtc_file(self.data_files[0]) else "npz"
        base_name = f"{sequence}_{self.atom_selection}_{len(self.data_files)}files_{file_ext}"
        self.train_data_path = os.path.join(data_dir, f"{base_name}_train.npy")
        self.val_data_path = os.path.join(data_dir, f"{base_name}_val.npy")
        self.test_data_path = os.path.join(data_dir, f"{base_name}_test.npy")

        # For compatibility with transferable case
        self.val_sequences = [sequence]
        self.test_sequences = [sequence]

        # Will be set during setup
        self.actual_num_atoms = None

    def _is_xtc_file(self, file_path: str) -> bool:
        """Check if file is XTC format."""
        return file_path.lower().endswith(('.xtc', '.trr', '.dcd'))

    def _resolve_data_files(self, data_dir: str, data_files: Union[str, List[str]]) -> List[str]:
        """Resolve data files from pattern or list."""
        if isinstance(data_files, str):
            pattern = os.path.join(data_dir, data_files)
            files = sorted(glob.glob(pattern))
            logging.info(f"Found {len(files)} files matching pattern: {data_files}")
        elif isinstance(data_files, list):
            files = [os.path.join(data_dir, f) for f in data_files]
            logging.info(f"Using specified {len(files)} files")
        else:
            raise ValueError("data_files must be string pattern or list of files")
        
        missing_files = [f for f in files if not os.path.exists(f)]
        if missing_files:
            raise FileNotFoundError(f"Data files not found: {missing_files}")
            
        if self.file_type == "auto":
            if self._is_xtc_file(files[0]):
                self.file_type = "xtc"
                logging.info(f"Auto-detected XTC trajectory files")
            else:
                self.file_type = "npz"
                logging.info(f"Auto-detected NPZ data files")
        
        if self.shuffle_files and len(files) > 1:
            random.shuffle(files)
            logging.info("Shuffled data file loading order")
            
        return files

    def _get_atom_selection(self, topology: md.Topology) -> np.ndarray:
        """Get atom indices based on selection criteria."""
        if self.atom_selection == "all":
            indices = np.arange(topology.n_atoms)
            logging.info(f"Selected ALL atoms: {len(indices)} atoms")
        elif self.atom_selection == "heavy":
            # All non-hydrogen atoms
            indices = topology.select("element != H")
            logging.info(f"Selected HEAVY atoms: {len(indices)} atoms")
        elif self.atom_selection == "backbone":
            # Backbone atoms (N, CA, C, O)
            indices = topology.select("backbone")
            logging.info(f"Selected BACKBONE atoms: {len(indices)} atoms")
        elif self.atom_selection == "ca":
            # Alpha carbons only
            indices = topology.select("name CA")
            logging.info(f"Selected ALPHA CARBONS: {len(indices)} atoms")
        elif self.atom_selection == "backbone_heavy":
            # Backbone heavy atoms (N, CA, C)
            indices = topology.select("backbone and element != H")
            logging.info(f"Selected BACKBONE HEAVY atoms: {len(indices)} atoms")
        else:
            # Custom selection string
            try:
                indices = topology.select(self.atom_selection)
                logging.info(f"Selected custom '{self.atom_selection}': {len(indices)} atoms")
            except Exception as e:
                logging.warning(f"Custom selection failed, using all atoms: {e}")
                indices = np.arange(topology.n_atoms)
        
        if len(indices) == 0:
            logging.warning(f"No atoms selected with criteria '{self.atom_selection}', using all atoms")
            indices = np.arange(topology.n_atoms)
            
        return indices

    def prepare_data(self):
        """Load and preprocess BioEmu data from NPZ or XTC files."""
        if not os.path.exists(self.topology_path):
            raise FileNotFoundError(f"Topology file not found: {self.topology_path}")

        # Load topology to determine atom selection
        try:
            topology = md.load_topology(self.topology_path)
            self.atom_indices = self._get_atom_selection(topology)
            self.actual_num_atoms = len(self.atom_indices)
            logging.info(f"Atom selection: {self.actual_numoms} atoms will be used")
        except Exception as e:
            logging.error(f"Error loading topology for atom selection: {e}")
            raise

        logging.info(f"Loading data from {len(self.data_files)} {self.file_type.upper()} files:")
        for i, data_file in enumerate(self.data_files):
            logging.info(f"  {i+1}. {os.path.basename(data_file)}")

        all_coordinates = []
        
        for data_file in self.data_files:
            logging.info(f"Processing {os.path.basename(data_file)}...")
            
            if self.file_type == "xtc":
                coordinates = self._load_xtc_file(data_file)
            else:
                coordinates = self._load_npz_file(data_file)
            
            if self.stride > 1:
                coordinates = coordinates[::self.stride]
                logging.info(f"  Applied stride {self.stride}: {len(coordinates)} samples")
            
            all_coordinates.append(coordinates)
            logging.info(f"  Loaded {len(coordinates)} samples")

        # Combine all coordinates
        combined_coordinates = np.concatenate(all_coordinates, axis=0)
        total_samples = len(combined_coordinates)
        logging.info(f"Combined total of {total_samples} samples from {len(self.data_files)} files")
        
        # Split data into train/val/test
        n_train = int(self.train_split * total_samples)
        n_val = int(self.val_split * total_samples)
        
        # Shuffle the combined data before splitting
        rng = np.random.default_rng(42)
        indices = rng.permutation(total_samples)
        shuffled_coordinates = combined_coordinates[indices]
        
        train_data = shuffled_coordinates[:n_train]
        val_data = shuffled_coordinates[n_train:n_train + n_val]
        test_data = shuffled_coordinates[n_train + n_val:]
        
        # Save split data
        np.save(self.train_data_path, train_data)
        np.save(self.val_data_path, val_data)
        np.save(self.test_data_path, test_data)
        
        logging.info(f"Data split: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
        logging.info(f"Saved processed data to {self.hparams.data_dir}")

    def _load_xtc_file(self, xtc_file: str) -> np.ndarray:
        """Load coordinates from XTC trajectory file with proper atom selection."""
        try:
            # Load trajectory using MDTraj with atom selection
            traj = md.load(xtc_file, top=self.topology_path, atom_indices=self.atom_indices)
            logging.info(f"  Loaded XTC trajectory: {traj.n_frames} frames, {traj.n_atoms} atoms")
            
            # Extract coordinates (in nanometers by default)
            coordinates = traj.xyz  # Shape: (frames, atoms, 3)
            
            # Convert from nanometers to angstroms if requested
            if self.use_angstroms:
                coordinates = coordinates * 10.0  # nm to angstroms
                logging.info("  Converted coordinates from nm to angstroms")
            
            # Validate we have the expected number of atoms
            if coordinates.shape[1] != self.actual_num_atoms:
                logging.warning(f"  Expected {self.actual_num_atoms} atoms, got {coordinates.shape[1]}")
            
            return coordinates
            
        except Exception as e:
            logging.error(f"Error loading XTC file {xtc_file}: {e}")
            raise

    def _load_npz_file(self, npz_file: str) -> np.ndarray:
        """Load coordinates from NPZ file."""
        data = np.load(npz_file)
        
        available_keys = list(data.files)
        logging.info(f"  Available keys: {available_keys}")
        
        if self.use_positions:
            if self.positions_key in data.files:
                coordinates = data[self.positions_key]
                logging.info(f"  Using '{self.positions_key}' array with shape: {coordinates.shape}")
            else:
                for key in data.files:
                    arr = data[key]
                    if arr.ndim == 3 and arr.shape[-1] == 3:
                        coordinates = arr
                        logging.info(f"  Using fallback array '{key}' with shape: {coordinates.shape}")
                        break
                else:
                    raise KeyError(f"Position key '{self.positions_key}' not found")
        else:
            if self.orientations_key in data.files:
                orientations = data[self.orientations_key]
                coordinates = orientations[:, :, 0, :]
            else:
                raise KeyError(f"Orientations key '{self.orientations_key}' not found")

        return self._validate_and_reshape_coordinates(coordinates)

    def _validate_and_reshape_coordinates(self, coordinates):
        """Validate coordinates shape and reshape if necessary."""
        if coordinates.ndim == 3 and coordinates.shape[-1] == 3:
            return coordinates
        elif coordinates.ndim == 4 and coordinates.shape[-2:] == (3, 3):
            logging.info("  Detected orientation data, extracting positions...")
            coordinates = coordinates[:, :, 0, :]
            logging.info(f"  Extracted positions shape: {coordinates.shape}")
            return coordinates
        else:
            logging.warning(f"  Unexpected coordinate shape: {coordinates.shape}. Attempting to reshape.")
            if coordinates.size % 3 == 0:
                frames = coordinates.shape[0]
                atoms = coordinates.size // (frames * 3)
                coordinates = coordinates.reshape(frames, atoms, 3)
                logging.info(f"  Reshaped coordinates to: {coordinates.shape}")
                return coordinates
            else:
                raise ValueError(f"Cannot reshape coordinates with shape {coordinates.shape} to (frames, atoms, 3)")

    def setup(self, stage: Optional[str] = None) -> None:
        """Load the preprocessed data and set up datasets."""
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number "
                    "of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size
        else:
            self.batch_size_per_device = self.hparams.batch_size

        # Check if processed data exists
        if not all(os.path.exists(path) for path in [self.train_data_path, self.val_data_path, self.test_data_path]):
            logging.info("Processed data not found, running prepare_data...")
            self.prepare_data()

        # Load the preprocessed data
        data_train = np.load(self.train_data_path)
        data_val = np.load(self.val_data_path)
        data_test = np.load(self.test_data_path)

        # Tensorize the data
        data_train = torch.from_numpy(data_train).float()
        data_val = torch.from_numpy(data_val).float()
        data_test = torch.from_numpy(data_test).float()

        # Load the topology file
        try:
            if self.topology_path.endswith('.pdb'):
                self.topology = md.load_topology(self.topology_path)
            elif self.topology_path.endswith('.psf'):
                self.topology = md.load_psf(self.topology_path)
            else:
                self.topology = md.load_topology(self.topology_path)

            logging.info(f"Loaded topology with {self.topology.n_atoms} atoms")
        except Exception as e:
            logging.warning(f"Could not load topology with mdtraj: {e}")
            self.topology = None

        # Update actual number of atoms based on processed data
        self.actual_num_atoms = data_train.shape[1]
        logging.info(f"Using {self.actual_num_atoms} atoms from processed data")

        # For compatibility to transferable BG
        self.topology_dict = {self.hparams.sequence: self.topology}

        # Compute std on standardized data
        data_train_com = data_train - data_train.mean(dim=1, keepdim=True)
        self.std = data_train_com.std()
        logging.info(f"Computed standardization std: {self.std}")

        # Setup transforms
        transform_list = [
            StandardizeTransform(self.std),
            Random3DRotationTransform(),
        ]
        if self.hparams.com_augmentation:
            transform_list.append(CenterOfMassTransform())

        train_transforms = torchvision.transforms.Compose(transform_list)
        test_transforms = StandardizeTransform(self.std)

        # Create datasets
        self.data_train = TensorDataset(
            data=data_train,
            transform=train_transforms,
        )

        self.data_val = TensorDataset(
            data=data_val,
            transform=test_transforms,
        )

        self.data_test = TensorDataset(
            data=data_test,
            transform=test_transforms,
        )

        logging.info(f"Final dataset sizes - Train: {len(self.data_train)}, Val: {len(self.data_val)}, Test: {len(self.data_test)}")

    def setup_potential(self):
        """
        Set up OpenMM potential for full atomic data.
        Now we can compute proper energies since we have complete atoms.
        """
        try:
            logging.info("Setting up OpenMM potential with full atomic data")
            
            # Load PDB file
            pdb = openmm.app.PDBFile(self.topology_path)
            topology = pdb.topology

            # Use appropriate force field
            # For proteins with explicit or implicit solvent
            try:
                # Try with explicit solvent first
                forcefield = openmm.app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
                system = forcefield.createSystem(
                    topology,
                    nonbondedMethod=openmm.app.NoCutoff,
                    constraints=openmm.app.HBonds,
                    rigidWater=True
                )
            except:
                # Fallback to implicit solvent
                forcefield = openmm.app.ForceField('amber14-all.xml', 'implicit/gbn2.xml')
                system = forcefield.createSystem(
                    topology,
                    nonbondedMethod=openmm.app.NoCutoff,
                    constraints=openmm.app.HBonds
                )

            # Create integrator
            integrator = openmm.LangevinMiddleIntegrator(
                self.hparams.temperature * openmm.unit.kelvin,
                1.0 / openmm.unit.picosecond,  # Friction coefficient
                2.0 * openmm.unit.femtoseconds,  # Time step
            )

            # Initialize potential
            platform_name = "CUDA" if torch.cuda.is_available() else "CPU"
            potential = OpenMMEnergy(
                bridge=OpenMMBridge(system, integrator, platform_name=platform_name)
            )

            logging.info("Successfully set up OpenMM potential with full atomic data")
            return potential

        except Exception as e:
            logging.error(f"Could not set up OpenMM potential: {e}")
            # Fallback to dummy potential
            logging.warning("Falling back to dummy potential")
            
            class DummyPotential:
                def energy(self, x):
                    return torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

            return DummyPotential()

    def prepare_eval(self, sequence: str, prefix: str):
        """
        Prepare evaluation data for full atomic data.
        """
        if prefix == "test":
            true_samples = self.data_test.data
        elif prefix == "val":
            true_samples = self.data_val.data
        else:
            raise ValueError(f"Unknown prefix: {prefix}. Use 'val' or 'test'.")

        # TICA model - now works with full atomic data
        tica_model = None
        if self.topology is not None:
            try:
                tica_model = get_tica_model(true_samples, self.topology)
            except Exception as e:
                logging.warning(f"Could not create TICA model: {e}")

        # Subsample the true trajectory
        subsample_rate = max(1, len(true_samples) // self.hparams.num_eval_samples)
        true_samples = true_samples[::subsample_rate]
        true_samples = self.normalize(true_samples)

        permutations = None
        encodings = None

        potential = self.setup_potential()
        energy_fn = lambda x: potential.energy(self.unnormalize(x)).flatten()

        return true_samples, permutations, encodings, energy_fn, tica_model
