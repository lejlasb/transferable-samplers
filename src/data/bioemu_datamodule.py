# src/data/local_single_peptide_datamodule.py
import logging
import os
from typing import Optional

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


class LocalSinglePeptideDataModule(BaseDataModule):
    def __init__(
        self,
        data_dir: str,
        sequence: str,
        temperature: float,
        num_dimensions: int,
        num_atoms: int,
        com_augmentation: bool = False,
        num_eval_samples: int = 10_000,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

        self.trajectory_name = f"{self.hparams.sequence}_{self.hparams.temperature}K"

        # Setup paths - match the original structure but without repo_name subdirectory
        self.trajectory_data_dir = f"{data_dir}/{self.trajectory_name}"
        self.train_data_path = f"{self.trajectory_data_dir}/{self.trajectory_name}_train.npy"
        self.val_data_path = f"{self.trajectory_data_dir}/{self.trajectory_name}_val.npy"
        self.test_data_path = f"{self.trajectory_data_dir}/{self.trajectory_name}_test.npy"
        self.pdb_path = f"{self.trajectory_data_dir}/{self.trajectory_name}.pdb"

        # For compatibility with transferable case
        self.val_sequences = [self.hparams.sequence]
        self.test_sequences = [self.hparams.sequence]

    def prepare_data(self):
        """Load + preprocessing data. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        # Create directory structure (like original does)
        os.makedirs(self.trajectory_data_dir, exist_ok=True)

        # Check if preprocessed files already exist
        if all(os.path.exists(path) for path in [self.train_data_path, self.val_data_path, self.test_data_path]):
            logging.info(f"Preprocessed data already exists in {self.trajectory_data_dir}")
            return

        # If preprocessed .npy files don't exist, create them from trajectory
        logging.info("Preprocessed .npy files not found, looking for trajectory files...")
        self._create_preprocessed_data()

    def _create_preprocessed_data(self):
        """Create preprocessed .npy files from trajectory data."""
        # Look for common trajectory file extensions
        trajectory_extensions = ['.xtc', '.trr', '.dcd', '.h5', '.nc']
        trajectory_file = None
        
        for ext in trajectory_extensions:
            potential_path = os.path.join(os.path.dirname(self.pdb_path), f"{self.trajectory_name}{ext}")
            if os.path.exists(potential_path):
                trajectory_file = potential_path
                logging.info(f"Found trajectory file: {trajectory_file}")
                break
        
        if trajectory_file is None:
            # Try any trajectory file in the directory
            data_dir = os.path.dirname(self.pdb_path)
            for ext in trajectory_extensions:
                files = [f for f in os.listdir(data_dir) if f.endswith(ext)]
                if files:
                    trajectory_file = os.path.join(data_dir, files[0])
                    logging.info(f"Found trajectory file: {trajectory_file}")
                    break
        
        if trajectory_file is None:
            raise FileNotFoundError(f"No trajectory file found for {self.trajectory_name}. "
                                  f"Looked for extensions: {trajectory_extensions}")

        # Load trajectory
        logging.info(f"Loading trajectory from {trajectory_file}...")
        traj = md.load(trajectory_file, top=self.pdb_path)
        
        # Extract coordinates and reshape
        coordinates = traj.xyz  # Shape: (n_frames, n_atoms, 3)
        coordinates_flat = coordinates.reshape(coordinates.shape[0], -1)  # Shape: (n_frames, n_atoms * 3)
        
        logging.info(f"Loaded {len(coordinates_flat)} samples with {coordinates_flat.shape[1]} features")

        # Split data
        n_total = len(coordinates_flat)
        n_train = int(0.8 * n_total)  # 80% train
        n_val = int(0.1 * n_total)    # 10% val
        
        # Shuffle before splitting
        rng = np.random.default_rng(42)
        indices = rng.permutation(n_total)
        shuffled_coordinates = coordinates_flat[indices]

        train_data = shuffled_coordinates[:n_train]
        val_data = shuffled_coordinates[n_train:n_train + n_val]
        test_data = shuffled_coordinates[n_train + n_val:]

        # Save as .npy files
        np.save(self.train_data_path, train_data)
        np.save(self.val_data_path, val_data)
        np.save(self.test_data_path, test_data)
        
        logging.info(f"Created preprocessed data: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number "
                    "of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size
        else:
            self.batch_size_per_device = self.hparams.batch_size

        # Load the data
        data_train = np.load(self.train_data_path)
        data_val = np.load(self.val_data_path)
        data_test = np.load(self.test_data_path)

        # Tensorize the data
        data_train = torch.from_numpy(data_train).float()
        data_val = torch.from_numpy(data_val).float()
        data_test = torch.from_numpy(data_test).float()

        # Load the PDB file
        self.pdb = openmm.app.PDBFile(self.pdb_path)

        # Load the topology from the PDB file
        self.topology = md.load_topology(self.pdb_path)

        # For compatibility to transferable BG
        self.topology_dict = {self.hparams.sequence: self.topology}

        # Compute std on standardized data
        self.std = (data_train - data_train.mean(dim=1, keepdim=True)).std()

        transform_list = [
            StandardizeTransform(self.std),
            Random3DRotationTransform(),
        ]
        if self.hparams.com_augmentation:
            transform_list.append(CenterOfMassTransform())
        train_transforms = torchvision.transforms.Compose(transform_list)
        self.data_train = TensorDataset(
            data=data_train,
            transform=train_transforms,
        )

        test_transforms = StandardizeTransform(self.std)
        self.data_val = TensorDataset(
            data=data_val,
            transform=test_transforms,
        )
        self.data_test = TensorDataset(
            data=data_test,
            transform=test_transforms,
        )

        logging.info(f"Train dataset size: {len(self.data_train)}")
        logging.info(f"Validation dataset size: {len(self.data_val)}")
        logging.info(f"Test dataset size: {len(self.data_test)}")

    def setup_potential(self):
        """
        Set up the OpenMM potential energy function.

        Returns:
            OpenMMEnergy: An energy function wrapper around the OpenMM system and integrator.
        """
        # Use the same logic as original for different sequences
        if self.hparams.sequence in ["Ace-A-Nme", "Ace-AAA-Nme"]:
            forcefield = openmm.app.ForceField("amber99sbildn.xml", "tip3p.xml", "amber99_obc.xml")

            system = forcefield.createSystem(
                self.pdb.topology,
                nonbondedMethod=openmm.app.NoCutoff,
                nonbondedCutoff=0.9 * openmm.unit.nanometer,
                constraints=None,
            )
            temperature = 300
            integrator = openmm.LangevinMiddleIntegrator(
                temperature * openmm.unit.kelvin,
                0.3 / openmm.unit.picosecond
                if self.hparams.sequence == "Ace-AAA-Nme"
                else 1.0 / openmm.unit.picosecond,
                1.0 * openmm.unit.femtosecond,
            )
            potential = OpenMMEnergy(bridge=OpenMMBridge(system, integrator, platform_name="CUDA"))
        else:
            # This will apply to your chignolin sequence "GYDPETGTWG"
            forcefield = openmm.app.ForceField("amber14-all.xml", "implicit/obc1.xml") # "amber14-all.xml", "implicit/obc1.xml"
            temperature = 310

            system = forcefield.createSystem(
                self.pdb.topology,
                nonbondedMethod=openmm.app.CutoffNonPeriodic,
                nonbondedCutoff=2.0 * openmm.unit.nanometer,
                constraints=None,
            )
            integrator = openmm.LangevinMiddleIntegrator(
                temperature * openmm.unit.kelvin,
                0.3 / openmm.unit.picosecond,
                1.0 * openmm.unit.femtosecond,
            )

            # Initialize potential
            potential = OpenMMEnergy(bridge=OpenMMBridge(system, integrator, platform_name="CUDA"))

        return potential

    def prepare_eval(self, sequence: str, prefix: str):
        """
        Prepare evaluation data and energy function for validation or test trajectories.

        Selects trajectory data based on the provided prefix, constructs a TICA model,
        subsamples the trajectory, applies normalization, and sets up a potential energy
        function. Returns all components required for evaluation.

        Args:
            sequence (str): Unused compatibility argument for integration with
                TransferablePeptideDatamodule.
            prefix (str): Dataset split to evaluate on. Must be either "val" or "test".

        Returns:
            tuple: A 5-tuple containing:
                - true_samples (torch.Tensor): Normalized and subsampled trajectory samples.
                - permutations (None): Placeholder for compatibility, not used here.
                - encodings (None): Placeholder for compatibility, not used here.
                - energy_fn (Callable): Function mapping positions → energy values.
                - tica_model: Model with TICA projection parameters computed from the trajectory.
        """
        if prefix == "test":
            true_samples = self.data_test.data
        elif prefix == "val":
            true_samples = self.data_val.data
        else:
            raise ValueError(f"Unknown prefix: {prefix}. Use 'val' or 'test'.")

        tica_model = get_tica_model(true_samples, self.topology)

        # Subsample the true trajectory
        true_samples = true_samples[:: len(true_samples) // self.hparams.num_eval_samples]
        true_samples = self.normalize(true_samples)

        permutations = None
        encodings = None
        potential = self.setup_potential()
        energy_fn = lambda x: potential.energy(self.unnormalize(x)).flatten()

        return true_samples, permutations, encodings, energy_fn, tica_model
