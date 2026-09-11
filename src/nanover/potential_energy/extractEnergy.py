from simtk import unit
from sys import stdout
import mdtraj as md
import numpy as np
import parmed as prm
from openmm.app import Simulation, PME, HBonds, AmberInpcrdFile, AmberPrmtopFile
from openmm import LangevinMiddleIntegrator


class EnergySelect:
    def __init__(self, topology_openmm, system, system_params, integrator_params, selection_md="all"):
        """
        Allows calculation of potential energy for either the full system or a selected subset.
        
        If the selection corresponds to all atoms, then the pre-built full system (with constraints/restrains)
        is used. Otherwise, the class uses ParmEd to slice the original structure and rebuild a new system
        using the same system parameters.
        
        Args:
          topology_openmm: an OpenMM Topology (from your prmtop)
          system: pre-built full OpenMM System (with constraints and frozen masses as applied externally)
          system_params: dict of parameters used to create the system (e.g., nonbondedMethod, cutoff, constraints)
          integrator_params: tuple (temperature, friction, timestep) for LangevinMiddleIntegrator.
          selection_md: either an MDTraj selection string or a list of atom indices.
        """
        # Save system creation parameters for later use.
        self.system_params = system_params
        
        # Convert the full OpenMM topology into an MDTraj topology to use the selection language.
        self.old_topology = md.Topology.from_openmm(topology_openmm)
        topology = md.Topology.from_openmm(topology_openmm)
        print("Total atoms:", topology.n_atoms)
        print("Unique resnames:", set(res.name for res in topology.residues))
        print("Residues:", [(res.index, res.name) for res in topology.residues])

        self.sub_ind = topology.select(selection_md)
        print(f"Selection '{selection_md}' matches {len(self.sub_ind)} atoms")

        # Determine the selected atom indices.
        if isinstance(selection_md, str):
            self.sub_ind = self.old_topology.select(selection_md)
        elif isinstance(selection_md, list):
            self.sub_ind = selection_md
        else:
            raise ValueError("selection_md must be an MDTraj selection string or a list of atom indices.")
        
        # Now, create the sub-topology for use in setting positions.
        # (If "all" atoms are selected, then sub_top = full topology.)
        sub_top = self.old_topology.subset(self.sub_ind)
        print(sub_top)
        # Convert the sub-topology back to an OpenMM topology.
        self.topology = sub_top.to_openmm()
    
        # Decide which system to use.
        # If the selection covers all atoms, use the pre-built full system.
        if len(self.sub_ind) == self.old_topology.n_atoms:
            self.system = system
        else:
            # Otherwise, rebuild a new system corresponding to the subset.
            # Use ParmEd to load the full structure from the prmtop and the full system.
            full_struct = prm.openmm.load_topology(topology_openmm, system)

            
            subset_struct = full_struct[self.sub_ind]
            subset_struct.box = full_struct.box[:] 
             # Fix missing bond types: For any bond in the subset that does not have a type,
            # assign a default bond type. (These default values—k=400 and req=1.0—should match what you expect.)
            default_bond_type = prm.topologyobjects.BondType(k=400, req=1.0)
            for bond in subset_struct.bonds:
                if bond.type is None:
                    bond.type = default_bond_type

            # Now, rebuild the system for the subset using the same system_params.
            self.system = subset_struct.createSystem(**system_params)
        
        # Create the integrator and simulation.
        self.integrator = LangevinMiddleIntegrator(*integrator_params)
        self.simulation = Simulation(self.topology, self.system, self.integrator)
    
    def calc_energy(self, positions):
        """
        Given positions (e.g., frame.xyz from MDTraj), set them as the positions of the simulation
        context. Returns the computed potential energy.
        """
        # Create an MDTraj trajectory from the provided positions and original full topology.
        traj = md.Trajectory(positions, self.old_topology)
        # In all cases, we want to feed the simulation (which now matches the sub-system)
        # with a positions array of shape (#selected atoms, 3).
        new_positions = traj.atom_slice(self.sub_ind).xyz[0]
        
        # (Optionally, ensure that periodic box vectors are set correctly if needed.)
        # For example, if the system uses PME and you have box vectors used in the original system:
        # box_vectors = system.getDefaultPeriodicBoxVectors()
        # self.simulation.context.setPeriodicBoxVectors(*box_vectors)
        
        self.simulation.context.setPositions(new_positions)
        # When testing energy, it is sometimes useful to enforce periodic boundaries.
        state = self.simulation.context.getState(getEnergy=True)
        return state.getPotentialEnergy()

