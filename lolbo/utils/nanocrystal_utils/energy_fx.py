from __future__ import division, unicode_literals, print_function

"""
This module performs relaxations and then populates the following data for
a model:

1. evaluate_energy : does structure relaxation, and gives energy

2. energy_obj : The first objective function (energy) is calculated
"""


from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.io.lammps.data import LammpsData
from pymatgen.core.periodic_table import Element
from pymatgen.io.vasp.inputs import Poscar

import os
import yaml
import shutil
import datetime
import numpy as np
import subprocess as sp
import re

DEBUG = False

# TODO: Create an energy_code base class for vasp_code and lammps_code

class lammps_code(object):

    def __init__(self, energy_params):
        """
        energy_params: dictionary of all the parameters

        Eg:
        ```python
        {'main_path': <path to directory in which fantastx is ran>,
        'shape': 'gb',
        'energy_files_path': <path_to_input_files>,
        'energy_exec_cmd': 'lmp_mpi -in in.min',
        'sym_mu_dict': {'Al': -3.35958515625, 'O': -6.76069604253},
        'atom_style': 'charge'}
        ```
        """
        self.main_path = energy_params['main_path']
        self.shape = energy_params['shape']
        # path to make new folder to do calculations
        self.relax_path = None
        # path to copy input files for each calculation
        self.energy_files_path = energy_params['files_path']
        # lammps execution command as a string
        # Ex: 'lmp_mpi -in in.min'
        self.energy_exec_cmd = energy_params['energy_exec_cmd']
        # Make a new folder to store all lammps.label
        # (log files) for convenience
        # save hollow_botz and hollow_topz for use in sd_flags
        self.hollow_botz = None
        self.hollow_topz = None

        # DU
        # Save species names and their chemical potentials for identification
        self.sym_mu_dict = {}
        for key, value in energy_params['element_syms'].items():
            self.sym_mu_dict[value] = energy_params['mu'][key]

        # chemical potentials of each species

        # default atom_style
        def_atom_style = 'charge'
        self.atom_style = def_atom_style
        if 'atom_style' in energy_params:
            self.atom_style = energy_params['atom_style']

    def prep_job_folder(self, astr, astr_label):
        """
        Function to:

        - Check the provided input files (if any)
        - Copy the input files to the model calc directory (relax_path)
            - The input files for lammps: **in.min** and **potential** file

        Arguments:

            model (obj): `structure_record.model()` object for which energy
             evaluation will be done

            reg_id (obj): `structure_record.register_id()` object for
             bookkeeping
        """
        main_path = self.main_path
        model_path = main_path + '/calcs/' + str(astr_label)
        os.mkdir(model_path)

        relax_path = main_path + '/calcs/' + str(astr_label) + '/relax'
        os.mkdir(relax_path)
        
        #model.relax_path = relax_path

        files_path = self.energy_files_path
        atom_style = self.atom_style
        # write model structure to POSCAR and store it in /relax
        new_poscar = relax_path + '/POSCAR_unrelaxed'
        sd_flags = [[0, 0, 0] for i in range(len(astr))]
        astr_poscar = Poscar(astr, selective_dynamics=sd_flags)
        astr_poscar.write_file(new_poscar)
        # check if both files exist.
        file_list = os.listdir(files_path)
        # For LAMMPS: check for in.min file in the files_path
        if 'in.min' in file_list:
            input_file = files_path + '/in.min'
        else:
            print('in.min file not present in energy_inputs_path. This file '
                  'is mandatory.')
        # copy files from files_path to relax_path
        shutil.copy(input_file, relax_path)
        # NOTE: The path to lammps potential file should be specified in in.min
        # get data_file and copy to calc path
        lammps_data = LammpsData.from_structure(astr, ff_elements=None,
                                                atom_style=atom_style)
        # write data file for lammps in relax_path
        data_file_path = relax_path + '/in.data'
        lammps_data.write_file(data_file_path)

        # (optional) make any changes to the in.min if necessary

        # returns nothing

    def get_score(self, astr, astr_label):
        """
        Starts the lammps relaxation in the calcs/<model label> path. Assigns
        the evaluated total energy and obj0_val to model attributes.

        Arguments:

            model (obj): `structure_record.model()` object for which energy
             evaluation will be done

            reg_id (obj): `structure_record.register_id()` object for
             bookkeeping
        """
        print(f"Prepping the job folder of model {astr_label}.")
        # prepare the folder to start energy calc
        self.prep_job_folder(astr, astr_label)
        # start the lammps calculation
        relax_path = self.main_path + '/calcs/' + str(astr_label) + '/relax'

        # os.chdir(relax_path)
        lammps_exec = self.energy_exec_cmd.split()
        with open(
            relax_path + '/log_lammps.{}'.format(astr_label), 'w') as log_file:
            lammps_job = sp.Popen(
                lammps_exec, stdout=sp.PIPE, stderr=sp.STDOUT, cwd=relax_path)
            for each_line in lammps_job.stdout:
                line = each_line.decode('utf-8')
                log_file.write(line)
        # wait for the calculation to finish
        lammps_job.wait()

        # save total energy to model attributes
        total_energy = None
        match = None
        pattern = re.compile("Energy initial, next-to-last, final")
        lines = open(f'{relax_path}/log_lammps.{astr_label}',
                        'r').read().splitlines()
        for line in lines:
            if match is not None:
                total_energy = float(line.split()[2])
            match = re.search(pattern, line)

        if not total_energy:
            print('Model {} energy not found in log_lammps.{} file'.format(
                astr_label, astr_label))
            print('LAMMPS relaxation on model {} NOT successful'.format(
                astr_label))
            # quit()

        symbols = []
        for i in astr.species:
            if i.symbol not in symbols:
                symbols.append(i.symbol)
        relaxed_astr = lammps_code.get_relaxed_cell(
            f'{relax_path}/rlx.str', f'{relax_path}/in.data', symbols)
        # save relaxed structure in model.astr and to poscar
        POSCAR_relaxed = relax_path + '/POSCAR_relaxed'
        relaxed_astr.sort()
        lammps_code.move_atoms_inside(relaxed_astr)
        relaxed_astr.to(filename=POSCAR_relaxed, fmt='poscar')
        
        comp_dict = relaxed_astr.composition.as_dict()
        astr_elems = [i.name for i in relaxed_astr.composition.elements]

        # DU
        free_en = total_energy
        for elem in astr_elems:
            if elem in self.sym_mu_dict.keys():
                free_en -= comp_dict[elem]*self.sym_mu_dict[elem]
            else:
                print("Error. LAMMPS species " + elem +
                        " not contained in input yaml file.")
                
        # Following are done in relax:
        # save relaxed_structure - done in do_relaxation
        # relaxed_structure is now the structure of the model
        # save other attributes of the model after relaxation
        # (energy, gamma etc)
        # checks if relaxation is successful; gives error message and do not go
        # ahead with the structure (goes back and creates new strucutre)

        return free_en

    @staticmethod
    def get_relaxed_cell(rlx_astr, data_in_path, element_symbols, last_only=True):
        """
        (written by Benjamin Revard)

        Parses the relaxed cell from the rlx.str file.

        Arguments:

            rlx_astr (str): the path (as a string) to the rlx.str file

            data_in_path (str): the path (as a string) to the in.data file

            element_symbols (tuple): a tuple containing the set of chemical
             symbols of all the elements in the compositions space

        Returns:

            Structure: relaxed cell as a pymatgen `Structure` object
        """

        # read the atom types and corresponding atomic masses from in.data
        with open(data_in_path, 'r') as data_in:
            lines = data_in.readlines()
        types_masses = {}
        for i in range(len(lines)):
            if 'Masses' in lines[i]:
                for j in range(len(element_symbols)):
                    types_masses[int(lines[i + j + 2].split()[0])] = float(
                        lines[i + j + 2].split()[1])

        # map the atom types to chemical symbols
        types_symbols = {}
        for symbol in element_symbols:
            for atom_type in types_masses:
                # round the atomic masses to one decimal point for comparison
                if format(float(Element(symbol).atomic_mass), '.1f') == format(
                        types_masses[atom_type], '.1f'):
                    types_symbols[atom_type] = symbol


        def get_relaxed_lattice_and_cart_coords(lines):
            # get the lattice vectors
            a_data = lines[5].split()
            b_data = lines[6].split()
            c_data = lines[7].split()

            # default assume tilt is 0
            xy, xz, yz = 0, 0, 0
            # parse the tilt factors if thye exist
            if len(a_data) > 2:
                xy = float(a_data[2])
            if len(b_data) > 2:
                xz = float(b_data[2])
            if len(c_data) > 2:
                yz = float(c_data[2])

            # parse the bounds
            xlo_bound = float(a_data[0])
            xhi_bound = float(a_data[1])
            ylo_bound = float(b_data[0])
            yhi_bound = float(b_data[1])
            zlo_bound = float(c_data[0])
            zhi_bound = float(c_data[1])

            # compute xlo, xhi, ylo, yhi, zlo and zhi according to the conversion
            # given by LAMMPS
            # http://lammps.sandia.gov/doc/Section_howto.html#howto-12
            xlo = xlo_bound - min([0.0, xy, xz, xy + xz])
            xhi = xhi_bound - max([0.0, xy, xz, xy + xz])
            ylo = ylo_bound - min(0.0, yz)
            yhi = yhi_bound - max([0.0, yz])
            zlo = zlo_bound
            zhi = zhi_bound

            # construct a Lattice object from the lo's and hi's and tilts
            a = [xhi - xlo, 0.0, 0.0]
            b = [xy, yhi - ylo, 0.0]
            c = [xz, yz, zhi - zlo]
            relaxed_lattice = Lattice([a, b, c])

            # get the number of atoms
            num_atoms = int(lines[3])

            # get the atom types and their Cartesian coordinates
            types = []
            relaxed_cart_coords = []
            for i in range(num_atoms):
                atom_info = lines[9 + i].split()
                types.append(int(atom_info[1]))
                relaxed_cart_coords.append([float(atom_info[2]) - xlo,
                                            float(atom_info[3]) - ylo,
                                            float(atom_info[4]) - zlo])
                
            return relaxed_lattice, relaxed_cart_coords, types

        # read the dump.atom file as a list of strings
        with open(rlx_astr) as f:
            dat_lines = f.readlines()
        
        dat_lines.reverse()
        last_traj_lines = []
        for l in dat_lines:
            last_traj_lines.append(l)
            if "ITEM: TIMESTEP" in l:
                break
        last_traj_lines.reverse()
        dat_lines.reverse()
        relaxed_lattice, relaxed_cart_coords, types = \
                    get_relaxed_lattice_and_cart_coords(last_traj_lines)

        # make a list of chemical symbols (one for each site)
        relaxed_symbols = []
        for atom_type in types:
            relaxed_symbols.append(types_symbols[atom_type])

        last_astr = Structure(relaxed_lattice, relaxed_symbols, 
                            relaxed_cart_coords, coords_are_cartesian=True)
        
        all_traj_astrs, traj_inds = [last_astr], [int(last_traj_lines[1])]
        
        if last_only:
            return all_traj_astrs, None
                
        # get all other intermediate steps
        nl_per_traj = len(last_traj_lines)
        total_trajs = int(len(dat_lines)/nl_per_traj)
        for ii in range(total_trajs-1):
            traj_lines = dat_lines[nl_per_traj*ii:nl_per_traj*(ii+1)]
            traj_ind = int(traj_lines[1])
            relaxed_lattice, relaxed_cart_coords, _ = \
                    get_relaxed_lattice_and_cart_coords(traj_lines)

            traj_astr = Structure(relaxed_lattice, relaxed_symbols, 
                            relaxed_cart_coords, coords_are_cartesian=True)
            all_traj_astrs.append(traj_astr)
            traj_inds.append(traj_ind)

        return all_traj_astrs, traj_inds

    @staticmethod
    def move_atoms_inside(astr):
        """
        For a given structure object, move all sites within the unit cell.
        Eg: [-0.1, 0.4, 1.2] --> [0.9, 0.4, 0.2]

        Arguments:

            astr (obj): pymatgen `Structure` object
        """
        species = astr.species
        fc = astr.frac_coords
        fc = np.where((fc < 0) | (fc > 1), fc - np.floor(fc), fc)

        # replace all the coords in astr
        all_inds = [i for i in range(len(species))]
        astr.remove_sites(all_inds)
        for sps, coords in zip(species, fc):
            astr.append(sps, coords, coords_are_cartesian=False)

    @staticmethod
    def get_step_energies(log_lammps_path, step_inds=None):
        """
        Given a list of integers that are step indices in a lammps relaxation,
        this method reads the log.lammps file and returns a list of 
        total energies corresponding to the step indices.

        Arguments:
        
        log_lammps (str): path to the log.lammps file

        step_inds (list): list of integers or strings of the step indices
        """
        with open(log_lammps_path) as f:
            log_lines = f.readlines()

        log_step_lines = []
        for i, l in enumerate(log_lines):
            if 'Step TotEng Volume' in l:
                ii =  i
            if 'Loop time of' in l:
                jj = i
                step_lines = log_lines[ii+1:jj]
                log_step_lines += step_lines
        
        all_steps, all_tot_ens = [], []
        for step_line in log_step_lines:
            assert len(log_lines[ii].split()) == len(step_line.split())
            all_steps.append(step_line.split()[0])
            all_tot_ens.append(float(step_line.split()[1]))

        if step_inds is None:
            return [all_tot_ens[-1]]

        step_tot_ens = []
        for ind in step_inds:
            step_tot_ens.append(all_tot_ens[all_steps.index(str(ind))])

        return step_tot_ens

    def rename_dirs(self, x_keys, valid_x_keys, new_x_keys):
        """
        """
        calcs_path = self.main_path + '/calcs'
        # Rename failed dirs to create empty space (to avoid overwriting)
        for x_key in x_keys:
            if x_key not in valid_x_keys:
                os.rename(calcs_path + f'/{x_key}', calcs_path + \
                    f'/failed_{x_key}_{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}')
        # Now, rename valid dirs to new key names
        for vx_key, nx_key in zip(valid_x_keys, new_x_keys):
            os.rename(calcs_path + f'/{vx_key}', calcs_path + f'/{nx_key}')



class vasp_code(object):

    def __init__(self, energy_params):
        """
        Takes as input `energy_params`, the `dictionary` of all the
        parameters, taken from the input yaml file.
        Eg:
        ```python
        {'main_path': <path to direcctory in which fantastx is ran>,
        'shape': 'gb',
        'energy_files_path': <path_to_input_files>,
        'energy_exec_cmd': 'srun <path_to_vasp_binary>',
        'sym_mu_dict': {'Al': -3.35958515625, 'O': -6.76069604253},
        'atom_style': 'charge'}
        ```
        """
        self.main_path = energy_params['main_path']
        self.shape = energy_params['shape']
        # path to make new folder to do calculations
        self.relax_path = None
        # path to copy input files for each calculation
        self.energy_files_path = energy_params['files_path']
        # vasp execution command as a string
        # Ex: 'mpirun <path_to_vasp_binary>'
        self.energy_exec_cmd = energy_params['energy_exec_cmd']
        # how many times to resubmit job if not converged
        self.resubmit = 0
        if 'resubmit' in energy_params:
            self.resubmit = energy_params['resubmit']

        #  All these parameters for use in sd_flags
        # These are stored in inputs after object creation
        self.hollow_botz = None
        self.hollow_topz = None
        self.substrate_thickness = None
        self.sd_cut_off = None
        self.sd_no_z = None

        # This will be used to make potcars
        all_pots = [i for i in os.listdir(self.energy_files_path) if
                    i.startswith('POTCAR')]
        all_pots = [self.energy_files_path + '/' + i for i in all_pots]
        pdict = {}
        for a_pot in all_pots:
            with open(a_pot) as f:
                lines = f.readlines()
                for line in lines:
                    # assuming only PBE TODO: LDA and others
                    if 'TITEL' in line:
                        x = line.split('PBE')[1].split()[0]
                        if '_' in x:
                            x = x.split('_')[0]
                        pdict[x] = a_pot
        self.pot_dict = pdict

        # DU
        # Save species names and their chemical potentials for identification
        self.sym_mu_dict = {}
        for key, value in energy_params['element_syms'].items():
            self.sym_mu_dict[value] = energy_params['mu'][key]

        # default parameters for INCAR (only if necessary)
        # Or directly use the input files the user provided.

    def prep_job_folder(self, model, reg_id):
        """
        Function to:
        - check the provided input files (if any)
        - copy the input files to the model calc directory (relax_path)
            - The input files for vasp: INCAR, KPOINTS, POTCAR & POSCAR from
             model
        Arguments:
            model (obj): structure_record.model() object for which energy
             evaluation will be done
            reg_id (obj): structure_record.register_id() object for bookkeeping
        """
        main_path = self.main_path
        # create folders for model and relaxation
        # ProcessPoolExecutor has multiple parallel processes make models
        # at same time. So, reg_id is not up to date always.
        while True:
            if not model.inheritance == 'from_file':
                try:
                    model_path = main_path + '/calcs/' + str(model.label)
                    os.mkdir(model_path)
                except FileExistsError:
                    model.label = reg_id.create_id()
                    continue
                else:
                    break
            else:
                model_path = main_path + '/calcs/' + str(model.label)
                os.mkdir(model_path)
                break

        relax_path = main_path + '/calcs/' + str(model.label) + '/relax'
        os.mkdir(relax_path)
        self.relax_path = relax_path
        model.relax_path = relax_path
        astr = model.astr
        files_path = self.energy_files_path

        # sort the structure according to electronegativities
        astr.sort()
        # get the sorted species in the structure
        sorted_elems = astr.composition.elements
        sorted_syms = [i.name for i in sorted_elems]
        # get potcar by concatenating the potcars in the same order
        all_lines = []
        for sps in sorted_syms:
            with open(self.pot_dict[sps]) as p:
                lines = p.readlines()
                all_lines = all_lines + lines

        # pot_path = self.relax_path + '/POTCAR'

        # write model structure to POSCAR and store it in /relax
        new_poscar = relax_path + '/POSCAR_unrelaxed'
        poscar = relax_path + '/POSCAR'
        potcar = relax_path + '/POTCAR'
        with open(potcar, 'w') as pot:
            pot.writelines(all_lines)

        if self.shape == 'cluster' or self.shape == 'bulk':
            model.astr.to(filename=new_poscar, fmt='poscar')

        if self.shape == "molecule":
            self.write_mol_poscar(model, new_poscar)

        if self.shape == 'gb':
            self.write_gb_poscar(model, new_poscar)

        if self.shape == 'surface':
            self.write_surface_poscar(model, new_poscar,
                                      sd_cut_off=self.sd_cut_off,
                                      sd_no_z=self.sd_no_z)
        # TODO: implement selective dynamics for cluster geometry

        shutil.copy(new_poscar, poscar)
        # copy INCAR, KPOINTS to the relax path. Modify the INCAR if the model is a molecule
        shutil.copy(files_path + '/INCAR', relax_path + '/INCAR')
        if self.shape == "molecule":
            if model.astr.charge != 0:
                z_val_dict = {}
                # grab default number of electrons and modify it by the charge
                pattern = re.compile("ZVAL")
                pot_i = 0
                for line in all_lines:
                    match = re.search(pattern, line)
                    if match is not None:
                        z_val = float(line.split()[5])
                        z_val_dict[sorted_syms[pot_i]] = z_val
                        pot_i += 1
                        if pot_i == len(sorted_syms):
                            break

                total_electrons = 0
                poscar = Poscar.from_file(poscar)
                for sym in sorted_syms:
                    num_atoms = poscar.structure.composition.as_dict()[sym]
                    total_electrons += num_atoms * z_val_dict[sym]

                # number of electrons increases with negative charge
                total_electrons -= model.astr.charge

                incar_file = open(relax_path + '/INCAR', 'a')
                incar_file.write(
                    "\nNELECT = " + str(int(total_electrons)) + "\n")
                incar_file.close()

            incar_file = open(relax_path + '/INCAR', 'a')
            # currently hard-coding in changes to MAGMOM
            iron_sites = [i for i in range(model.astr.num_sites)
                          if model.astr.sites[i].specie.symbol == "Fe"]
            iron_magmoms = ["5.0" if model.astr.sites[i].specie.oxi_state >
                            2.1 else "4.0" for i in iron_sites]
            if iron_sites[0] == 0:
                pre_iron_str = ""
            else:
                pre_iron_str = str(iron_sites[0]) + "*0.6 "
            iron_str = " ".join(iron_magmoms) + " "
            post_iron_str = str(model.astr.num_sites - iron_sites[-1] - 1)
            post_iron_str += "*0.6\n"
            incar_file.write(
                "\nMAGMOM = " + pre_iron_str + iron_str + post_iron_str
            )
            print("\nMAGMOM = " + pre_iron_str + iron_str + post_iron_str)
            incar_file.close()

        shutil.copy(files_path + '/KPOINTS', relax_path + '/KPOINTS')

        print('Job prep finished. Submitting...')

    def relax(self, model, reg_id):
        """
        Starts the VASP relaxation in the calcs/<model label> path. Assigns
        the evaluated total energy and obj0_val to model attributes.
        Does not return anything
        Arguments:
            model (obj): `structure_record.model()` object for which energy
             evaluation will be done
            reg_id (obj): `structure_record.register_id()` object for
             bookkeeping
        """
        # prepare the folder to start energy calc
        self.prep_job_folder(model, reg_id)
        # start the vasp calculation
        if DEBUG:
            en_mod = np.random.uniform(7, 13)
            model.tot_en = -40 + en_mod
            model.obj0_val = en_mod
            model.converged = True
        else:
            self.run_vasp(model)

    def re_relax(self, model):
        """
        !!! Deprecated
            Checks if converged, resubmits if resubmit > 0
            save output files fo previous run with _resubmited_number
        """
        if not model.converged and self.resubmit != 0:
            relax_path = self.main_path + '/calcs/' + \
                str(model.label) + '/relax'
            os.chdir(relax_path)
            shutil.copy('OUTCAR', 'OUTCAR_{}'.format(self.resubmit-1))
            shutil.copy('CONTCAR', 'CONTCAR_{}'.format(self.resubmit-1))
            shutil.copy('OSZICAR', 'OSZICAR_{}'.format(self.resubmit-1))
            shutil.copy('POSCAR', 'POSCAR_{}'.format(self.resubmit-1))
            shutil.copy('CONTCAR', 'POSCAR')
            self.run_vasp(model)
        self.resubmit = self.resubmit - 1

    def run_vasp(self, model):
        """
        Runs vasp in the job directory (relax_path), checks if converged
        and resubmits if necessary. Saves energy and objective function
        value to model object.
        Arguments:
            model (obj): `structure_record.model()` object for which energy
             evaluation will be done
        """
        vasp_exec = self.energy_exec_cmd.split()
        log_file = open(model.relax_path + '/job.log', 'w')
        err_file = open(model.relax_path + '/job.err', 'w')
        sp.call(vasp_exec, stdout=log_file,
                stderr=err_file, cwd=model.relax_path)
        # sp.call will wait for the calculation to finish
        log_file.close()
        err_file.close()

        # TODO: get energy
        # check if calculation is converged
        converged = False
        outcar = model.relax_path + '/OUTCAR'
        with open(outcar) as out:
            lines = out.readlines()
            for line in lines:
                if 'reached required accuracy' in line:
                    converged = True
                    break
        if not converged:
            print('Energy calculation of model {} not'
                  ' converged'.format(model.label))

        # if converged, get energy
        if converged:
            model.converged = converged

            # get total energy from output files
            oszicar = model.relax_path + '/OSZICAR'
            with open(oszicar) as oz:
                lines = oz.readlines()
            if lines[-1].split()[3] == 'E0=':
                total_energy = float(lines[-1].split()[4])
                model.tot_en = total_energy

            # get relaxed structure and oxidize it if original was oxidized
            try:
                contcar = model.relax_path + '/CONTCAR'
                shutil.copy(contcar, model.relax_path + '/POSCAR_relaxed')
                relaxed_astr = Structure.from_file(contcar)

                oxi_states = []
                oxi_states_exist = False
                for site in model.astr.sites:
                    if hasattr(site.specie, 'oxi_state'):
                        oxi_states.append(site.specie.oxi_state)
                        oxi_states_exist = True
                    else:
                        oxi_states.append(0)
                if oxi_states_exist:
                    relaxed_astr.add_oxidation_state_by_site(oxi_states)
                relaxed_astr.sort()
                self.move_atoms_inside(relaxed_astr)
                model.astr = relaxed_astr
            except:
                print('Relaxed structure not available in CONTCAR')

            # evaluate objective function and save as model attribute
            comp_dict = relaxed_astr.composition.as_dict()
            astr_elems = [i.name for i in
                          relaxed_astr.composition.elements]

            # DU
            # Evaluate free energy by calculating
            # chemical potential contribution
            if self.shape == "molecule":
                model.obj0_val = total_energy/model.astr.num_sites
            else:
                free_en = total_energy
                for elem in astr_elems:
                    if elem in self.sym_mu_dict.keys():
                        free_en -= comp_dict[elem]*self.sym_mu_dict[elem]
                    else:
                        print("Error. VASP species " + elem +
                              " not contained in input yaml file.")
                model.obj0_val = float(free_en)

    def move_atoms_inside(self, astr):
        """
        For a given structure object, move all sites within the unit cell.
        Eg: [-0.1, 0.4, 1.2] --> [0.9, 0.4, 0.2]
        Arguments:
            astr (obj): pymatgen `Structure` object
        """
        species = astr.species
        fc = astr.frac_coords
        fc = np.where((fc < 0) | (fc > 1), fc - np.floor(fc), fc)

        # replace all the coords in astr
        all_inds = [i for i in range(len(species))]
        astr.remove_sites(all_inds)
        for sps, coords in zip(species, fc):
            astr.append(sps, coords, coords_are_cartesian=False)

    def write_mol_poscar(self, model, file_name):
        """
        For a newly created molecule model, set sd_flags for the central
        fragment to be [F, F, F], set all other sd_flags to be [T, T, T].

        Arguments:
            model (obj): `structure_record.model()` object for which energy
             evaluation will be done
            file_name (str): the file name of the structure to be written
             as poscar. 
        """
        mol = model.molecule_representation
        if 'fixed_atoms' in mol.keys():
            sd_flags = [[False, False, False] if i in mol['fixed_atoms']
                        else [True, True, True] for i
                        in range(model.astr.num_sites)]
        else:
            sd_flags = [[True, True, True]
                        for i in range(model.astr.num_sites)]

        model.astr.add_site_property("selective_dynamics", sd_flags)
        mol_poscar = Poscar(model.astr)
        mol_poscar.write_file(file_name)

    def write_gb_poscar(self, model, file_name):
        """
        For a newly created model, set sd_flags to each site according to its
        z-coordinate. For 'gb' gemoetry, all interface region atoms would have
        [T,T,T] and others would have [F,F,F]. Then writes the POSCAR file in
        relax_path.
        Arguments:
            model (obj): `structure_record.model()` object for which energy
             evaluation will be done
            file_name (str): the file name of the structure to be written as
             POSCAR
        """
        frac_zmin, frac_zmax = self.hollow_botz, self.hollow_topz
        if model.sd_true_above is not None:
            frac_zmin = model.sd_true_above  # smaller z-coordinate
        if model.sd_true_below is not None:
            frac_zmax = model.sd_true_below  # larger z-coordinate

        frac_zs = model.astr.frac_coords[:, 2]
        bs = []
        for z in frac_zs:
            b = 0
            if frac_zmin < z < frac_zmax:
                b = 1
            bs.append(b)
        sd_flags = [[bool(i), bool(i), bool(i)] for i in bs]
        gb_poscar = Poscar(model.astr, selective_dynamics=sd_flags)
        gb_poscar.write_file(file_name)

    def write_surface_poscar(self, model, file_name, sd_cut_off=None,
                             sd_no_z=False):
        """
        For a newly created model in `'surface'` geometry, set `sd_flags`
        to each site according to its z-coordinate. Assigns [T, T, T]
        to atoms above `sd_cut_ff` if provided, else uses the substrate
        thickness as `sd_cut_off`. Sets [T, T, F] for atoms if `sd_no_z`
        is `True`.
        Arguments:
            model (obj): structure_record.model() object for which energy
             evaluation will be done
            file_name (str): the file name of the structure to be written
             as POSCAR
            sd_cut_off (float): the cut off distance from bottom of the slab.
             The atoms below it will be frozen. Default is substrate
             thickness.
            sd_no_z (bool): set to True to allow the atoms to relax
             in z-direction
        """
        if not sd_cut_off:  # automatically freeze substrate
            sd_cut_off = self.substrate_thickness

        slab_sites = model.astr.sites
        bot_z_cart = model.astr.cart_coords[:, 2].min()
        surface_inds = [i for i, site in enumerate(slab_sites) if
                        site.coords[2] - bot_z_cart > sd_cut_off]

        sd_flags = []
        for i in range(len(slab_sites)):
            sd_flag = [0, 0, 0]
            if i in surface_inds:
                if sd_no_z is True:
                    sd_flag = [1, 1, 0]
                else:
                    sd_flag = [1, 1, 1]
            sd_flags.append(sd_flag)
        sd_flags = [[bool(flag[0]), bool(flag[1]), bool(flag[2])]
                    for flag in sd_flags]

        gb_poscar = Poscar(model.astr, selective_dynamics=sd_flags)
        gb_poscar.write_file(file_name)

    def rename_dirs(self, x_keys, valid_x_keys, new_x_keys):
        """
        """
        calcs_path = self.main_path + '/calcs'
        # Rename failed dirs to create empty space (to avoid overwriting)
        for x_key in x_keys:
            if x_key not in valid_x_keys:
                os.rename(calcs_path + f'/{x_key}', calcs_path + \
                    f'/failed_{x_key}_{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}')
        # Now, rename valid dirs to new key names
        for vx_key, nx_key in zip(valid_x_keys, new_x_keys):
            os.rename(calcs_path + f'/{vx_key}', calcs_path + f'/{nx_key}')



def get_energy_params(i_dict):
    """
    Determines all the parameters, mandatory and optional, to be
    used to make energy object for each calculation.

    Arguments:

        i_dict (dict): dictionary of all the user-provided input parameters
        read from yaml file

    Returns:

        dict: all the determined parameters
    """
    energy_params = {}
    # add main_path, i.e., where the search started to energy_params
    energy_params['main_path'] = i_dict['main_path']

    # energy_code
    if 'energy_code' not in i_dict:
        print('Please provide energy code details. This is mandatory')
    else:
        energy_params['energy_code'] = i_dict['energy_code']

    # energy code execution command (Mandatory)
    if 'energy_exec_cmd' not in i_dict:
        print('Please provide the execution command for the energy '
            'code. This is mandatory. Ex: \"lmp_mpi -in in.min\"')
    else:
        energy_params['energy_exec_cmd'] = i_dict['energy_exec_cmd']

    # DU
    # chemical potentials
    species_dict = i_dict['species']
    mu = {}
    for species in species_dict.keys():
        index = int(species[7:])
        if 'mu' in species_dict[species].keys():
            mu[index] = species_dict[species]['mu']
        else:
            print('Error encountered with species ' + str(index) + ': '
                'Chemical potentials must be provided in the dictionary'
                + ' for each species!')
            mu[index] = 0

    energy_params['mu'] = mu

    # Number of times to resubmit if not converged (for vasp)
    energy_params['resubmit'] = 0
    if energy_params['energy_code'] == 'vasp':
        if 'resubmit' in i_dict:
            energy_params['resubmit'] = i_dict['resubmit']

    # any specific energy code parameters like atom_style etc
    if 'energy_code_params' in i_dict:
        energy_code_params = i_dict['energy_code_params']
        energy_code_keys = list(energy_code_params.keys())
        for key in energy_code_keys:
            energy_params[key] = energy_code_params[key]

    # files_path
    if 'energy_files_path' not in i_dict:
        print('Please provide path to folder with input energy files for'
            ' relaxation.')
    else:
        energy_params['files_path'] = i_dict['energy_files_path']

    energy_params['shape'] = i_dict['shape']
    energy_params['element_syms'] = i_dict['element_syms']

    return energy_params

def make_energy_code_object(yaml_file, main_path):
    """
    """
    main_path = os.getcwd()
    # read input file and make input dictionary
    with open(yaml_file) as ifile:
        i_dict = yaml.load(ifile, Loader=yaml.FullLoader)
        i_dict['main_path'] = main_path

    # make energy_code object
    energy_params = get_energy_params(i_dict)

    energy_pkg = i_dict['energy_code']  # 'vasp' or 'lammps'
    if energy_pkg == 'vasp':
        energy_code = vasp_code(energy_params)
    elif energy_pkg == 'lammps':
        energy_code = lammps_code(energy_params)
    else:
        print('Please set energy_code in inputs as one of vasp'
            'or lammps only')
    print(f"Energy code: {energy_pkg}")
    

    if energy_params['shape'] == 'gb' and 'shape_params' in i_dict:
        energy_code.hollow_botz = i_dict['shape_params']['hollow_botz']
        energy_code.hollow_topz = i_dict['shape_params']['hollow_topz']
        return energy_code

    if energy_params['shape'] == 'surface' and 'shape_params' in i_dict:
        energy_code.substrate_thickness = i_dict['shape_params']['substrate_thickness']
        energy_code.sd_cut_off = i_dict['shape_params']['sd_cut_off']
        energy_code.sd_no_z = i_dict['shape_params']['sd_no_z']
        return energy_code
    
    return energy_code

