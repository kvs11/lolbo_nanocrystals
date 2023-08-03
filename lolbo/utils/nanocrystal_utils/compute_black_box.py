"""
Includes functions related to calculation of DFT/LAMMPS energies of new candidates. 

Methods are inspired from energy.py in FANTASTX
"""
import numpy as np

def get_y_val_from_input(input_x):
    """
    Takes the 164x3 input vector

    POSCAR and POTCAR
        Get abc and angles
        Get fractional coordinates
        Get species in the same order
        Use the species list to make POTCAR
        Create structure with abc, angles, coords and species 
        Make POSCAR

    KPOINTS
        Use an automatic mesh KPOINTS with (say) 30 kpoints per inv. å
        Copy from pre-existing path

    INCAR
        Use Pymatgen Incar() class
        Take an existing INCAR, and add necessary tags. 

    submit VASPjob (for debugging do a dummy job)

    >>>>>
    Question: This approach to be fully-automatic, 
    it should run like FANTASTX on dask workers; 
    And then call Lolbo-model updates using a 
    different set of dask-workers that use GPUs.

    The overall workflow should be something like this - 

        start FX-Lolbo-VAE on few cpu cores (master)
        Load VAE+BO with initial training data on cpu. 

        Loop this:
            Create a set of new candidates that needs "query_oracle"
            Initiate dask workers with CPUs (use GPUs if that is beneficial for overall workflow)
                run VASP and get y_value for the generated candidates
                send data to master (and close worker if it needs to stay idle for too long?)
            Initiate new dask worker with GPUs
                update BO surrogate model or total VAE+BO joint model
            Get new candidates on master
    <<<<<<
    


    """
    y_val = np.random.uniform(0.5, 2.5)
    
    do_lammps = False
    #if do_lammps:
    #    # 1. Get the POSCAR file 



    return y_val


def run_vasp():
    print ('Running VASP placeholder')


def get_y_vals_from_a_predictor():
    pass