"""
Microbeads to Detect Matrix Deformations
==============================================

This example calculates matrix deformations around a Human Umbilical Vein Endothelial Cells (HUVECs) that invades 
in a polyethylene glycol (PEG) hydrogel cell as measured and described in `[Jorge Barrasa-Fano et al., 2021] <https://www.sciencedirect.com/science/article/pii/S2352711021000625>`_.                                                
The data is available from the authors website `here <https://www.mech.kuleuven.be/en/bme/research/mechbio/lab/tfmlab>`_.
Images of microbeads (channel 'ch01') between a relaxed and deformed stacks are used to calculate matrix deformations (image on the right).


 .. figure:: ../images/Microbeads.png


"""

import saenopy


# sphinx_gallery_thumbnail_path = '../../saenopy/img/thumbnails/Bead_example_icon.png'


# %%
# Loading the Stacks
# ------------------
#
# Saenopy is very flexible in loading stacks from any filename structure.
# We replace the number of the channels "ch00" with a channel placeholder "ch{c:01}" to indicate that this refers to
# the channels and which channel to use as the first channel where the deformations should be detected.
# We replace the number of the z slice "z000" with a z placeholder "z{z}" to indicate that this number refers to the
# z slice. We do the same for the deformed state and for the reference stack.
# Here images of microbeads (channel 'ch01') are used to calculate matrix deformation.

# load the relaxed and the contracted stack
# {z} is the placeholder for the z stack
# {c} is the placeholder for the channels
# {t} is the placeholder for the time points
results = saenopy.get_stacks( 
    'TestDataTFMlabKULeuven\Stressed_z{z}_ch{c:01}.tif',
    reference_stack='TestDataTFMlabKULeuven\Relaxed_z{z}_ch{c:01}.tif',
    output_path='3_BeadMeasurement/example_output',
    voxel_size=[0.567, 0.567, 0.493])

# %%
# Detecting the Deformations
# --------------------------
# Saenopy uses 3D Particle Image Velocimetry (PIV) with the following parameters 
# to calculate matrix deformations between a deformed and relaxed state.  
#
# +------------------+-------+
# | Piv Parameter    | Value |
# +==================+=======+
# | element_size     |    5  |
# +------------------+-------+
# | window_size      |    25 |
# +------------------+-------+
# | signal_to_noise  |   1.1 |
# +------------------+-------+
# | drift_correction | True  |
# +------------------+-------+
#

# define the parameters for the piv deformation detection
piv_parameters = {'element_size': 5.0, 'window_size': 25.0, 'signal_to_noise': 1.1, 'drift_correction': True}


# iterate over all the results objects
for result in results:
    # set the parameters
    result.piv_parameters = piv_parameters
    # get count
    count = len(result.stacks)
    if result.stack_reference is None:
        count -= 1
    # iterate over all stack pairs
    for i in range(count):
        # get two consecutive stacks 
        if result.stack_reference is None:
            stack1, stack2 = result.stacks[i], result.stacks[i + 1]
        # or reference stack and one from the list
        else:
            stack1, stack2 = result.stack_reference, result.stacks[i]
        # and calculate the displacement between them
        result.mesh_piv[i] = saenopy.get_displacements_from_stacks(stack1, stack2,
                                                                   piv_parameters["window_size"],
                                                                   piv_parameters["element_size"],
                                                                   piv_parameters["signal_to_noise"],
                                                                   piv_parameters["drift_correction"])
    # save the displacements
    result.save()
