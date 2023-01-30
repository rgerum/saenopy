"""
Multicellular TFM (intestinal organoids)
=======================

This example evaluates contractile forces of an individual intestinal organoid embedded in 1.2mg/ml collagen imaged with confocal reflection microscopy. 
The relaxed stacks were recorded after Triton x-100 treatment. We evaluate the contraction between ~23h in collagen compared 
to the state after drug relaxation (indicated at the end of the video). This example can also be evaluated with the graphical user interface.

 .. figure:: ../images/Gif_organoid.gif
 

"""

import saenopy

# sphinx_gallery_thumbnail_path = '../../saenopy/img/thumbnails/StainedOrganoid_icon.png'

# %%
# Downloading the example data files
# ----------------------------------
# An individual intestinal organoid is recorded at one positions (Pos07) in collagen 1.2 mg/ml.  
# Images are recorded in confocal reflection channel (channels (ch00, ch00) after ~23h in collagen (t50) and after drug relaxation approx. 2 hours later (t6).
# The lower time index t6 is due of the start of a new image series and refers to the image AFTER relaxation. 
# The stack has 52z positions (z00-z51). T


# download the data
saenopy.loadExample("OrganoidTFM")

# %%
# Loading the Stacks
# ------------------
#
# Saenopy is very flexible in loading stacks from any filename structure.
# Here we replace the number in the position "Pos004" with an asterisk "Pos*" to batch process all positions.
# We replace the number of the channels "ch00" with a channel placeholder "ch{c:00}" to indicate that this refers to
# the channels and which channel to use as the first channel where the deformations should be detected.
# We replace the number of the z slice "z000" with a z placeholder "z{z}" to indicate that this number refers to the
# z slice. We do the same for the deformed state and for the reference stack.

# load the relaxed and the contracted stack
# {z} is the placeholder for the z stack
# {c} is the placeholder for the channels
# {t} is the placeholder for the time points
results = saenopy.get_stacks(
    '4_OrganoidTFM/Pos007_S001_t50_z{z}_ch00.tif',
    reference_stack='4_OrganoidTFM/Pos007_S001_t6_z{z}_ch00.tif',
    output_path='4_OrganoidTFM/example_output',
    voxel_size=[1.444, 1.444, 1.976])

# %%
# Detecting the Deformations
# --------------------------
# Saenopy uses 3D Particle Image Velocimetry (PIV) with the following parameters 
# to calculate matrix deformations between the deformed and relaxed state. 
# 
#
# +------------------+-------+
# | Piv Parameter    | Value |
# +==================+=======+
# | elementsize      |    30 |
# +------------------+-------+
# | win_mu           |    40 |
# +------------------+-------+
# | signoise_filter  |   1.3 |
# +------------------+-------+
# | drift_correction | True  |
# +------------------+-------+
#

# define the parameters for the piv deformation detection
params = {'elementsize': 30.0, 'win_um': 40.0, 'signoise_filter': 1.3, 'drift_correction': True}

# iterate over all the results objects
for result in results:
    # set the parameters
    result.piv_parameter = params
    # get count
    count = len(result.stack)
    if result.stack_reference is None:
        count -= 1
    # iterate over all stack pairs
    for i in range(count):
        # get both stacks, either reference stack and one from the list
        if result.stack_reference is None:
            stack1, stack2 = result.stack[i], result.stack[i + 1]
        # or two consecutive stacks
        else:
            stack1, stack2 = result.stack_reference, result.stack[i]
        # and calculate the displacement between them
        result.mesh_piv[i] = saenopy.get_displacements_from_stacks(stack1, stack2,
                                                                   params["win_um"],
                                                                   params["elementsize"],
                                                                   params["signoise_filter"],
                                                                   params["drift_correction"])
    # save the displacements
    result.save()

# %%
# Generating the Finite Element Mesh
# ----------------------------------
# Interpolate the found deformations onto a new mesh which will be used for the regularisation. 
# In this case, we are experimental limited (due to the size, strength and spatial expansion of the organoid and our objective) and can not image the
# complete matrix deformation field around the organoid. To obtain higher accuracy for a cropped deformation field, 
# we perform the force reconstruction in a larger volume and interpolate the data on a larger mesh as follows.
#
# +------------------+-------+
# | Mesh Parameter   | Value |
# +==================+=======+
# | element_size     |    30 |
# +------------------+-------+
# | mesh_size_same   | False |
# +------------------+-------+
# | mesh_size_x      |  900  |
# +------------------+-------+
# | mesh_size_y      |  900  |
# +------------------+-------+
# | mesh_size_z      |  900  |
# +------------------+-------+
# | reference_stack  |'first'|
# +------------------+-------+
#

# define the parameters to generate the solver mesh and interpolate the piv mesh onto it
params = {'reference_stack': 'first', 'element_size': 30, 'mesh_size_same': False, 'mesh_size_x': 900.0, 'mesh_size_y': 900.0, 'mesh_size_z': 900.0}

# iterate over all the results objects
for result in results:
    # correct for the reference state
    displacement_list = saenopy.substract_reference_state(result.mesh_piv, params["reference_stack"])
    # set the parameters
    result.interpolate_parameter = params
    # iterate over all stack pairs
    for i in range(len(result.mesh_piv)):
        # and create the interpolated solver mesh
        result.solver[i] = saenopy.interpolate_mesh(result.mesh_piv[i], displacement_list[i], params)
    # save the meshes
    result.save()

# %%
# Calculating the Forces
# ----------------------
# Define the material model and run the regularisation to fit the measured deformations and get the forces. Here we define a low convergence criterion
# and let the algorithm run for the maximal amount of steps that we define to 500. Afterwards, we can check the regularisation_results in the console or
# in the graphical user interface.
#
# +--------------------+---------+
# | Material Parameter | Value   |
# +====================+=========+
# | k                  |  6062   |
# +--------------------+---------+
# | d0                 | 0.0025  |
# +--------------------+---------+
# | lambda_s           |  0.0804 |
# +--------------------+---------+
# | ds                 | 0.034   |
# +--------------------+---------+
#
# +--------------------------+---------+
# | Regularisation Parameter | Value   |
# +==========================+=========+
# | alpha                    |  10**10 |
# +--------------------------+---------+
# | stepper                  |    0.33 |
# +--------------------------+---------+
# | i_max                    |    500  |
# +--------------------------+---------+
# | rel_conv_crit            |  3e-05  |
# +--------------------------+---------+
#

# define the parameters to generate the solver mesh and interpolate the piv mesh onto it
params = {'k': 6062.0, 'd0': 0.0025, 'lambda_s': 0.0804, 'ds':  0.034, 'alpha': 10**10, 'stepper': 0.33, 'i_max': 500, 'rel_conv_crit': 3e-05}

# iterate over all the results objects
for result in results:
    result.solve_parameter = params
    for M in result.solver:
        # set the material model
        M.setMaterialModel(saenopy.materials.SemiAffineFiberMaterial(
            params["k"],
            params["d0"],
            params["lambda_s"],
            params["ds"],
        ))
        # find the regularized force solution
        M.solve_regularized(stepper=params["stepper"], i_max=params["i_max"],rel_conv_crit=params["rel_conv_crit"], alpha=params["alpha"], verbose=True)
    # save the forces
    result.save()
    
    

# %%
# Display Results
# ----------------------
#
# ToDo
#   
# The reconstructed force field (right) generates a reconstructed deformation field (middle)
# that recapitulates the measured matrix deformation field (left). The overall cell contractility is 
# calculated as all forcecomponents pointing to the force epicenter.
#
  
 