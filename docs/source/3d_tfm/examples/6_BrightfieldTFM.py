"""
Brightfield TFM & Cropped Input
==============================================

This example calculates traction forces around an individual immune cell (NK92 natural killer cell)
during migrtion in a collagen 1.2mg/ml networks (Batch C). 

Here simple brightfield image stacks are used to calculate 
the matrix deformations & forces around the cell.

We compare the contracted state with the previous time step 30 seconds earlier.
The cell of interest is cropped from a larger field of view. 

This example can also be evaluated within the graphical user interface.


 .. figure:: ../images/examples/brightfield_tfm/2d.gif
    :scale: 100%
    :align: center
    

"""

import saenopy


# sphinx_gallery_thumbnail_path = '../../saenopy/img/thumbnails/BFTFM_2.png'


# %%
# Loading the Stacks
# ------------------
#
# Saenopy is very flexible in loading stacks from any filename structure.
# Here, we use only brightfield images without additional channels. 
# We compare the stack in the contracted state with a reference state 30 seconds earlier.
# We load one multitiff file per stack, where each individual image corresponds to the [z] position.
# Experimentally, it can be convenient to take a larger field of view and crop it later to an area of interest.  
# For this purpose, we use the "crop" parameter, which specifies the boundaries of region of intrest (in pixel). 
# The "crop" parameter can be set both from the user interface and from the Python code.
#
#  .. figure:: ../images/examples/brightfield_tfm/crop.png
#
#
# We load the relaxed and the contracted stack by using 
# the placeholder [z] for the z stack in mutlitiffs
results = saenopy.get_stacks( 
    'BrightfieldNK92Data/2023_02_14_12_0920_stack.tif[z]',
    reference_stack='BrightfieldNK92Data/2023_02_14_12_0850_stack.tif[z]',
    output_path='BrightfieldNK92Data/example_output',
    voxel_size=[0.15, 0.15, 2.0],
    crop={'x': (1590, 2390), 'y': (878, 1678), 'z': (30, 90)},
    )




# %%
# Detecting the Deformations
# --------------------------
# Saenopy uses 3D Particle Image Velocimetry (PIV) with the following parameters 
# to calculate matrix deformations between a deformed and relaxed state.  
#
# 
#
# +------------------+-------+
# | Piv Parameter    | Value |
# +==================+=======+
# | element_size     |  4.8  |
# +------------------+-------+
# | window_size      |  12.0 |
# +------------------+-------+
# | signal_to_noise  |   1.3 |
# +------------------+-------+
# | drift_correction | True  |
# +------------------+-------+
#
# Small image features enable to measure 3D deformations from the brightfield stack 
#
# .. figure:: ../images/examples/brightfield_tfm/bf_scroll.gif
#   :scale: 100%
#   :align: center
# 
#


# define the parameters for the piv deformation detection
piv_parameters = {'element_size': 4.8, 'window_size': 12.0, 'signal_to_noise': 1.3, 'drift_correction': True}


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
        # or two consecutive stacks
        if result.stack_reference is None:
            stack1, stack2 = result.stacks[i], result.stacks[i + 1]
        # get both stacks, either reference stack and one from the list
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
    

  
# %%
# Generating the Finite Element Mesh
# ----------------------------------
# We interpolate the found deformations onto a new mesh which will be used for the regularisation. 
# with the following parameters.
#
# +------------------+-------+
# | Mesh Parameter   | Value |
# +==================+=======+
# | element_size     |   4.0 |
# +------------------+-------+
# | mesh_size_same   | 'piv' |
# +------------------+-------+
# | reference_stack  | 'next'|
# +------------------+-------+
#

# define the parameters to generate the solver mesh and interpolate the piv mesh onto it
mesh_parameters = {'reference_stack': 'next', 'element_size': 4.0, 'mesh_size': 'piv'}

# iterate over all the results objects
for result in results:
    # correct for the reference state
    displacement_list = saenopy.subtract_reference_state(result.mesh_piv, mesh_parameters["reference_stack"])
    # set the parameters
    result.mesh_parameters = mesh_parameters
    # iterate over all stack pairs
    for i in range(len(result.mesh_piv)):
        # and create the interpolated solver mesh
        result.solvers[i] = saenopy.interpolate_mesh(result.mesh_piv[i], displacement_list[i], mesh_parameters)
    # save the meshes
    result.save()

# %%
# Calculating the Forces
# ----------------------
# Define the material model and run the regularisation to fit the measured deformations and get the forces. 
#
# +--------------------+---------+
# | Material Parameter | Value   |
# +====================+=========+
# | k                  |  6062   |
# +--------------------+---------+
# | d_0                | 0.0025  |
# +--------------------+---------+
# | lambda_s           |  0.0804 |
# +--------------------+---------+
# | d_s                | 0.034   |
# +--------------------+---------+
#
# +--------------------------+---------+
# | Regularisation Parameter | Value   |
# +==========================+=========+
# | alpha                    |  10**11 |
# +--------------------------+---------+
# | step_size                |    0.33 |
# +--------------------------+---------+
# | max_iterations           |     300 |
# +--------------------------+---------+
# | rel_conv_crit            |    0.01 |
# +--------------------------+---------+
#

# define the parameters to generate the solver mesh and interpolate the piv mesh onto it
material_parameters = {'k': 6062.0, 'd_0': 0.0025, 'lambda_s': 0.0804, 'd_s':  0.034}
solve_parameters = {'alpha': 10**11, 'step_size': 0.33, 'max_iterations': 300, 'rel_conv_crit': 0.01}

# iterate over all the results objects
for result in results:
    result.material_parameters = material_parameters
    result.solve_parameters = solve_parameters
    for index, M in enumerate(result.solvers):
        # set the material model
        M.set_material_model(saenopy.materials.SemiAffineFiberMaterial(
            material_parameters["k"],
            material_parameters["d_0"],
            material_parameters["lambda_s"],
            material_parameters["d_s"],
        ))
        # find the regularized force solution
        M.solve_regularized(alpha=solve_parameters["alpha"], step_size=solve_parameters["step_size"],
                            max_iterations=solve_parameters["max_iterations"],
                            rel_conv_crit=solve_parameters["rel_conv_crit"], verbose=True)
        # save the forces
        result.save()
        # clear the cache of the solver
        results.clear_cache(index)

    

# %%
# Display Results
# ----------------------
#
#  .. figure:: ../images/examples/brightfield_tfm/reconstruction.png
#   
# The reconstructed force field (right) generates a reconstructed deformation field (left)
# that recapitulates the measured matrix deformation field (upper video). The overall cell contractility is 
# calculated as all force components pointing to the force epicenter.
#
#  .. figure:: ../images/examples/brightfield_tfm/nans.png
#
# The cell occupied area is omitted since the signal to noise filter replaces the limited information with Nan values (Grey Dots).
# Therefore, no additional segmentation is required. Since we are working with simple brightfield images here, we 
# do not have information below and above the cell.
#