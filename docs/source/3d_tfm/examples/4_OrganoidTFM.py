"""
Multicellular TFM (intestinal organoids)
==============================================

This example evaluates contractile forces of an individual intestinal organoid embedded in 1.2mg/ml collagen imaged with confocal reflection microscopy. 
The relaxed stacks were recorded after Triton x-100 treatment. We evaluate the contraction between ~23h in collagen compared 
to the state after drug relaxation (indicated at the end of the video). This example can also be evaluated with the graphical user interface.

 .. figure:: ../images/examples/organoid_tfm/organoid.gif
    :scale: 50%
    :align: center

"""

import saenopy

# sphinx_gallery_thumbnail_path = '../../saenopy/img/thumbnails/StainedOrganoid_icon.png'

# %%
# Downloading the example data files
# ----------------------------------
# An individual intestinal organoid is recorded at one positions (Pos07) in collagen 1.2 mg/ml.  
# Images are recorded in confocal reflection channel (ch00) after ~23h in collagen (t50) and after drug relaxation
# approx. 2 hours later (t6).
# The lower time index t6 is due of the start of a new image series and refers to the image AFTER relaxation. 
# The stack has 52 z positions (z00-z51). T
#
# ::
#
#    4_OrganoidTFM
#    ├── Pos007_S001_t50_z00_ch00.tif
#    ├── Pos007_S001_t50_z01_ch00.tif
#    ├── Pos007_S001_t50_z02_ch00.tif
#    ├── ...
#    ├── Pos007_S001_t6_z00_ch00.tif
#    ├── Pos007_S001_t6_z01_ch00.tif
#    ├── Pos007_S001_t6_z02_ch00.tif
#    └── ...
#


# download the data
saenopy.load_example("OrganoidTFM")

# %%
# Loading the Stacks
# ------------------
#
# Saenopy is very flexible in loading stacks from any filename structure.
# Here we do not have multiple positions, so we do not need to use and asterisk * for batch processing.
# We do not have multiple channels, so we do not need a channel placeholder.
# We replace the number of the z slice "z00" with a z placeholder "z{z}" to indicate that this number refers to the
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
# | element_size     |    30 |
# +------------------+-------+
# | window_size      |    40 |
# +------------------+-------+
# | signal_to_noise  |   1.3 |
# +------------------+-------+
# | drift_correction | True  |
# +------------------+-------+
#

# define the parameters for the piv deformation detection
piv_parameters = {'element_size': 30.0, 'window_size': 40.0, 'signal_to_noise': 1.3, 'drift_correction': True}


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

# %%
# Generating the Finite Element Mesh
# ----------------------------------
# Interpolate the found deformations onto a new mesh which will be used for the regularisation. 
# In this case, we are experimental limited (due to the size, strength and spatial expansion of the organoid and our objective) and can not image the
# complete matrix deformation field around the organoid. To obtain higher accuracy for a cropped deformation field, 
# we perform the force reconstruction in a volume with increased z-height. 
#
# +------------------+------------------+
# | Mesh Parameter   | Value            |
# +==================+==================+
# | element_size     |    30            |
# +------------------+------------------+
# | mesh_size        |  (738, 738, 738) |
# +------------------+------------------+
# | reference_stack  | 'first'          |
# +------------------+------------------+
#

# define the parameters to generate the solver mesh and interpolate the piv mesh onto it
mesh_parameters = {'reference_stack': 'first', 'element_size': 30, 'mesh_size': (738, 738, 738)}

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
# Define the material model and run the regularisation to fit the measured deformations and get the forces. Here we define a low convergence criterion
# and let the algorithm run for the maximal amount of steps that we define to 1400. Afterwards, we can check the regularisation_results in the console or
# in the graphical user interface.
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
# | alpha                    |  10**10 |
# +--------------------------+---------+
# | step_size                |    0.33 |
# +--------------------------+---------+
# | max_iterations           |   1400  |
# +--------------------------+---------+
# | rel_conv_crit            |  1e-7   |
# +--------------------------+---------+
#

# define the parameters to generate the solver mesh and interpolate the piv mesh onto it
material_parameters = {'k': 6062.0, 'd_0': 0.0025, 'lambda_s': 0.0804, 'd_s':  0.034}
solve_parameters = {'alpha': 10**10, 'step_size': 0.33, 'max_iterations': 1400, 'rel_conv_crit': 1e-7}

# iterate over all the results objects
for result in results:
    result.material_parameters = material_parameters
    result.solve_parameters = solve_parameters
    for M in result.solvers:
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
  
 