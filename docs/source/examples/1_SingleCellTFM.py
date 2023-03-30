"""
Classic TFM (liver fibroblasts)
=======================

This example evaluates three hepatic stellate cells in 1.2mg/ml collagen with relaxed and deformed stacks. The relaxed stacks were recorded with cytochalasin D treatment of the cells.
This example can also be evaluated with the graphical user interface.

 .. figure:: ../images/Liver_fibroblasts.png
 

"""

import saenopy

# sphinx_gallery_thumbnail_path = '../../saenopy/img/thumbnails/liver_fibroblast_icon.png'

# %%
# Downloading the example data files
# ----------------------------------
# The folder structure is as follows. There are three cells recorded at different positions in the gel (Pos004, Pos007, Pos008) and three
# channels (ch00, ch01, ch02). The stack has 376 z positions (z000-z375). All positions are recorded once in the active
# state ("Deformed") and once after relaxation with Cyto D as the reference state ("Relaxed").
#
# ::
#
#    1_ClassicSingleCellTFM
#    ├── Deformed
#    │   └── Mark_and_Find_001
#    │       ├── Pos004_S001_z000_ch00.tif
#    │       ├── Pos004_S001_z000_ch01.tif
#    │       ├── Pos004_S001_z000_ch02.tif
#    │       ├── Pos004_S001_z001_ch02.tif
#    │       ├── ...
#    │       ├── Pos007_S001_z001_ch00.tif
#    │       ├── ...
#    │       ├── Pos008_S001_z001_ch02.tif
#    │       └── ...
#    └── Relaxed
#        └── Mark_and_Find_001
#            ├── Pos004_S001_z000_ch00.tif
#            ├── Pos004_S001_z000_ch01.tif
#            ├── Pos004_S001_z000_ch02.tif
#            ├── Pos004_S001_z001_ch02.tif
#            ├── ...
#            ├── Pos007_S001_z001_ch00.tif
#            ├── ...
#            ├── Pos008_S001_z001_ch02.tif
#            └── ...
#

# download the data
saenopy.loadExample("ClassicSingleCellTFM")

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
    '1_ClassicSingleCellTFM/Deformed/Mark_and_Find_001/Pos*_S001_z{z}_ch{c:00}.tif',
    reference_stack='1_ClassicSingleCellTFM/Relaxed/Mark_and_Find_001/Pos*_S001_z{z}_ch{c:00}.tif',
    output_path='1_ClassicSingleCellTFM/example_output',
    voxel_size=[0.7211, 0.7211, 0.988])

# %%
# Detecting the Deformations
# --------------------------
# Saenopy uses 3D Particle Image Velocimetry (PIV) with the following parameters 
# to calculate matrix deformations between a deformed and relaxed state 
# for three example cells.
#
# +------------------+-------+
# | Piv Parameter    | Value |
# +==================+=======+
# | elementsize      |    14 |
# +------------------+-------+
# | win_mu           |    35 |
# +------------------+-------+
# | signoise_filter  |   1.3 |
# +------------------+-------+
# | drift_correction | True  |
# +------------------+-------+
#

# define the parameters for the piv deformation detection
params = {'elementsize': 14.0, 'win_um': 35.0, 'signoise_filter': 1.3, 'drift_correction': True}

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
        # get two consecutive stacks
        if result.stack_reference is None:
            stack1, stack2 = result.stack[i], result.stack[i + 1]
        # or reference stack and one from the list 
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
# Interpolate the found deformations onto a new mesh which will be used for the regularisation. We use identical element
# size of deformation detection mesh here and keep the overall mesh size the same.
#
# +------------------+-------+
# | Mesh Parameter   | Value |
# +==================+=======+
# | element_size     |    14 |
# +------------------+-------+
# | mesh_size_same   | True  |
# +------------------+-------+
# | reference_stack  |'first'|
# +------------------+-------+
#

# define the parameters to generate the solver mesh and interpolate the piv mesh onto it
params = {'reference_stack': 'first', 'element_size': 14.0, 'mesh_size_same': True}
       

# iterate over all the results objects
for result in results:
    # correct for the reference state
    displacement_list = saenopy.subtract_reference_state(result.mesh_piv, params["reference_stack"])
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
# Define the material model and run the regularisation to fit the measured deformations and get the forces.
#
# +--------------------+---------+
# | Material Parameter | Value   |
# +====================+=========+
# | k                  |    6062 |
# +--------------------+---------+
# | d0                 |  0.0025 |
# +--------------------+---------+
# | lambda_s           |  0.0804 |
# +--------------------+---------+
# | ds                 |  0.034  |
# +--------------------+---------+
#
# +--------------------------+---------+
# | Regularisation Parameter | Value   |
# +==========================+=========+
# | alpha                    |  10**10 |
# +--------------------------+---------+
# | stepper                  |    0.33 |
# +--------------------------+---------+
# | i_max                    |    400  |
# +--------------------------+---------+
# | rel_conv_crit            |  0.009  |
# +--------------------------+---------+
#

# define the parameters to generate the solver mesh and interpolate the piv mesh onto it
params = {'k': 6062.0, 'd0': 0.0025, 'lambda_s': 0.0804, 'ds':  0.034, 'alpha': 10**10, 'stepper': 0.33, 'i_max': 400, 'rel_conv_crit': 0.009}

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
# .. figure:: ../images/Liver_Fibroblast_workflow.png
#   
# The reconstructed force field (right) generates a reconstructed deformation field (middle)
# that recapitulates the measured matrix deformation field (left). The overall cell contractility is 
# calculated as all forcecomponents pointing to the force epicenter.
#
  
 