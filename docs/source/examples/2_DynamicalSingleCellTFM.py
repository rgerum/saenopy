"""
Dynamical Single Cell TFM
=========================

This example evaluates a single natural killer cell that migrated through 1.2mg/ml collagen, recorded for 24min.



 .. figure:: ../images/Gif_nk_dynamic_example.gif


This example can also be evaluated with the graphical user interface.


"""
# sphinx_gallery_thumbnail_path = '../../saenopy/img/examples/example2.png'



import saenopy

# %%
# Downloading the example data files
# ----------------------------------
# The folder structure is as follows. There is one position recorded in the gel (Pos002) and two
# channels (ch00, ch01). The stack has 162 z positions (z000-z161). The position is recorded for 24 time steps
# (t00-t23).
#
# ::
#
#    2_DynamicalSingleCellTFM
#    └── data
#        ├── Pos002_S001_t00_z000_ch00.tif
#        ├── Pos002_S001_t00_z000_ch01.tif
#        ├── Pos002_S001_t00_z001_ch00.tif
#        ├── Pos002_S001_t00_z001_ch01.tif
#        ├── Pos002_S001_t00_z002_ch00.tif
#        ├── ...
#        ├── Pos002_S001_t01_z000_ch00.tif
#        ├── Pos002_S001_t01_z000_ch01.tif
#        ├── Pos002_S001_t01_z001_ch00.tif
#        └── ...
#

# download the data
saenopy.loadExample("DynamicalSingleCellTFM")

# %%
# Loading the Stacks
# ------------------
#
# Saenopy is very flexible in loading stacks from any filename structure.
# Here we do not have multiple positions so we do not need to use and asterisk * for batch processing.
# We replace the number of the channels "ch00" with a channel placeholder "ch{c:00}" to indicate that this refers to
# the channels and which channel to use as the first channel where the deformations should be detected.
# We replace the number of the z slice "z000" with a z placeholder "z{z}" to indicate that this number refers to the
# z slice.
# We replace the number of the time step "t00" with a time placeholder "t{t}" to indicate that this number refers to the
# time step.

# load the relaxed and the contracted stack
# {z} is the placeholder for the z stack
# {c} is the placeholder for the channels
# {t} is the placeholder for the time points
# results = saenopy.get_stacks('2_DynamicalSingleCellTFM/data/Pos002_S001_t{t}_z{z}_ch{c:00}.tif',
#                              output_path='2_DynamicalSingleCellTFM\example_output',
#                              voxel_size=[0.2407, 0.2407, 1.0071], time_delta=60)

results = saenopy.get_stacks(r'C:\Users\User\AppData\Local\rgerum\saenopy\2_DynamicalSingleCellTFM\data/Pos002_S001_t{t}_z{z}_ch{c:00}.tif',
                             output_path=r'C:\Users\User\Desktop\aaaaaa\2_DynamicalSingleCellTFM\example_output',
                             voxel_size=[0.2407, 0.2407, 1.0071], time_delta=60, load_existing=True)




# %%
# Detecting the Deformations
# --------------------------
# Saenopy uses here a 3D particle image velocimetry (PIV).
#
# +------------------+-------+
# | Piv Parameter    | Value |
# +==================+=======+
# | elementsize      |     4 |
# +------------------+-------+
# | win_mu           |    12 |
# +------------------+-------+
# | signoise_filter  |   1.3 |
# +------------------+-------+
# | drift_correction | True  |
# +------------------+-------+

# define the parameters for the piv deformation detection
params = {'elementsize': 4.0, 'win_um': 12.0, 'signoise_filter': 1.3, 'drift_correction': True}


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
        
        # Due acceleration of the galvo stage there can be shaking in the 
        # lower or upper part of the stack. Therefore we recorded larger 
        # z-regions and then discard the upper or lower parts        
        # stack1, stack2 = np.array(stack1[:,:,20:-20]), np.array(stack2[:,:,20:-20])
            
        
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
# Interpolate the found deformations onto a new mesh which will be used for the regularisation. We use the same element
# size of deformation detection mesh here and we also keep the overall mesh size the same. We define that as an
# undeformed reference we want to use the median of all stacks.
#
# +------------------+--------+
# | Mesh Parameter   | Value  |
# +==================+========+
# | reference_stack  | median |
# +------------------+--------+
# | element_size     |      4 |
# +------------------+--------+
# | mesh_size_same   | True   |
# +------------------+--------+

# define the parameters to generate the solver mesh and interpolate the piv mesh onto it
params = {'reference_stack': 'median', 'element_size': 4.0, 'mesh_size_same': True}

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
# Define the material model and run the regularisation to fit the measured deformations and get the forces.
#
# +--------------------+---------+
# | Material Parameter | Value   |
# +====================+=========+
# | k                  |    1449 |
# +--------------------+---------+
# | d0                 | 0.0022 |
# +--------------------+---------+
# | lambda_s           |  0.032  |
# +--------------------+---------+
# | ds                 | 0.055   |
# +--------------------+---------+
#
# +--------------------------+---------+
# | Regularisation Parameter | Value   |
# +==========================+=========+
# | alpha                    |  10**10 |
# +--------------------------+---------+
# | stepper                  |    0.33 |
# +--------------------------+---------+
# | i_max                    |    100  |
# +--------------------------+---------+

# define the parameters to generate the solver mesh and interpolate the piv mesh onto it
params = {'k': 1449.0, 'd0': 0.0022, 'lambda_s': 0.032, 'ds': 0.055, 'alpha':  10**10, 'stepper': 0.33, 'i_max': 100}


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
        M.solve_regularized(stepper=params["stepper"], i_max=params["i_max"], alpha=params["alpha"], verbose=True)
    # save the forces
    result.save()

