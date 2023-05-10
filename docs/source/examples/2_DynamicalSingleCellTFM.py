"""
Dynamic TFM (immune cells)
=========================================

This example evaluates a single natural killer cell that migrated through 1.2mg/ml collagen, recorded for 24min.



 .. figure:: ../images/Gif_nk_dynamic_example.gif
    :scale: 40%
    :align: center

This example can also be evaluated with the graphical user interface.


"""


# sphinx_gallery_thumbnail_path = '../../saenopy/img/thumbnails/Dynamic_icon.png'


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
saenopy.load_example("DynamicalSingleCellTFM")

# %%
# Loading the Stacks
# ------------------
#
# Saenopy is very flexible in loading stacks from any filename structure.
# Here we do not have multiple positions, so we do not need to use and asterisk * for batch processing.
# We replace the number of the channels "ch00" with a channel placeholder "ch{c:00}" to indicate that this refers to
# the channels and which channel to use as the first channel where the deformations should be detected.
# We replace the number of the z slice "z000" with a z placeholder "z{z}" to indicate that this number refers to the
# z slice.
# We replace the number of the time step "t00" with a time placeholder "t{t}" to indicate that this number refers to the
# time step.
# Due to technical reasons and the soft nature of collagen hydrogels, 
# acquiring fast image stacks  with our galvo stage (within 10 seconds) can cause external motion 
# in the upper and lower region the image stack, as the stage accelerates and
# decelerates here. Therefore, we always acquire a larger stack in z-direction and then discard 
# the upper and lower regions (20 images here).


# load the relaxed and the contracted stack
# {z} is the placeholder for the z stack
# {c} is the placeholder for the channels
# {t} is the placeholder for the time points

results = saenopy.get_stacks(r'2_DynamicalSingleCellTFM\data\Pos*_S001_t{t}_z{z}_ch{c:00}.tif',
                             r'2_DynamicalSingleCellTFM\example_output',
                             voxel_size=[0.2407, 0.2407, 1.0071], 
                             time_delta=60,
                             crop={"z": (20, -20)}
                             )


# %%
# Detecting the Deformations
# --------------------------
# 
# .. figure:: ../images/nk_dynamic_stacks.png
#
# Saenopy uses 3D Particle Image Velocimetry (PIV) to calculate the collagen matrix deformations 
# generated by the natural killer cell at different times. For this, we use the following parameters.
#
# +------------------+-------+
# | Piv Parameter    | Value |
# +==================+=======+
# | element_size     |     4 |
# +------------------+-------+
# | window_size      |    12 |
# +------------------+-------+
# | signal_to_noise  |   1.3 |
# +------------------+-------+
# | drift_correction | True  |
# +------------------+-------+
#

# define the parameters for the piv deformation detection
piv_parameters = {'element_size': 4.0, 'window_size': 12.0, 'signal_to_noise': 1.3, 'drift_correction': True}


# iterate over all the results objects
for result in results:
    # set the parameters
    result.piv_parameters = piv_parameters
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
        
        # due to the acceleration of the galvo stage there can be shaking in the
        # lower or upper part of the stack. Therefore, we recorded larger
        # z-regions and then discard the upper or lower parts        

        # and calculate the displacement between them
        result.mesh_piv[i] = saenopy.get_displacements_from_stacks(stack1, stack2,
                                                                    piv_parameters["window_size"],
                                                                    piv_parameters["element_size"],
                                                                    piv_parameters["signal_to_noise"],
                                                                    piv_parameters["drift_correction"])
    # save the displacements
    result.save()


# %%
# Visualizing Results
# ----------------------------------
# You can save the resulting 3D fields by simply using the **export image** dialog.
# Here we underlay a bright field image of the cell for a better overview
# and export a **.gif** file
# 
# .. figure:: ../images/nk_dynamic_piv_export_4fps.gif   
#   :scale: 40%
#   :align: center
#
            

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
# | mesh_size        | "piv"   |
# +------------------+--------+
#

# define the parameters to generate the solver mesh and interpolate the piv mesh onto it
mesh_parameters = {'reference_stack': 'median', 'element_size': 4.0, 'mesh_size': "piv"}

# iterate over all the results objects
for result in results:
    # correct for the reference state
    displacement_list = saenopy.subtract_reference_state(result.mesh_piv, mesh_parameters["reference_stack"])
    # set the parameters
    result.mesh_parameters = mesh_parameters
    # iterate over all stack pairs
    for i in range(len(result.mesh_piv)):
        # and create the interpolated solver mesh
        result.solver[i] = saenopy.interpolate_mesh(result.mesh_piv[i], displacement_list[i], mesh_parameters)
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
# | d_0                | 0.0022  |
# +--------------------+---------+
# | lambda_s           |  0.032  |
# +--------------------+---------+
# | d_s                | 0.055   |
# +--------------------+---------+
#
# +--------------------------+---------+
# | Regularisation Parameter | Value   |
# +==========================+=========+
# | alpha                    |  10**10 |
# +--------------------------+---------+
# | step_size                |    0.33 |
# +--------------------------+---------+
# | max_iterations           |    100  |
# +--------------------------+---------+
#

# define the parameters to generate the solver mesh and interpolate the piv mesh onto it
material_parameters = {'k': 1449.0, 'd_0': 0.0022, 'lambda_s': 0.032, 'd_s': 0.055}
solve_parameters = {'alpha':  10**10, 'step_size': 0.33, 'max_iterations': 100}


# iterate over all the results objects
for result in results:
    result.material_parameters = material_parameters
    result.solve_parameters = solve_parameters
    for M in result.solver:
        # set the material model
        M.set_material_model(saenopy.materials.SemiAffineFiberMaterial(
            material_parameters["k"],
            material_parameters["d_0"],
            material_parameters["lambda_s"],
            material_parameters["d_s"],
        ))
        # find the regularized force solution
        M.solve_regularized(alpha=solve_parameters["alpha"], step_size=solve_parameters["step_size"],
                            max_iterations=solve_parameters["max_iterations"], verbose=True)
    # save the forces
    result.save()

