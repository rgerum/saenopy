import saenopy


# download the data
saenopy.loadExample("DynamicalSingleCellTFM")

# load the time series stack, {z} is the placeholder for the z stack, {t} is the placeholder for the time steps
# use * as a placeholder to import multiple experiments at once
results = saenopy.get_stacks(r'2_DynamicalSingleCellTFM\data\Pos*_S001_t{t}_z{z}_ch00.tif',
                             r'2_DynamicalSingleCellTFM\example_output',
                             voxel_size=[0.2407, 0.2407, 1.0071], time_delta=60)

# define the parameters for the piv deformation detection
params = {'win_um': 12.0, 'elementsize': 4.0, 'signoise_filter': 1.3, 'drift_correction': True}

# iterate over all the results objects
for result in results:
    # set the parameters
    result.piv_parameter = params
    # iterate over all stack pairs
    for i in range(len(result.stack) - 1):
        # and calculate the displacement between them
        result.mesh_piv[i] = saenopy.get_displacements_from_stacks(result.stack[i], result.stack[i + 1],
                                                           params["win_um"],
                                                           params["elementsize"],
                                                           params["signoise_filter"],
                                                           params["drift_correction"])
    # save the displacements
    result.save()

# define the parameters to generate the solver mesh and interpolate the piv mesh onto it
params = {'reference_stack': 'first', 'element_size': 4.0, 'inner_region': 100.0, 'thinning_factor': 0.2, 'mesh_size_same': True, 'mesh_size_x': 200.0, 'mesh_size_y': 200.0, 'mesh_size_z': 200.0}

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

# define the parameters to generate the solver mesh and interpolate the piv mesh onto it
params = {'k': 1449.0, 'd0': 0.00215, 'lambda_s': 0.032, 'ds': 0.055, 'alpha': 10.0, 'stepper': 0.33, 'i_max': 100}

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

