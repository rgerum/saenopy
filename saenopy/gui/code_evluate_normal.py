import saenopy
from saenopy.solver import get_cell_boundary

# load the relaxed and the contracted stack
# {z} is the placeholder for the z stack
# {c} is the placeholder for the channels
# {t} is the placeholder for the time points
# use * to load multiple stacks for batch processing
# load_existing=True allows to load an existing file of these stacks if it already exists
#results = saenopy.get_stacks(
#    '/home/richard/.local/share/saenopy/1_ClassicSingleCellTFM/example_output/../Deformed/Mark_and_Find_001/Pos004_S001_z{z}_ch{c:00}.tif',
#    reference_stack='/home/richard/.local/share/saenopy/1_ClassicSingleCellTFM/example_output/../Relaxed/Mark_and_Find_001/Pos004_S001_z{z}_ch{c:00}.tif',
#   output_path='/home/richard/.local/share/saenopy/1_ClassicSingleCellTFM/example_output',
#    voxel_size=[0.7211, 0.7211, 0.988],
#    crop=None,
#    load_existing=True)
import shutil
from pathlib import Path
size = 14
saeno = "oldbulk"
saeno = "new"
boundary = True
for pos in ["004", "007", "008"]:
  for a in [2, 3, 10]:
# for saeno in ["old", "new"]:
#  for boundary in [True, False]:
#   for size in [14, 12, 10]:

    target_filename = "/home/richard/.local/share/saenopy/1_ClassicSingleCellTFM/example_output/Pos"+pos+"_S001_z{z}_ch{c00}_a-"+str(a)+".saenopy"
    #if Path(target_filename).exists():
    #    continue
    shutil.copy("/home/richard/.local/share/saenopy/1_ClassicSingleCellTFM/example_output/Pos"+pos+"_S001_z{z}_ch{c00}.saenopy",
                target_filename)
    results = saenopy.load_results(target_filename)
    # or if you want to explicitly load existing results files
    # use * to load multiple result files for batch processing
    # results = saenopy.load_results('/home/richard/.local/share/saenopy/1_ClassicSingleCellTFM/example_output/Pos004_S001_z{z}_ch{c00}.saenopy')

    # define the parameters for the piv deformation detection
    piv_parameters = {'window_size': 35.0, 'element_size': 14.0, 'signal_to_noise': 1.3, 'drift_correction': True}

    # iterate over all the results objects

    # define the parameters to generate the solver mesh and interpolate the piv mesh onto it
    mesh_parameters = {'reference_stack': 'first', 'element_size': size, 'mesh_size': 'piv'}

    if 1:
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

    if 0:
        for result in results:
            if saeno == "old" or saeno == "oldbulk":
                result.solvers[0].mesh.regularisation_mask[:] = 1
            if saeno != "oldbulk":
                get_cell_boundary(result, element_size=mesh_parameters["element_size"]*1e-6, pos=pos, boundary=boundary)
            result.save()

    # define the parameters to generate the solver mesh and interpolate the piv mesh onto it
    material_parameters = {'k': 6062.0, 'd_0': 0.0025, 'lambda_s': 0.0804, 'd_s': 0.034}
    solve_parameters = {'alpha': 10000000000*a, 'step_size': 0.33, 'max_iterations': 200, 'rel_conv_crit': 0.0009, 'prev_t_as_start': True}

    # iterate over all the results objects
    for result in results:
        result.mesh_parameters = material_parameters
        result.solve_parameters = solve_parameters
        for index, M in enumerate(result.solvers):
            # optionally copy the displacement field from the previous time step as a starting value
            if index > 0 and solve_parameters["prev_t_as_start"]:
                M.mesh.displacements[:] = result.solvers[index - 1].mesh.displacements.copy()

            # set the material model
            M.set_material_model(saenopy.materials.SemiAffineFiberMaterial(
                material_parameters["k"],
                material_parameters["d_0"],
                material_parameters["lambda_s"],
                material_parameters["d_s"],
            ))
            # find the regularized force solution
            M.solve_regularized(alpha=solve_parameters["alpha"], step_size=solve_parameters["step_size"],
                                max_iterations=solve_parameters["max_iterations"], rel_conv_crit=solve_parameters["rel_conv_crit"],
                                verbose=True)
            # save the forces
            result.save()
            # clear the cache of the solver
            result.clear_cache(index)

