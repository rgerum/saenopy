Create Finite Element Mesh
==========================
reference stack
~~~~~~~~~~~~~~~
This parameter can have values depending on the type of experiment. If the experiment has a
specified reference stack (with the relaxed cell) the appropriate option will be:

    - reference stack

If the measurement is a time series, it is also possible to set a single time point as the reference state 
(e.g. after cell seeding or relaxation) using the "reference stack" option. Further options are to calculate the 
deformations and forces beetween consecutive stacks at each time step ("next"), or to locally accumulate the 
deformations over time at each voxel and calculate the cumulated forces at each time step ("cumul."). Furthermore, if 
the matrix is in an undeformed state for most of the time (e.g. moving cells imaged over a long periods of time), 
the median state over subsequent stacks can be calculated and used as a reference stack ("median"). 
 
    - reference stack
    - next
    - median
    - cumul.
    

mesh element size
~~~~~~~~~~~~~~~~~
The element size for the mesh to which the mesh from the PIV step will be interpolated.
Smaller element sizes will produce a finer mesh, which will capture more detail of the deformation field, but will take longer to 
calculate in the Fit Forces step. To reduce the loss of information when interpolating to a new mesh, it may 
be beneficial to use the same mesh size as the final mesh or an integral multiple of it.

.. figure:: images/parameters/mesh_size.png


mesh size same
~~~~~~~~~~~~~~
Whether to use the same mesh size as the piv mesh. If unchecked, a custom mesh size can be specified. Smaller mesh sizes
then the PIV mesh would cut off information. Larger mesh sizes can be good if the piv mesh is very small and the spatial
decay of the deformation field is not well captured. Larger mesh sizes can lead to more accurate force reconstructions.

.. figure:: images/parameters/mesh_total_size.png

2D deformation exclusion masks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The optional ``ignore_mask_2d`` argument excludes target vectors after PIV
interpolation. Nonzero pixels of the 2D YX mask exclude the entire Z column;
zero pixels keep it. Images are aligned by their centres using physical voxel
sizes. Targets outside the mask image remain included.

Load a mask directly from an image file, without a GUI or a Result object::

    mask = saenopy.load_2d_deformation_masks("ignore.png", voxel_size=(dx, dy, dz))[0]
    solver = saenopy.interpolate_mesh(
        piv_mesh, displacements, mesh_parameters,
        ignore_mask_2d=mask)

For multiple deformation time points, use the same mask or a naturally sorted
``{t}`` image series. Mask ``i`` applies to deformation ``i``, corresponding to
the first active image of each pair::

    masks = saenopy.load_2d_deformation_masks(
        "ignore_t{t}.png", voxel_size=result.stacks[0].voxel_size)
    if len(masks) not in (1, len(displacement_list)):
        raise ValueError("Provide one mask for all frames or one per deformation.")
    for i, displacements in enumerate(displacement_list):
        result.solvers[i] = saenopy.interpolate_mesh(
            result.mesh_piv[i], displacements, result.mesh_parameters,
            ignore_mask_2d=masks[0 if len(masks) == 1 else i])
    result.save()

A NumPy YX array can also be passed directly with
``ignore_mask_2d=mask_yx, mask_voxel_size=(dx, dy, dz)``. The older argument name
``custom_mask`` remains an alias. Three-dimensional masks are rejected.

In the GUI, **use 2D exclusion mask** is unchecked by default. Checking it
reveals the source selector with **Add custom mask…**. Unchecking it disables
mask exclusion. **Use original voxel size** initially copies the matrix voxel size;
uncheck it to edit X/Y in the same row. One image can apply to all time
points, or a series can match the matrix frames or deformation pairs. Sources
and pixel sizes are saved with the result, and saved active masks restore the
checkbox. Reset and re-interpolate an existing mesh after changing the mask.

The image dialog adds one channel of one measurement. Channel and position
values stay fixed; batch/all selections are unavailable. Choose **one image
for all time points** to reuse a selected time point, or **one image per time
point** to expose the time index. The dialog shows the required series length
and checks it before import. Filenames and image resolution may differ from
the matrix images; time points are paired in natural filename order.
