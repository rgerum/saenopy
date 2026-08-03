"""
.. _SectionExampleSingleCell:

Classic TFM (liver fibroblasts)
===============================

This example evaluates three hepatic stellate cells in 1.2mg/ml collagen with relaxed and deformed stacks. The relaxed stacks were recorded with cytochalasin D treatment of the cells.
This example can also be evaluated with the graphical user interface.

 .. figure:: ../images/examples/single_cell_tfm/liver_fibroblasts.png
 

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
saenopy.load_example("ClassicSingleCellTFM")

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
# | element_size     |    14 |
# +------------------+-------+
# | window_size      |    35 |
# +------------------+-------+
# | signal_to_noise  |   1.3 |
# +------------------+-------+
# | drift_correction | True  |
# +------------------+-------+
#

# define the parameters for the piv deformation detection
piv_parameters = {'element_size': 14.0, 'window_size': 35.0, 'signal_to_noise': 1.3, 'drift_correction': True}

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
# Interpolate the found deformations onto a new mesh which will be used for the regularisation. We use identical element
# size of deformation detection mesh here and keep the overall mesh size the same.
#
# +------------------+-------+
# | Mesh Parameter   | Value |
# +==================+=======+
# | element_size     |    14 |
# +------------------+-------+
# | mesh_size        | 'piv' |
# +------------------+-------+
# | reference_stack  |'first'|
# +------------------+-------+
#

# define the parameters to generate the solver mesh and interpolate the piv mesh onto it
mesh_parameters = {'reference_stack': 'first', 'element_size': 14.0, 'mesh_size': 'piv'}
       

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
# | k                  |    6062 |
# +--------------------+---------+
# | d_0                |  0.0025 |
# +--------------------+---------+
# | lambda_s           |  0.0804 |
# +--------------------+---------+
# | d_s                |  0.034  |
# +--------------------+---------+
#
# +--------------------------+---------+
# | Regularisation Parameter | Value   |
# +==========================+=========+
# | alpha                    |  10**10 |
# +--------------------------+---------+
# | step_size                |    0.33 |
# +--------------------------+---------+
# | max_iterations           |    400  |
# +--------------------------+---------+
# | rel_conv_crit            |  0.009  |
# +--------------------------+---------+
#

# define the parameters to generate the solver mesh and interpolate the piv mesh onto it
material_parameters = {'k': 6062.0, 'd_0': 0.0025, 'lambda_s': 0.0804, 'd_s':  0.03}
solve_parameters = {'alpha': 10**10, 'step_size': 0.33, 'max_iterations': 400, 'rel_conv_crit': 0.009}

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
# The reconstructed force field (right) generates a reconstructed deformation
# field (middle) that recapitulates the measured matrix deformation field
# (left). The overall cell contractility is calculated as all force components
# pointing to the force epicenter.
#
# The three views share one camera, so dragging or scrolling in any of them
# turns all three together. The sliders below change all three at once.
#
# .. raw:: html
#
#     <div style="display:grid;max-width:100%;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;font-family:inherit">
#       <div>
#         <div style="text-align:center;color:#888;font-size:1.05em;margin-bottom:.3em">Measured<br/>Deformations</div>
#         <div class="saenopy-view" data-field="measured deformations" data-max="10.8" style="height:260px;max-width:100%;overflow:hidden"></div>
#       </div>
#       <div>
#         <div style="text-align:center;color:#888;font-size:1.05em;margin-bottom:.3em">Reconstructed<br/>Deformations</div>
#         <div class="saenopy-view" data-field="fitted deformations" data-max="10.8" style="height:260px;max-width:100%;overflow:hidden"></div>
#       </div>
#       <div>
#         <div style="text-align:center;color:#888;font-size:1.05em;margin-bottom:.3em">Reconstructed<br/>Forces</div>
#         <div class="saenopy-view" data-field="fitted forces" data-log="1" style="height:260px;max-width:100%;overflow:hidden"></div>
#       </div>
#     </div>
#     <div style="margin-top:1em;padding:.9em 1em;border:1px solid #ddd;border-radius:6px;background:#fafafa;font-size:.85em;display:grid;gap:.45em;max-width:100%">
#       <strong style="font-size:1.05em">Rendering controls</strong>
#       <label style="display:flex;align-items:center;gap:.5em">
#         <span style="width:9em">arrow length</span>
#         <input type="range" id="sn-span" min="0.02" max="0.30" step="0.005" value="0.1" style="flex:1">
#         <output for="sn-span" style="width:4em;text-align:right;font-variant-numeric:tabular-nums">0.1</output>
#       </label>
#       <label style="display:flex;align-items:center;gap:.5em">
#         <span style="width:9em">arrow thickness</span>
#         <input type="range" id="sn-thick" min="0.1" max="3" step="0.05" value="1" style="flex:1">
#         <output for="sn-thick" style="width:4em;text-align:right;font-variant-numeric:tabular-nums">1</output>
#       </label>
#       <label style="display:flex;align-items:center;gap:.5em">
#         <span style="width:9em">opacity</span>
#         <input type="range" id="sn-op" min="0.05" max="1" step="0.05" value="1" style="flex:1">
#         <output for="sn-op" style="width:4em;text-align:right;font-variant-numeric:tabular-nums">1</output>
#       </label>
#       <label style="display:flex;align-items:center;gap:.5em">
#         <span style="width:9em">scale max (µm)</span>
#         <input type="range" id="sn-max" min="1" max="20" step="0.1" value="10.8" style="flex:1">
#         <output for="sn-max" style="width:4em;text-align:right;font-variant-numeric:tabular-nums">10.8</output>
#       </label>
#       <label style="display:flex;align-items:center;gap:.5em">
#         <span style="width:9em">zoom</span>
#         <input type="range" id="sn-zoom" min="0.6" max="4" step="0.05" value="1.9" style="flex:1">
#         <output for="sn-zoom" style="width:4em;text-align:right;font-variant-numeric:tabular-nums">1.9</output>
#       </label>
#       <label style="display:flex;align-items:center;gap:.5em">
#         <span style="width:9em">colormap</span>
#         <select id="sn-cmap" style="flex:1"><option>turbo</option><option>viridis</option></select>
#         <span style="width:4em"></span>
#       </label>
#       <label style="display:flex;align-items:center;gap:.5em">
#         <span style="width:9em">force scale</span>
#         <select id="sn-mode" style="flex:1"><option value="log">log</option><option value="linear">linear</option></select>
#         <span style="width:4em"></span>
#       </label>
#       <div style="color:#777">Values are applied to all three panels live. The scale max applies to the two deformation panels.</div>
#     </div>
#     <script type="importmap">
#       {"imports": {
#         "three": "https://unpkg.com/three@0.183.0/build/three.module.js",
#         "three/addons/": "https://unpkg.com/three@0.183.0/examples/jsm/"
#       }}
#     </script>
#     <script type="module">
#       import { init } from "../../_static/js/saenopy_viewer.mjs";
#       const views = [];
#       let syncing = false;
#       const sync = (source) => {
#         if (syncing) return;
#         syncing = true;
#         for (const v of views) {
#           if (v === source) continue;
#           v.camera.position.copy(source.camera.position);
#           v.camera.quaternion.copy(source.camera.quaternion);
#           v.controls.target.copy(source.controls.target);
#           v.controls.update();
#         }
#         syncing = false;
#       };
#       for (const node of document.querySelectorAll(".saenopy-view")) {
#         await init({
#           bundle: "../../_static/data/single-cell-007.sfb.gz",
#           field: node.dataset.field,
#           scale_mode: node.dataset.log ? "log" : "linear",
#           scale_max: node.dataset.max ? Number(node.dataset.max) : undefined,
#           dom_node: node,
#           height: "260px",
#           arrow_span: 0.1,
#           arrow_thickness: 1,
#           arrow_opacity: 1,
#           zoom: 1.9,
#           cube: "field",
#           cube_color: 0x000000,
#           background: "#ffffff",
#           logo_width: "0px",
#           mouse_control: true,
#           show_controls: false,
#           show_colormap: false,
#           on_ready: (params, redraw, ctx) => {
#             const view = { params, redraw, camera: ctx.camera,
#                            controls: ctx.controls, isForce: !!node.dataset.log };
#             views.push(view);
#             ctx.controls.addEventListener("change", () => sync(view));
#           },
#         });
#       }
#       const apply = (fn) => { for (const v of views) { fn(v); v.redraw(); } };
#       const bind = (id, fn) => {
#         const el = document.getElementById(id);
#         const out = document.querySelector(`output[for="${id}"]`);
#         el.addEventListener("input", () => {
#           if (out) out.textContent = el.value;
#           apply((v) => fn(v, el.value));
#         });
#       };
#       bind("sn-span",  (v, x) => { v.params.arrow_span = Number(x); });
#       bind("sn-thick", (v, x) => { v.params.arrow_thickness = Number(x); });
#       bind("sn-op",    (v, x) => { v.params.arrow_opacity = Number(x); });
#       bind("sn-max",   (v, x) => { if (!v.isForce) v.params.scale_max = Number(x); });
#       bind("sn-zoom",  (v, x) => {
#         v.camera.zoom = Number(x);
#         v.camera.updateProjectionMatrix();
#       });
#       document.getElementById("sn-cmap").addEventListener("change", (e) =>
#         apply((v) => { v.params.cmap = e.target.value; }));
#       document.getElementById("sn-mode").addEventListener("change", (e) =>
#         apply((v) => { if (v.isForce) v.params.scale_mode = e.target.value; }));
#     </script>
#
