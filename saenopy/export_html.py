import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from pathlib import Path
import saenopy
import numpy as np
from PIL import Image
import json
import zipfile
from io import BytesIO

def export_cmaps(cmaps):
    text = "export const cmaps = {\n"

    for name in cmaps:
        cmap = plt.get_cmap("viridis")

        colors = []
        for i in range(256):
            colors.append("0x" + to_hex(cmap(i))[1:])

        text += f"  \"{name}\": [" + ",".join(colors) + "],\n"
    text = text[:-2]
    text += "\n}"
    return text

#print(export_cmaps(["viridis", "turbo"]))

def export_html(result: saenopy.Result, path, zip_filename="data.zip", stack_quality=75, stack_downsample=4, stack_downsample_z=2):
    path = Path(path)
    path.mkdir(exist_ok=True)

    with zipfile.ZipFile(path / zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipFp:

        path_data = "0/"
        def save(path, data, dtype):
            #new_data = np.zeros(data.shape, dtype=dtype)
            #new_data[:] = data
            #print(new_data.strides, new_data.shape)

            image_file = BytesIO()
            np.save(image_file, np.asarray(data).astype(dtype))
            zipFp.writestr(path, image_file.getvalue())

        data = {}
        mesh = result.solvers[0].mesh
        piv = result.mesh_piv[0]

        path_data = "mesh_piv/0/"
        save(path_data + "nodes.npy", piv.nodes, np.float32)
        save(path_data + "displacements_measured.npy", piv.displacements_measured, np.float32)

        path_data = "solvers/0/mesh/"
        save(path_data + "nodes.npy", mesh.nodes, np.float32)
        save(path_data + "displacements_target.npy", mesh.displacements_target, np.float32)
        save(path_data + "displacements.npy", mesh.displacements, np.float32)
        save(path_data + "forces.npy", -mesh.forces * mesh.regularisation_mask[:, None], np.float32)

        #path_stack = path_data + "stack"
        #path_stack.mkdir(exist_ok=True)

        z_count = 0
        for z in range(result.get_data_structure()["z_slices_count"]):
            if z % stack_downsample_z != 0:
                continue
            image_file = BytesIO()

            im = result.stacks[0].get_image(z, 0)
            im = im - np.min(im)
            im = im / np.max(im) * 255
            im = Image.fromarray(im.astype(np.uint8))
            if stack_downsample > 1:
                im = im.resize((im.width//stack_downsample, im.height//stack_downsample), Image.Resampling.LANCZOS)
            im.save(image_file, 'JPEG', quality=stack_quality)
            zipFp.writestr(f"stacks/0/0/{z//stack_downsample_z:03d}.jpg", image_file.getvalue())
            z_count += 1
        voxel_size = result.get_data_structure()["voxel_size"]
        voxel_size[2] = voxel_size[2] * stack_downsample_z

        save("stacks/0/voxel_size.npy", voxel_size, np.float32)

    im_shape = [int(x) for x in result.get_data_structure()["im_shape"]]
    im_shape[2] = im_shape[2]//stack_downsample_z

    data["path"] = zip_filename

    data["stacks"] = {
        "channels": ["0"],
        "z_slices_count": z_count,
        "im_shape": tuple(im_shape),
        "voxel_size": tuple(voxel_size),
    }
    """
    data["fields"] = {
        "measured deformations": {"nodes": "mesh_piv/0/nodes.npy", "vectors": "mesh_piv/0/displacements_measured.npy", "unit": "µm", "factor": 1e6},
        "target deformations": {"nodes": "solvers/0/mesh/nodes.npy", "vectors": "solvers/0/mesh/displacements_target.npy", "unit": "µm", "factor": 1e6},
        "fitted deformations": {"nodes": "solvers/0/mesh/nodes.npy", "vectors": "solvers/0/mesh/displacements.npy", "unit": "µm", "factor": 1e6},
        "fitted forces": {"nodes": "solvers/0/mesh/nodes.npy", "vectors": "solvers/0/mesh/forces.npy", "unit": "nN", "factor": 1e9},
    }
    """
    data["time_point_count"] = result.get_data_structure()["time_point_count"]

    print(data)

    html = """<!doctype html>
    <html lang="en">
      <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>Saenopy Viewer</title>
        <style>
          html {
            height: 100%;
          }
          body {
            margin: 0;
            background: black;
            height: 100%;
          }
        </style>

        <script type="importmap">
          {
            "imports": {
              "three": "https://unpkg.com/three@v0.158.0/build/three.module.js",
              "three/addons/": "https://unpkg.com/three@v0.158.0/examples/jsm/",
              "3d_viewer": "./3d_viewer.mjs"
            }
          }
        </script>
      </head>
      <body>
        <script type="module">
          import { init } from "3d_viewer";
          init({data: """+json.dumps(data)+"""});
        </script>
      </body>
    </html>
        """

    with open(path / "index.html", "w") as fp:
        fp.write(html)
        #with open(path_data / "data.json", "w") as fp:
        #    json.dump(data, fp)


result = saenopy.load("/home/richard/.local/share/saenopy/1_ClassicSingleCellTFM/example_output/Pos007_S001_z{z}_ch{c00}.saenopy")
export_html(result, "tmp_test_export", "data.zip", stack_quality=75, stack_downsample=4, stack_downsample_z=1)
