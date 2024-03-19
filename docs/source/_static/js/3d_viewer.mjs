import * as THREE from "three";
//import * as THREE from "https://unpkg.com/three@v0.158.0/build/three.module.js";
import { OrbitControls } from "three/addons/controls/OrbitControls";
//import { OrbitControls } from "https://unpkg.com/three@v0.158.0/examples/jsm/controls/OrbitControls";
import { mergeGeometries } from "three/addons/utils/BufferGeometryUtils.js";
//import { mergeBufferGeometries } from "https://unpkg.com/three@v0.158.0/examples/jsm/utils/BufferGeometryUtils.js";
import { GUI } from "three/addons/libs/lil-gui.module.min.js";
//import { GUI } from "https://unpkg.com/three@v0.158.0/examples/jsm/libs/lil-gui.module.min.js";

import { loadNpy } from "./load_numpy.js";
import { cmaps } from "./colormaps.js";

import {
  BlobWriter,
  BlobReader,
  ZipReader,
} from "https://unpkg.com/@zip.js/zip.js/index.js";
// Creates a ZipReader object reading the zip content via `zipFileReader`,
// retrieves metadata (name, dates, etc.) of the first entry, retrieves its
// content via `helloWorldWriter`, and closes the reader.


const zip_entries = {};
async function get_file_from_zip(url, filename, return_type = "blob") {
  if (zip_entries[url] === undefined) {
    async function get_entries(url) {
      let zipReader;
      if (typeof url === "string") {
        zipReader = new ZipReader(
          new BlobReader(await (await fetch(url)).blob()),
        );
      } else {
        zipReader = new ZipReader(new BlobReader(url));
      }
      const entries = await zipReader.getEntries();
      const entry_map = {};
      for (let entry of entries) {
        console.log(entry.filename);

        entry_map[entry.filename] = entry;
      }
      await zipReader.close();
      return entry_map;
    }
    zip_entries[url] = get_entries(url);
  }
  const entry = (await zip_entries[url])[filename];
  if (!entry) console.error("file", filename, "not found in", url);

  if (entry.filename === filename) {
    const blob = await entry.getData(new BlobWriter());
    if (return_type === "url") return URL.createObjectURL(blob);
    if (return_type === "texture")
      return new THREE.TextureLoader().load(URL.createObjectURL(blob));
    return blob;
  }
}

const ccs_prefix = "saenopy_";

// Arrowhead geometry (cone)
const arrowheadGeometry = new THREE.ConeGeometry(0.5, 1, 6);
arrowheadGeometry.translate(0, 0.5, 0); // Translate to align the base of the cone with the origin

// Shaft geometry (cylinder)
const shaftGeometry = new THREE.CylinderGeometry(0.2, 0.2, 2, 6);
shaftGeometry.translate(0, -1, 0); // Translate to align the cylinder correctly with the cone base

// Merge geometries
const arrowGeometry = mergeGeometries(
  [arrowheadGeometry, shaftGeometry],
  false,
);
arrowGeometry.rotateX(Math.PI / 2);
arrowGeometry.translate(0, 0, 2);

function color_to_hex(color) {
  let hex = color.toString(16);
  while (hex.length < 6) {
    hex = "0" + hex;
  }
  return "#" + hex;
}
function colormap_to_gradient(colormap) {
  let gradient = [];
  for (let i = 0; i < colormap.length; i++) {
    gradient.push(color_to_hex(colormap[i]));
  }
  return "linear-gradient(90deg, " + gradient.join(", ") + ")";
}

function inject_style(style) {
  var styles = document.createElement("style");
  styles.setAttribute("type", "text/css");
  styles.textContent = style;
  document.head.appendChild(styles);
}

function add_logo(parentDom, params) {
  const logo = document.createElement("img");
  logo.className = ccs_prefix + "logo";
  logo.src =
    "https://saenopy.readthedocs.io/en/latest/_static/img/Logo_black.png";
  parentDom.appendChild(logo);
  inject_style(`
       .${ccs_prefix}logo {
           position: absolute;
           left: 0;
           top: 0;
           width: min(${params.logo_width}, 40%);
       }`);
}

function add_drop_show(parentDom, params) {
  const dropshow = document.createElement("div");
  dropshow.className = ccs_prefix + "drop_show";
  dropshow.innerText = "drop .savenopy file"
  parentDom.appendChild(dropshow);
  inject_style(`
        .${ccs_prefix}drop_show {
           position: absolute;
           left: 0;
           top: 0;
           bottom: 0;
           right: 0;
           border: 5px dashed darkred;
           border-radius: 20px;
           pointer-events: none;
           margin: 0px;
           z-index: 100000;
          font-size: 2rem;
          color: white;
          text-align: center;
          display: grid;
          place-content: center;
          font-family: sans;
          backdrop-filter: blur(3px) brightness(0.5);
          opacity: 0;
          transition: opacity 300ms, margin 300ms;
       }
       .${ccs_prefix}highlight .${ccs_prefix}drop_show {
           opacity: 1;
           margin: 10px;
       }`);
}

function add_colormap_gui(parentDom, params) {
  const colorbar = document.createElement("div");
  colorbar.className = ccs_prefix + "colorbar";
  parentDom.parentElement.appendChild(colorbar);
  const colorbar_gradient = document.createElement("div");
  colorbar_gradient.className = ccs_prefix + "colorbar_gradient";
  colorbar.appendChild(colorbar_gradient);
  colorbar_gradient.style.background = colormap_to_gradient(cmaps[params.cmap]);

  const colorbar_title = document.createElement("div");
  colorbar_title.className = ccs_prefix + "title";
  colorbar.appendChild(colorbar_title);

  const ticks = [];
  function add_tick() {
    const colorbar_number = document.createElement("div");
    colorbar_number.innerText = "0";
    colorbar_number.style.left = "10%";
    colorbar_number.className = ccs_prefix + "tick";
    colorbar.appendChild(colorbar_number);
    ticks.push(colorbar_number);
  }
  for (let i = 0; i < 5; i++) {
    add_tick();
  }

  let last_props = {};
  function update(colormap, max, title) {
    if (max !== last_props.max) {
      colorbar.style.display = max === 0 ? "none" : "block";
      for (let i = 0; i < ticks.length; i++) {
        ticks[i].style.left = (i / (ticks.length - 1)) * 100 + "%";
        ticks[i].innerText = ((i / (ticks.length - 1)) * max).toFixed(1);
      }
      last_props.max = max;
    }
    if (colormap !== last_props.colormap) {
      colorbar_gradient.style.background = colormap_to_gradient(
        cmaps[params.cmap],
      );
      last_props.colormap = colormap;
    }
    if (title !== last_props.title) {
      colorbar_title.innerText = title;
      last_props.title = title;
    }
  }

  inject_style(`
         .${ccs_prefix}colorbar {
            font-family: sans-serif;
            position: absolute;
            bottom: 30px;
            left: 20px;
            width: min(300px, 100% - 40px);
            height: 20px;
            background-color: white;   
            color: white;         
         }
         .${ccs_prefix}colorbar .${ccs_prefix}colorbar_gradient {
            width: 100%;
            height: 20px;
         }
         .${ccs_prefix}colorbar .${ccs_prefix}title {
            position: absolute;
              left: 50%;
              top: 0px;
              text-align: center;
              transform: translate(-50%, -100%);
              white-space: nowrap;
              padding-bottom: 5px;
         }
         .${ccs_prefix}colorbar .${ccs_prefix}tick {
             position: absolute;
             bottom: 0;
             left: 0;
             text-align: center;
             --tick-height: 7px;
             transform: translate(-50%, calc(100% + var(--tick-height)));
         }
         .${ccs_prefix}colorbar .${ccs_prefix}tick::before {
             content: "";
              width: 2px;
              height: var(--tick-height);
              display: block;
              background: white;
              position: absolute;
              top: 0;
              left: calc(50% - 1px);
              transform: translateY(-100%);
         }
         `);
  return update;
}

function init_scene(dom_elem) {
  // if no element is defined, add a new one to the body
  if (!dom_elem) {
    dom_elem = document.createElement("canvas");
    document.body.appendChild(dom_elem);
  }
  if (dom_elem.tagName !== "CANVAS") {
    const canvas = document.createElement("canvas");
    dom_elem.appendChild(canvas);
    dom_elem = canvas;
  }
  dom_elem.style.display = "block";

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(
    75,
    window.innerWidth / window.innerHeight,
    0.1,
    1500,
  );
  scene.camera = camera;
  const renderer = new THREE.WebGLRenderer({
    alpha: true,
    canvas: dom_elem,
    antialias: true,
  });
  //renderer.setSize(window.innerWidth, window.innerHeight);
  //document.body.appendChild(renderer.domElement);
  scene.renderer = renderer;
  window.scene = scene;

  function onWindowResize() {
    let container = dom_elem.parentElement;
    let width = container.clientWidth;
    let height = container.clientHeight;
    dom_elem.style.width = width;
    dom_elem.style.height = height;

    // Update camera aspect ratio
    camera.aspect = width / height;
    camera.updateProjectionMatrix();

    // Update renderer size
    renderer.setSize(width, height);
  }
  scene.onWindowResize = onWindowResize;
  document.onWindowResize = onWindowResize;
  onWindowResize();
  // Add a resize event listener
  window.addEventListener("resize", onWindowResize, false);

  return scene;
}

function set_camera(scene, r, theta_deg, phi_deg) {
  let camera = scene.camera;

  // Convert current camera position to spherical coordinates
  const currentSpherical = new THREE.Spherical().setFromVector3(
    camera.position,
  );

  // Only update radius if `r` is provided
  const radius = r !== undefined ? r : currentSpherical.radius;

  // Convert degrees to radians, and only update if values are provided
  let theta =
    theta_deg !== undefined
      ? (theta_deg * Math.PI) / 180
      : currentSpherical.theta;
  let phi =
    phi_deg !== undefined ? (phi_deg * Math.PI) / 180 : currentSpherical.phi; // Assuming phi is always provided for simplicity

  // Convert spherical to Cartesian coordinates for the camera position
  const spherical = new THREE.Spherical(radius, phi, theta);
  const position = new THREE.Vector3().setFromSpherical(spherical);

  // Set camera position
  camera.position.set(position.x, position.y, position.z);

  // Make the camera look at the scene center or any other point of interest
  camera.lookAt(scene.position);
}

function add_cube(scene, params) {
  // Cube setup
  const geometry = new THREE.BoxGeometry(1, 1, 1);
  const wireframe = new THREE.EdgesGeometry(geometry);
  const material = new THREE.LineBasicMaterial({ color: 0xffffff });
  let cube = new THREE.LineSegments(wireframe, material);
  scene.add(cube);

  function update_cube() {
    if (params.cube === "field") {
      cube.scale.x = params.extent[1] * 1e6 * 2;
      cube.scale.y = params.extent[3] * 1e6 * 2;
      cube.scale.z = params.extent[5] * 1e6 * 2;
    } else if (params.cube === "stack") {
      cube.scale.x =
        params.data.stacks.im_shape[0] * params.data.stacks.voxel_size[0];
      cube.scale.y =
        params.data.stacks.im_shape[1] * params.data.stacks.voxel_size[1];
      cube.scale.z =
        params.data.stacks.im_shape[2] * params.data.stacks.voxel_size[2];
    } else {
      cube.scale.x = 0;
      cube.scale.y = 0;
      cube.scale.z = 0;
    }
  }
  update_cube();
  return update_cube;
}

function pad_zero(num, places) {
  return String(num).padStart(places, "0");
}

const pending = {
  state: "pending",
};

function getPromiseState(promise) {
  // We put `pending` promise after the promise to test,
  // which forces .race to test `promise` first
  return Promise.race([promise, pending]).then(
    (value) => {
      if (value === pending) {
        return value;
      }
      return {
        state: "resolved",
        value,
      };
    },
    (reason) => ({ state: "rejected", reason }),
  );
}

async function add_image(scene, params) {
  let w = params.data.stacks.im_shape[0] * params.data.stacks.voxel_size[0];
  let h = params.data.stacks.im_shape[1] * params.data.stacks.voxel_size[1];
  let d = 0;
  // Image setup
  const imageGeometry = new THREE.PlaneGeometry(w, h);
  // a black image texture to start
  const texture = new THREE.TextureLoader().load(
    "data:image/gif;base64,R0lGODlhAQABAAD/ACwAAAAAAQABAAACADs=",
  );
  const imageMaterial = new THREE.MeshBasicMaterial({ map: texture });
  imageMaterial.side = THREE.DoubleSide;
  let imagePlane = new THREE.Mesh(imageGeometry, imageMaterial);
  imagePlane.rotation.x = -Math.PI / 2;
  scene.add(imagePlane);

  //const texture_await = get_textures_from_zip("data/stack/stack.zip");
  //let textures;

  const textures = [];
  for (let i = 0; i < params.data.stacks.z_slices_count; i++) {
    textures.push(
      get_file_from_zip(
        params.data.path,
        "stacks/0/" +
          params.data.stacks.channels[0] +
          "/" +
          pad_zero(i, 3) +
          ".jpg",
        "texture",
      ),
    );
    textures[i].then((v) => {
      textures[i] = v;
    });
  }

  async function update() {
    if (params.image === "z-pos") {
      imagePlane.position.y =
        (-params.data.stacks.im_shape[2] * params.data.stacks.voxel_size[2]) /
          2 +
        params.z * params.data.stacks.voxel_size[2];
      imagePlane.scale.x = 1;
    } else if (params.image === "floor") {
      imagePlane.position.y =
        (-params.data.stacks.im_shape[2] * params.data.stacks.voxel_size[2]) /
        2;
      imagePlane.scale.x = 1;
    } else {
      imagePlane.scale.x = 0;
    }
    const z = Math.floor(params.z);
    if (textures[z].then === undefined)
      imagePlane.material.map = await textures[z];
    else {
      textures[z].then(update);
    }
  }
  update();
  return update;
}

async function add_test(scene, params) {
  const color = new THREE.Color();

  let count = 0;
  //const material = new THREE.MeshBasicMaterial({ color: 0xffffff });
  const material = new THREE.MeshPhongMaterial({ color: 0xdfdfdf });
  let mesh = undefined;
  const dummyObject = new THREE.Object3D();

  const light = new THREE.HemisphereLight(0xffffff, 0x888888, 3);
  light.position.set(0, 1, 0);
  scene.add(light);
  scene.light = light;

  function convert_pos(x, y, z) {
    const f = 1e6;
    return new THREE.Vector3(-y * f, z * f, -x * f);
  }
  function convert_vec(x, y, z) {
    const f = params.data.fields[params.field]?.factor || 1;
    return new THREE.Vector3(-y * f, z * f, -x * f);
  }
  const colormap_update = add_colormap_gui(scene.renderer.domElement, params);

  let scaleFactor = 1;
  let last_field = {};
  let nodes, vectors;
  async function draw() {
    let arrows = last_field.arrows || [];
    let max_length = last_field.max_length || 0;
    let needs_update = false;

    if (params.path !== last_field.path || params.field !== last_field.field || params.data.path !== last_field.data_path) {
      max_length = 0;
      arrows = [];
      last_field.path = params.path;
      last_field.field = params.field;
      last_field.data_path = params.data.path;
      needs_update = true;
      if (params.field !== "none") {
        try {
          nodes = await loadNpy(
            await get_file_from_zip(
              params.data.path,
              params.data.fields[params.field].nodes,
              "blob",
            ),
          );
          vectors = await loadNpy(
            await get_file_from_zip(
              params.data.path,
              params.data.fields[params.field].vectors,
              "blob",
            ),
          );
        } catch (e) {}
      }

      if (!nodes || !vectors) {
      } else {
        for (let i = 0; i < nodes.header.shape[0]; i++) {
          const position = nodes.header.fortran_order
            ? convert_pos(
                nodes[i],
                nodes[i + nodes.header.shape[0]],
                nodes[i + nodes.header.shape[0] * 2],
              )
            : convert_pos(
                nodes[i * nodes.header.shape[1]],
                nodes[i * nodes.header.shape[1] + 1],
                nodes[i * nodes.header.shape[1] + 2],
              );
          const orientationVector = vectors.header.fortran_order
            ? convert_vec(
                vectors[i],
                vectors[i + vectors.header.shape[0]],
                vectors[i + vectors.header.shape[0] * 2],
              )
            : convert_vec(
                vectors[i * vectors.header.shape[1]],
                vectors[i * vectors.header.shape[1] + 1],
                vectors[i * vectors.header.shape[1] + 2],
              );
          const target = position.clone().add(orientationVector);
          const scaleValue = orientationVector.length();
          if (scaleValue > max_length) {
            max_length = scaleValue;
          }
          arrows.push([position, target, scaleValue]);
        }
      }

      if (nodes) {
        const [min_x, max_x] = get_extend(nodes, 0);
        const [min_y, max_y] = get_extend(nodes, 1);
        const [min_z, max_z] = get_extend(nodes, 2);
        params.extent = [min_x, max_x, min_y, max_y, min_z, max_z];
      }
    }
    last_field.arrows = arrows;
    last_field.max_length = max_length;

    const cmap = cmaps[params.cmap];
    if (last_field.cmap !== params.cmap) {
      last_field.cmap = params.cmap;
      needs_update = true;
    }

    if (arrows.length > count) {
      count = arrows.length;
      if (mesh) scene.remove(mesh);
      mesh = new THREE.InstancedMesh(arrowGeometry, material, count);
      scene.add(mesh);
    }
    if (mesh) mesh.count = arrows.length;
    if (last_field.scale !== params.scale) {
      last_field.scale = params.scale;
      needs_update = true;
    }
    if (needs_update) {
      for (let i = 0; i < arrows.length; i++) {
        const [position, target, scaleValue] = arrows[i];

        dummyObject.position.copy(position);
        dummyObject.lookAt(target);
        dummyObject.scale.set(
          scaleValue * params.scale,
          scaleValue * params.scale,
          scaleValue * params.scale,
        ); // Set uniform scale based on the vector's length
        dummyObject.updateMatrix();

        mesh.setMatrixAt(i, dummyObject.matrix);
        mesh.setColorAt(
          i,
          color.setHex(
            cmap[
              Math.min(
                cmap.length - 1,
                Math.floor((scaleValue / max_length) * (cmap.length - 1)),
              )
            ],
          ),
        );
      }
      if (mesh) {
        mesh.instanceMatrix.needsUpdate = true;
        mesh.instanceColor.needsUpdate = true;
      }
    }

    colormap_update(
      params.cmap,
      params.show_colormap ? max_length : 0,
      params.data.fields[params.field]
        ? `${params.field} (${params.data.fields[params.field].unit})`
        : "",
    );
  }
  await draw();

  function setScale(scaleValue) {
    scaleFactor = scaleValue;
    draw();
  }

  return { setScale, draw };
}

function get_extend(nodes, offset) {
  let max = -Infinity;
  let min = Infinity;
  for (let i = 0; i < nodes.length / 3; i++) {
    const v = nodes[i * 3 + offset];
    if (v > max) {
      max = v;
    }
    if (v < min) {
      min = v;
    }
  }
  return [min, max];
}

async function load_add_field(scene, params) {
  const controls = await add_test(scene, params);
  return controls.draw;
}

export async function init(initial_params) {
  const params = {
    scale: 1,
    cmap: "turbo", // ["turbo", "viridis"]
    field: "fitted deformations", // fitted deformations
    z: 0,
    cube: "field", // ["none", "stack", "field"]
    image: "z-pos", // ["none", "z-pos", "floor"]
    background: "black",
    height: "400px",
    width: "auto",
    logo_width: "200px",
    zoom: 1,
    animations: [],
    pre_load_images: true,
    show_controls: true,
    show_colormap: true,
    mouse_control: true,
    extent: [0, 1, 0, 1, 0, 1],
    ...initial_params,
  };
  params.data = {
    fields: {
      "measured deformations": {
        nodes: "mesh_piv/0/nodes.npy",
        vectors: "mesh_piv/0/displacements_measured.npy",
        unit: "\u00b5m",
        factor: 1000000.0,
      },
      "target deformations": {
        nodes: "solvers/0/mesh/nodes.npy",
        vectors: "solvers/0/mesh/displacements_target.npy",
        unit: "\u00b5m",
        factor: 1000000.0,
      },
      "fitted deformations": {
        nodes: "solvers/0/mesh/nodes.npy",
        vectors: "solvers/0/mesh/displacements.npy",
        unit: "\u00b5m",
        factor: 1000000.0,
      },
      "fitted forces": {
        nodes: "solvers/0/mesh/nodes.npy",
        vectors: "solvers/0/mesh/forces.npy",
        unit: "nN",
        factor: 1000000000.0,
      },
    },
    ...params.data,
  };

  if (initial_params.dom_node) {
    initial_params.dom_node.style.position = "relative";
    initial_params.dom_node.style.background = params.background;
    initial_params.dom_node.style.minHeight = params.height;
    initial_params.dom_node.style.width = params.width;
  }

  // Scene setup
  const scene = init_scene(initial_params.dom_node, params);

  add_logo(scene.renderer.domElement.parentElement, params);
  add_drop_show(scene.renderer.domElement.parentElement, params);

  if (params.data === undefined) {
    const data = await (await fetch(initial_params.path + "/data.json")).json();
    params.data = data;
  }
  const update_image = params.data.stacks
    ? await add_image(scene, params)
    : () => {};

  if (params.mouse_control) {
    const controlsCam = new OrbitControls(scene.camera, scene.renderer.domElement);
    controlsCam.update();
    scene.controls = controlsCam;
  }

  //add_arrows();
  const update_field = params.data.fields
    ? await load_add_field(scene, params)
    : () => {};
  const update_cube = add_cube(scene, params);

  const radius = params.extent[0]
    ? params.extent[1] * 1e6 * 4
    : params.data.stacks.im_shape[0] * params.data.stacks.voxel_size[0] * 2;
  set_camera(scene, radius / params.zoom, 30, 60);

  async function update_all() {
    update_image();
    await update_field();
    update_cube();
  }
  // Animation loop
  animate(scene, params, update_all);

  if (params.show_controls) {
    const gui = new GUI({ container: scene.renderer.domElement.parentElement });
    gui.domElement.classList.add("autoPlace");
    gui.domElement.style.position = "absolute";
    window.gui = gui;

    const options = ["none"];
    for (let name in params.data.fields) {
      options.push(name);
    }
    if (options.length > 1) {
      gui.add(params, "scale", 0, 10).onChange(update_all);
      gui.add(params, "field", options).onChange(update_all);
    }
    if (options.length > 1)
      gui.add(params, "cmap", Object.keys(cmaps)).onChange(update_all);
    const cube_options = ["none"];
    if (options.length > 1) cube_options.push("field");
    if (params.data.stacks) cube_options.push("stack");
    gui.add(params, "cube", cube_options).onChange(update_all);
    if (params.data.stacks)
      gui.add(params, "image", ["none", "z-pos", "floor"]).onChange(update_all);
    if (options.length > 1)
      gui.add(params, "show_colormap").onChange(update_all);
    if (params.data.stacks)
      gui
        .add(params, "z", 0, params.data.stacks.z_slices_count - 1, 1)
        .onChange(update_all);

    gui.close();
  }
  add_drop(scene.renderer.domElement.parentElement, params, update_all);
}

let animation_time = new Date();
function animate(scene, params, update_all) {
  requestAnimationFrame(() => animate(scene, params, update_all));

  const current_time = new Date();
  const delta_t = (current_time - animation_time) / 1000;
  animation_time = current_time;

  if (params.mouse_control) scene.controls.update();
  for (let animation of params.animations) {
    if (animation.type === "scan") {
      animation.z = animation.z || 0;
      animation.z += (animation.speed || 10) * delta_t;
      params.z = Math.floor(animation.z % params.data.stacks.z_slices_count);
      update_all();
    }
    if (animation.type === "rotate") {
      const campos = new THREE.Spherical().setFromVector3(camera.position);
      campos.theta += (((animation.speed || 10) * Math.PI) / 180) * delta_t;
      scene.camera.position.setFromSpherical(campos);
      scene.camera.lookAt(scene.position);
    }
    if (animation.type === "scroll-tilt") {
      if (
        scene.renderer.domElement.getBoundingClientRect().top !==
        animation.last_top_pos
      ) {
        const factor =
          (scene.renderer.domElement.getBoundingClientRect().top +
            scene.renderer.domElement.getBoundingClientRect().height) /
          (window.innerHeight +
            scene.renderer.domElement.getBoundingClientRect().height);
        const top = animation.top || 120;
        const bottom = animation.bottom || 30;
        set_camera(
          scene,
          undefined,
          undefined,
          top * (1 - factor) + bottom * factor,
        );
        animation.last_top_pos =
          scene.renderer.domElement.getBoundingClientRect().top;
      }
    }
  }

  if (scene.light) {
    const campos = new THREE.Spherical().setFromVector3(scene.camera.position);
    const lightpos = new THREE.Spherical(
      campos.radius,
      campos.phi,
      campos.theta + (30 / 180) * Math.PI,
    );
    scene.light.position.setFromSpherical(lightpos);
  }

  scene.renderer.render(scene, scene.camera);
}

function add_drop(dropZone, params, update_all) {
  // Prevent default drag behaviors
  ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, preventDefaults, false);
    document.body.addEventListener(eventName, preventDefaults, false);
  });

  // Highlight drop area when item is dragged over it
  ['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, highlight, false);
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, unhighlight, false);
  });

  // Handle dropped files
  dropZone.addEventListener('drop', handleDrop, false);

  function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
  }

  function highlight(e) {
    dropZone.classList.add(ccs_prefix+'highlight');
  }

  function unhighlight(e) {
    dropZone.classList.remove(ccs_prefix+'highlight');
  }

  function handleDrop(e) {
    var dt = e.dataTransfer;
    var files = dt.files;

    handleFiles(files);
  }

  function handleFiles(files) {
    ([...files]).forEach(loadFile);
  }

  function loadFile(file) {
    console.log("loadFile", file)
    params.data.path = file;
    update_all();
    return
    var reader = new FileReader();
    reader.readAsText(file); // Or readAsDataURL(file) for images
    reader.onloadend = function() {
      console.log(reader.result); // Do something with the file content
      // For example, display the file content in the drop zone
      dropZone.textContent = reader.result;
    };
  }
}
