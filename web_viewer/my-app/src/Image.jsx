import {useLoader} from "@react-three/fiber";
import {TextureLoader} from "three";
import {useState} from "react";
import {useQuery} from "@tanstack/react-query";
import {get_file_from_zip} from "./load_from_zip.js";

export function Image({source, im_shape, voxel_size}) {
  const [z, setZ] = useState(0);
  window.setZ = setZ;
  const texture = useLoader(TextureLoader,  "data:image/gif;base64,R0lGODlhAQABAAD/ACwAAAAAAQABAAACADs=");

  const { isPending, error, data } = useQuery({
    queryKey: [`stacks/0/0/${pad_zero(z, 3)}.jpg`],
    queryFn: () =>
      get_file_from_zip(source, `stacks/0/0/${pad_zero(z, 3)}.jpg`, "blob").then((blob) => new TextureLoader().load(URL.createObjectURL(blob)))
  })
  const current_texture = (isPending || (error !== null)) ? texture : data
  const pos = (-im_shape[2] * voxel_size[2]) / 2 + z * voxel_size[2];
  const w = im_shape[0] * voxel_size[0];
  const h = im_shape[1] * voxel_size[1];
  console.log(pos, w, h)

  return <mesh
    rotation={[-Math.PI / 2, 0, 0]}
    position={[0, pos, 0]}
  >
    <planeGeometry args={[w, h]}/>
    <meshBasicMaterial map={current_texture}/>
  </mesh>
}

function pad_zero(num, places) {
  return String(num).padStart(places, "0");
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