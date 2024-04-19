import {loadNpy} from "../../../docs/source/_static/js/load_numpy.js";
import {cmaps} from "../../../docs/source/_static/js/colormaps.js";
import React, {useRef, useEffect} from "react";
import * as THREE from "three";
import {useQuery} from "@tanstack/react-query";
import {get_file_from_zip} from "./load_from_zip.js";
import {arrowGeometry} from "./arrow_geometry.js";


const dummyObject = new THREE.Object3D();
function getMatrix(position, target, scale) {
    dummyObject.position.set(position[0], position[1], position[2]);
    dummyObject.lookAt(new THREE.Vector3(target[0], target[1], target[2]));
    dummyObject.scale.set(scale, scale, scale);
    dummyObject.updateMatrix();
    return dummyObject.matrix;
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

function array_get_entry(nodes, i) {
  if(nodes.header.fortran_order)
    return [nodes[i], nodes[i + nodes.header.shape[0]], nodes[i + nodes.header.shape[0] * 2]]
  return [nodes[i * nodes.header.shape[1]], nodes[i * nodes.header.shape[1] + 1], nodes[i * nodes.header.shape[1] + 2]]
}

function array_pos(nodes, i, f) {
  const [x, y, z] = array_get_entry(nodes, i);
  return [-y * f, z * f, -x * f]
}

export function useVectorField({source, field1, field2}) {
const loader1 = useQuery({
    queryKey: [field1],
    queryFn: () => get_file_from_zip(source, field1).then(blob => loadNpy(blob))})
  const loader2 = useQuery({
    queryKey: [field2],
    queryFn: () => get_file_from_zip(source, field2).then(blob => loadNpy(blob))})

  if(loader1.data && loader2.data) {
    function get_data(i) {
      const pos = array_pos(loader1.data, i, 1e6);
      const vec = array_pos(loader2.data, i, 1e6);
      const scale = Math.sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2);
      const target = [pos[0] + vec[0], pos[1] + vec[1], pos[2] + vec[2]];
      return {pos, target, scale}
    }
    let max_length = 0;
    const count = loader1.data.length/3;
    for (let i = 0; i < count; i++) {
      const {pos, target, scale} = get_data(i);
      if(scale > max_length)
        max_length = scale;
    }
    const [min_x, max_x] = get_extend(loader1.data, 0);
    const [min_y, max_y] = get_extend(loader1.data, 1);
    const [min_z, max_z] = get_extend(loader1.data, 2);
    const extent = [min_x, max_x, min_y, max_y, min_z, max_z];

    return {isPending: false, error: null, data: {get_data, count, max_length, extent}}
  }
  return {isPending: loader1.isPending || loader2.isPending, error: loader1.error || loader2.error, data: null}
}