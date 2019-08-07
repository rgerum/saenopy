import numpy as np
from .FiniteBodyForces import FiniteBodyForces


def load_gmsh(filename):
    nodes = None
    tetrahedra = None
    entities = {}

    def read_node_block(file_iter):
        line = next(file_iter)
        entityDim, entityTag, parametric, numNodesInBlock = line.split()
        first_index = 0
        last_index = 1
        indices = np.zeros(int(numNodesInBlock), dtype=int)
        for i in range(int(numNodesInBlock)):
            line = next(file_iter)
            if i == 0:
                first_index = int(line)
            if i == int(numNodesInBlock)-1:
                last_index = int(line)+1
            indices[i] = int(line)
            pass
        if file_iter.nodes is None:
            file_iter.nodes = np.zeros((int(last_index), 3))
        if file_iter.nodes.shape[0] < last_index:
            file_iter.nodes = np.concatenate((file_iter.nodes, np.zeros((last_index-file_iter.nodes.shape[0], 3))))
        data = np.zeros((int(numNodesInBlock), 3))
        for i in range(int(numNodesInBlock)):
            line = next(file_iter)
            if 1:#entityDim != "0" and entityDim != "1" and en:
                data[i, :] = [float(s) for s in line.split()]
        file_iter.nodes[indices, :] = data
        return entityDim+"_"+entityTag, data, first_index

    def read_nodes(file_iter):
        if version_major == "4":
            line = next(file_iter).split()
            try:
                numEntityBlocks, numNodes, minNodeTag, maxNodeTag = line
            except ValueError:
                numEntityBlocks, numNodes = line
            for i in range(int(numEntityBlocks)):
                tag, data, first_index = read_node_block(file_iter)
                entities[tag]["nodes"] = data
                entities[tag]["first_index"] = first_index
        if version_major == "2":
            global nodes
            numNodes = int(next(file_iter))
            data = np.zeros((int(numNodes), int(3)))
            for i in range(numNodes):
                line = next(file_iter)
                id, x, y, z = line.split()
                data[int(id)-1, :] = [float(x), float(y), float(z)]
            return data

    def read_elements_block(file_iter):
        line = next(file_iter)
        entityDim, entityTag, elementType, numElementsInBlock = line.split()
        nodes_per_type = {"1": 2, "2": 3, "3": 4, "4":4, "5":8, "6":6, "7":5, "8":3, "9":6, "10":9, "11":10, "12":27, "13":18, "14":14, "15": 1}
        if elementType in nodes_per_type.keys():
            data = np.zeros((int(numElementsInBlock), nodes_per_type[elementType]), dtype=int)
            for i in range(int(numElementsInBlock)):
                line = next(file_iter)
                #if elementType == "5":
                #    print("e", line)
                data[i, :] = [int(s) for s in line.split()[1:]]
        else:
            for i in range(int(numElementsInBlock)):
                line = next(file_iter)
            data = None
        return entityDim+"_"+entityTag, data

    def read_elements(file_iter):
        if version_major == "4":
            numEntityBlocks, numNodes, minNodeTag, maxNodeTag = next(file_iter).split()
            for i in range(int(numEntityBlocks)):
                tag, data = read_elements_block(file_iter)
                #if entities[tag]["first_index"]:
                #    data -= entities[tag]["first_index"]+1
                entities[tag]["elements"] = data
        if version_major == "2":
            numElements = int(next(file_iter))
            data = np.zeros((int(numElements), 4))
            for i in range(numElements):
                line = next(file_iter)
                id, p1, p2, p3, p4, a, b, c, d = line.split()
                data[int(id)-1, :] = [int(a), int(b), int(c), int(d)]
            return data

    def read_entities(file_iter):
        numPoints, numCurves, numSurfaces, numVolumes = next(file_iter).split()
        for i in range(int(numPoints)):
            line = next(file_iter)
            tag, _ = line.split(maxsplit=1)
            entities["0_"+tag] = dict(type="point")
        for i in range(int(numCurves)):
            line = next(file_iter)
            tag, _ = line.split(maxsplit=1)
            entities["1_"+tag] = dict(type="curve")
        for i in range(int(numSurfaces)):
            line = next(file_iter)
            tag, _ = line.split(maxsplit=1)
            entities["2_"+tag] = dict(type="surface")
        for i in range(int(numVolumes)):
            line = next(file_iter)
            data = line.split()
            volumeTag, minX, minY, minZ, maxX, maxY, maxZ = data[:7]
            numPhysicalTags = int(data[7])
            phys_tags = []
            for t in data[7:numPhysicalTags]:
                phys_tags.append(int(t))
            numBoundngSurfaces = int(data[7+numPhysicalTags])
            surf_tags = []
            for t in data[7+numPhysicalTags:7+numPhysicalTags+numBoundngSurfaces]:
                surf_tags.append(int(t))
            entities["3_"+volumeTag] = dict(type="volume", minX=float(minX), minY=float(maxY), minZ=float(minZ),
                                                           maxX=float(maxX), maxY=float(maxY), maxZ=float(maxZ),
                                                           physical_tags=phys_tags, surface_tags=surf_tags)

    # def get the version and type
    with open(filename, "rb") as fp:
        file_iter = iter(fp)
        for line in file_iter:
            if line.startswith(b"$MeshFormat"):
                line = next(file_iter)
                version, file_type, data_size = line.split()
                try:
                    version_major, version_minor = version.split(b".")
                except ValueError:
                    version_major = version
                    version_minor = "0"
                if int(file_type) == 1:
                    raise IOError("Gmesh reader does not support binary format")
                if int(version_major) != 4 and int(version_major) != 2:
                    raise IOError("Gmesh file version %s not supported" % version)

    with open(filename, "r") as fp:
        file_iter = iter(fp)
        file_iter.nodes = None
        for line in file_iter:
            if line.startswith("$MeshFormat"):
                line = next(file_iter)
                version, file_type, data_size = line.split()
                try:
                    version_major, version_minor = version.split(".")
                except ValueError:
                    version_major = version
                    version_minor = "0"
            if line.startswith("$Entities"):
                read_entities(file_iter)
            if line.startswith("$Nodes"):
                nodes = read_nodes(file_iter)
            if line.startswith("$Elements"):
                tetrahedra = read_elements(file_iter)
        nodes = file_iter.nodes

    for entityId in entities:
        entity = entities[entityId]
        if entity["type"] == "volume":
            #nodes = entity["nodes"]
            tetrahedra = entity["elements"]
            #print("volumne", tetrahedra, entity)

    if tetrahedra.shape[1] == 4:
        M = FiniteBodyForces()
        M.setNodes(nodes)
        M.setTetrahedra(tetrahedra-1)
    else:
        from .FiniteBodyForcesHex import FiniteBodyForces
        M = FiniteBodyForces()
        unique_indices = np.unique(tetrahedra)
        index_lookup = np.ones(max([tetrahedra.max()+1, unique_indices.max()+1]), dtype=int)*np.nan
        index_lookup[unique_indices] = np.arange(len(unique_indices), dtype=int)
        M.setNodes(nodes[unique_indices, :])
        M.setHexahedra(index_lookup[tetrahedra])
    return M
