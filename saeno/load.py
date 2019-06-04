import numpy as np
from .FiniteBodyForces import FiniteBodyForces


def load_gmsh(filename):
    nodes = None
    tetrahedra = None
    entities = {}

    def read_node_block(file_iter):
        line = next(file_iter)
        entityDim, entityTag, parametric, numNodesInBlock = line.split()
        for i in range(int(numNodesInBlock)):
            line = next(file_iter)
            pass
        data = np.zeros((int(numNodesInBlock), int(entityDim)))
        for i in range(int(numNodesInBlock)):
            line = next(file_iter)
            data[i, :] = [float(s) for s in line.split()]
        return entityTag, data

    def read_nodes(file_iter):
        if version_major == "4":
            numEntityBlocks, numNodes, minNodeTag, maxNodeTag = next(file_iter).split()
            for i in range(int(numEntityBlocks)):
                tag, data = read_node_block(file_iter)
                #node_blocks[tag] = data
                entities[int(tag)]["nodes"] = data
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
        if elementType == "4":
            data = np.zeros((int(numElementsInBlock), 4))
            for i in range(int(numElementsInBlock)):
                line = next(file_iter)
                data[i, :] = [int(s) for s in line.split()[1:]]
        else:
            data = None
        return entityTag, data

    def read_elements(file_iter):
        if version_major == "4":
            numEntityBlocks, numNodes, minNodeTag, maxNodeTag = next(file_iter).split()
            for i in range(int(numEntityBlocks)):
                tag, data = read_elements_block(file_iter)
                #elements_blocks[tag] = data
                entities[int(tag)]["elements"] = data
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
        for i in range(int(numCurves)):
            line = next(file_iter)
        for i in range(int(numSurfaces)):
            line = next(file_iter)
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
            entities[int(volumeTag)] = dict(type="volume", minX=float(minX), minY=float(maxY), minZ=float(minZ),
                                                           maxX=float(maxX), maxY=float(maxY), maxZ=float(maxZ),
                                                           physical_tags=phys_tags, surface_tags=surf_tags)

    # def get the version and type
    with open(filename, "rb") as fp:
        file_iter = iter(fp)
        for line in file_iter:
            if line.startswith(b"$MeshFormat"):
                line = next(file_iter)
                version, file_type, data_size = line.split()
                version_major, version_minor = version.split(b".")
                if int(file_type) == 1:
                    raise IOError("Gmesh reader does not support binary format")
                if int(version_major) != 4 and int(version_major) != 2:
                    raise IOError("Gmesh file version %s not supported" % version)

    with open(filename, "r") as fp:
        file_iter = iter(fp)
        for line in file_iter:
            if line.startswith("$MeshFormat"):
                line = next(file_iter)
                version, file_type, data_size = line.split()
                version_major, version_minor = version.split(".")
            if line.startswith("$Entities"):
                read_entities(file_iter)
            if line.startswith("$Nodes"):
                nodes = read_nodes(file_iter)
            if line.startswith("$Elements"):
                tetrahedra = read_elements(file_iter)

    for entityId in entities:
        entity = entities[entityId]
        if entity["type"] == "volume":
            nodes = entity["nodes"]
            tetrahedra = entity["elements"]

    M = FiniteBodyForces()
    M.setNodes(nodes)
    M.setTetrahedra(tetrahedra-1)
    return M
