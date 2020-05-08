import numpy as np
__name2__ = __name__
__name__ = "saenopy"
from .solver import Solver


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
            file_iter.nodes = np.zeros((int(numNodes), int(3)))
            for i in range(numNodes):
                line = next(file_iter)
                id, x, y, z = line.split()
                file_iter.nodes[int(id)-1, :] = [float(x), float(y), float(z)]
            return file_iter.nodes

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
            #if elementType == "4":
            #    print(data, data.min(), data.max(), np.unique(data).shape)
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
            data = np.zeros((int(numElements), 4), dtype=int)
            for i in range(numElements):
                line = next(file_iter)
                line = line.split()
                id = line[0]
                # only use thetrahedra
                if line[1] == "4":
                    a, b, c, d = line[-4:]
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
        #print("---- entity ----", entityId, entity["type"], len(entity["elements"]))
        #print(entities[entityId])
        if entity["type"] == "volume":
            #nodes = entity["nodes"]
            tetrahedra = entity["elements"]
            print(tetrahedra.min(), tetrahedra.max())
            #print("volumne", tetrahedra, entity)
    #print(entities.keys())
    #print(nodes[0])
    #print(nodes[1])
    #print(nodes[2])
    #print(nodes[3])

    if tetrahedra.shape[1] == 4:
        from .solver import Solver
        M = Solver()
        tetrahedra -= 1
        unique_indices = np.unique(tetrahedra[tetrahedra[:, 0] != -1])
        index_lookup = np.ones(max([tetrahedra.max() + 1, unique_indices.max() + 1]), dtype=int) * np.nan
        index_lookup[unique_indices] = np.arange(len(unique_indices), dtype=int)
        M.setNodes(nodes)#[unique_indices, :])
        M.setTetrahedra(tetrahedra)#index_lookup[tetrahedra])
    else:
        from .FiniteBodyForcesHex import FiniteBodyForces
        M = FiniteBodyForces()
        unique_indices = np.unique(tetrahedra)
        index_lookup = np.ones(max([tetrahedra.max()+1, unique_indices.max()+1]), dtype=int)*np.nan
        index_lookup[unique_indices] = np.arange(len(unique_indices), dtype=int)
        M.setNodes(nodes[unique_indices, :])
        M.setHexahedra(index_lookup[tetrahedra])

    return M

def ensure_array_length(arr, length):
    if arr is None:
        return np.zeros((length, 3))
    if arr.shape[0] < length:
        additional_arr = np.zeros((length - arr.shape[0], 3))
        return np.concatenate((arr, additional_arr))
    return arr

class special_file_iter:
    def __init__(self, fp):
        self.iter = iter(fp)

    def __iter__(self):
        return self

    def __next__(self):
        l = next(self.iter)
        return l.strip()

    def get_next_n_lines(self, n):
        for i in range(n):
            yield next(self.iter)

def gmsh_get_version(filename):
    with open(filename, "r") as fp:
        file_iter = special_file_iter(fp)
        file_iter.nodes = None
        for line in file_iter:
            if line == "$MeshFormat":
                line = next(file_iter)
                version, file_type, data_size = line.split()
                try:
                    version_major, version_minor = version.split(".")
                except ValueError:
                    version_major = version
                    version_minor = "0"
    return version_major, version_minor

def load_gmsh4(filename):
    nodes = None
    tetrahedra = None
    entities = {}

    def read_node_block(file_iter):
        line = next(file_iter)
        # print("read_node_block", line)
        entityTag, entityDim, parametric, numNodesInBlock = line.split()
        data = np.loadtxt(file_iter.get_next_n_lines(int(numNodesInBlock)), dtype=float).reshape(-1, 4)
        # the first part contains the indices
        indices = data[:, 0].astype(int)
        # the second the point data
        data = data[:, 1:]
        # print(indices.min(), indices.max(), indices.shape, data.shape)
        file_iter.nodes = ensure_array_length(file_iter.nodes, indices.max() + 1)
        file_iter.nodes[indices, :] = data
        return entityDim + "_" + entityTag, data, indices[0]

    def read_nodes(file_iter):
        line = next(file_iter).split()
        try:
            numEntityBlocks, numNodes, minNodeTag, maxNodeTag = line
        except ValueError:
            numEntityBlocks, numNodes = line

        # iterate over all node blocks
        for i in range(int(numEntityBlocks)):
            tag, data, first_index = read_node_block(file_iter)
            entities[tag]["nodes"] = data
            entities[tag]["first_index"] = first_index

    def read_elements_block(file_iter):
        line = next(file_iter)
        # print("read_elements_block", line)
        entityTag, entityDim, elementType, numElementsInBlock = line.split()
        nodes_per_type = {"1": 2, "2": 3, "3": 4, "4": 4, "5": 8, "6": 6, "7": 5, "8": 3, "9": 6, "10": 9, "11": 10,
                          "12": 27, "13": 18, "14": 14, "15": 1}
        if elementType in nodes_per_type.keys():
            n = nodes_per_type[elementType]
            data = np.loadtxt(file_iter.get_next_n_lines(int(numElementsInBlock)), dtype=int).reshape(-1, n + 1)[:, 1:]
            # if elementType == "4":
            #    print("44444", data, data.min(), data.max(), np.unique(data).shape, file_iter.nodes.shape)
        else:
            # ignore data
            _ = [l for l in file_iter.get_next_n_lines(int(numElementsInBlock))]
            data = None
        return entityDim + "_" + entityTag, data

    def read_elements(file_iter):
        line = next(file_iter)
        numEntityBlocks, numNodes = line.split()
        for i in range(int(numEntityBlocks)):
            tag, data = read_elements_block(file_iter)
            entities[tag]["elements"] = data

    def read_entities(file_iter):
        numPoints, numCurves, numSurfaces, numVolumes = next(file_iter).split()
        for i in range(int(numPoints)):
            line = next(file_iter)
            tag, _ = line.split(maxsplit=1)
            entities["0_" + tag] = dict(type="point")
        for i in range(int(numCurves)):
            line = next(file_iter)
            tag, _ = line.split(maxsplit=1)
            entities["1_" + tag] = dict(type="curve")
        for i in range(int(numSurfaces)):
            line = next(file_iter)
            tag, _ = line.split(maxsplit=1)
            entities["2_" + tag] = dict(type="surface")
        for i in range(int(numVolumes)):
            line = next(file_iter)
            data = line.split()
            volumeTag, minX, minY, minZ, maxX, maxY, maxZ = data[:7]
            numPhysicalTags = int(data[7])
            phys_tags = []
            for t in data[7:numPhysicalTags]:
                phys_tags.append(int(t))
            numBoundngSurfaces = int(data[7 + numPhysicalTags])
            surf_tags = []
            for t in data[7 + numPhysicalTags:7 + numPhysicalTags + numBoundngSurfaces]:
                surf_tags.append(int(t))
            entities["3_" + volumeTag] = dict(type="volume", minX=float(minX), minY=float(maxY), minZ=float(minZ),
                                              maxX=float(maxX), maxY=float(maxY), maxZ=float(maxZ),
                                              physical_tags=phys_tags, surface_tags=surf_tags)

    with open(filename, "r") as fp:
        file_iter = special_file_iter(fp)
        file_iter.nodes = None
        for line in file_iter:
            if line == "$MeshFormat":
                line = next(file_iter)
                version, file_type, data_size = line.split()
                try:
                    version_major, version_minor = version.split(".")
                except ValueError:
                    version_major = version
                    version_minor = "0"
            if line == "$Entities":
                read_entities(file_iter)
            if line == "$Nodes":
                nodes = read_nodes(file_iter)
            if line == "$Elements":
                tetrahedra = read_elements(file_iter)
        nodes = file_iter.nodes

    for entityId in entities:
        entity = entities[entityId]
        # print("---- entity ----", entityId, entity["type"], len(entity["elements"]))
        # print(entities[entityId])
        if entity["type"] == "volume":
            # nodes = entity["nodes"]
            tetrahedra = entity["elements"]
            # print(tetrahedra.min(), tetrahedra.max())
            # print("volumne", tetrahedra, entity)

    return tetrahedra, nodes


def load_gmsh(filename):
    nodes = None
    tetrahedra = None
    entities = {}

    def read_node_block(file_iter):
        line = next(file_iter)
        #print("read_node_block", line)
        entityDim, entityTag, parametric, numNodesInBlock = line.split()
        # the first part contains the indices
        indices = np.loadtxt(file_iter.get_next_n_lines(int(numNodesInBlock)), dtype=int).reshape(-1)
        # the second the point data
        data = np.loadtxt(file_iter.get_next_n_lines(int(numNodesInBlock)), dtype=float)[..., :3].reshape(-1, 3)
        #print(indices.min(), indices.max(), indices.shape, data.shape)
        file_iter.nodes = ensure_array_length(file_iter.nodes, indices.max()+1)
        file_iter.nodes[indices, :] = data
        return entityDim+"_"+entityTag, data, indices[0]

    def read_nodes(file_iter):
        if version_major == "4":
            line = next(file_iter).split()
            try:
                numEntityBlocks, numNodes, minNodeTag, maxNodeTag = line
            except ValueError:
                numEntityBlocks, numNodes = line

            # iterate over all node blocks
            for i in range(int(numEntityBlocks)):
                tag, data, first_index = read_node_block(file_iter)
                entities[tag]["nodes"] = data
                entities[tag]["first_index"] = first_index
        if version_major == "2":
            global nodes
            numNodes = int(next(file_iter))
            file_iter.nodes = np.zeros((int(numNodes), int(3)))
            for i in range(numNodes):
                line = next(file_iter)
                id, x, y, z = line.split()
                file_iter.nodes[int(id)-1, :] = [float(x), float(y), float(z)]
            return file_iter.nodes

    def read_elements_block(file_iter):
        line = next(file_iter)
        #print("read_elements_block", line)
        entityDim, entityTag, elementType, numElementsInBlock = line.split()
        nodes_per_type = {"1": 2, "2": 3, "3": 4, "4":4, "5":8, "6":6, "7":5, "8":3, "9":6, "10":9, "11":10, "12":27, "13":18, "14":14, "15": 1}
        if elementType in nodes_per_type.keys():
            n = nodes_per_type[elementType]
            data = np.loadtxt(file_iter.get_next_n_lines(int(numElementsInBlock)), dtype=int).reshape(-1, n+1)[:, 1:]
            #if elementType == "4":
            #    print("44444", data, data.min(), data.max(), np.unique(data).shape, file_iter.nodes.shape)
        else:
            # ignore data
            _ = [l for l in file_iter.get_next_n_lines(int(numElementsInBlock))]
            data = None
        return entityDim+"_"+entityTag, data

    def read_elements(file_iter):
        if version_major == "4":
            numEntityBlocks, numNodes, minNodeTag, maxNodeTag = next(file_iter).split()
            for i in range(int(numEntityBlocks)):
                tag, data = read_elements_block(file_iter)
                entities[tag]["elements"] = data

        if version_major == "2":
            numElements = int(next(file_iter))
            data = np.zeros((int(numElements), 4), dtype=int)
            for i in range(numElements):
                line = next(file_iter)
                line = line.split()
                id = line[0]
                # only use thetrahedra
                if line[1] == "4":
                    a, b, c, d = line[-4:]
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

    version_major, version_minor = gmsh_get_version(filename)

    if version_major == "4" and version_minor == "0":
        tetrahedra, nodes = load_gmsh4(filename)
    else:
        with open(filename, "r") as fp:
            file_iter = special_file_iter(fp)
            file_iter.nodes = None
            for line in file_iter:
                if line == "$MeshFormat":
                    line = next(file_iter)
                    version, file_type, data_size = line.split()
                    try:
                        version_major, version_minor = version.split(".")
                    except ValueError:
                        version_major = version
                        version_minor = "0"
                if line == "$Entities":
                    read_entities(file_iter)
                if line == "$Nodes":
                    nodes = read_nodes(file_iter)
                if line == "$Elements":
                    tetrahedra = read_elements(file_iter)
            nodes = file_iter.nodes

        for entityId in entities:
            entity = entities[entityId]
            #print("---- entity ----", entityId, entity["type"], len(entity["elements"]))
            #print(entities[entityId])
            if entity["type"] == "volume":
                #nodes = entity["nodes"]
                tetrahedra = entity["elements"]
                #print(tetrahedra.min(), tetrahedra.max())
                #print("volumne", tetrahedra, entity)

    if tetrahedra.shape[1] == 4:
        from .solver import Solver
        M = Solver()
        #tetrahedra -= 1
        #unique_indices = np.unique(tetrahedra)
        #print(nodes.shape, tetrahedra.shape, tetrahedra.min(), tetrahedra.max())
        #print(nodes[0])
        #nodes[tetrahedra]
        #index_lookup = np.ones(max([tetrahedra.max() + 1, unique_indices.max() + 1]), dtype=int) * np.nan
        #index_lookup[unique_indices] = np.arange(len(unique_indices), dtype=int)
        if 0:
            M.setNodes(nodes[unique_indices, :])
            M.setTetrahedra(index_lookup[tetrahedra])
        else:
            M.setNodes(nodes)
            M.setTetrahedra(tetrahedra)
    else:
        from .FiniteBodyForcesHex import FiniteBodyForces
        M = FiniteBodyForces()
        unique_indices = np.unique(tetrahedra)
        index_lookup = np.ones(max([tetrahedra.max()+1, unique_indices.max()+1]), dtype=int)*np.nan
        index_lookup[unique_indices] = np.arange(len(unique_indices), dtype=int)
        M.setNodes(nodes[unique_indices, :])
        M.setHexahedra(index_lookup[tetrahedra])

    return M


if __name2__ == "__main__":
    #load_gmsh(r"D:\Repositories\saenopy\david\mesh_rheology\plateplate_d20h1_s0.09.msh")
    load_gmsh(r"D:\Repositories\saenopy\david\mesh_rheology\plateplate_d20h1_s0.05.msh")
