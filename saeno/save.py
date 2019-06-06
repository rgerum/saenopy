import base64
import numpy as np
import os


def ensure_file_extension(filename, ext):
    basename, file_ext = os.path.splitext(filename)
    if file_ext != ext:
        return filename + ext
    return filename


def save_vtp(filename, M):
    """
    Export a mesh to the .vtp file format which can be opened in ParaView.

    Parameters
    ----------
    filename : str
        The file where to store the mesh.
    mesh : :py:class:`~.FiniteBodyForces.FiniteBodyForces`
        The mesh to save.
    """
    xml = """<?xml version="1.0"?>
<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian" header_type="UInt64">
<PolyData>
    <Piece NumberOfPoints="%d" NumberOfVerts="0" NumberOfLines="%d" NumberOfStrips="0" NumberOfPolys="0">
        <CellData></CellData>
        <Points>
            <DataArray Name="Points" NumberOfComponents="3" type="Float32" format="binary">
            %s
            </DataArray>
        </Points>
        <Verts>
        </Verts>
        <Lines>
            <DataArray type="UInt32" Name="connectivity" format="binary">
             %s
            </DataArray>
            <DataArray type="UInt32" Name="offsets" format="binary">
             %s
            </DataArray>
        </Lines>
        <Strips></Strips>
        <Polys></Polys>
        <PointData>
            <DataArray Name="Displacement" NumberOfComponents="3" type="Float32" format="binary">
                %s
            </DataArray>
            <DataArray Name="Force" NumberOfComponents="3" type="Float32" format="binary">
                %s
            </DataArray>
        </PointData>
    </Piece>
</PolyData>
</VTKFile>
"""
    line_pairs = set()
    for tet in M.T:
        for i in range(4):
            for j in range(4):
                t1, t2 = tet[i], tet[j]
                if t1 >= t2:
                    continue
                line_pairs.add((t1, t2))

    line_pairs = np.array(list(line_pairs)).ravel()
    n_lines = len(line_pairs) // 2

    def to_ascii(a):
        return " ".join(str(i) for i in a.ravel())

    def to_binary(a):
        byts = base64.b64encode(np.array(a.nbytes, np.uint64).tobytes() + a.tobytes())
        return repr(byts)[2:-1]

    xml = xml % (
    M.R.shape[0], n_lines, to_binary(M.R.astype("float32")), to_binary(line_pairs.astype("uint32")), to_binary((np.arange(n_lines) * 2 + 2).astype("uint32")), to_binary(M.U.astype("float32")),
    to_binary(M.f.astype("float32")))
    with open(ensure_file_extension(filename, ".vtp"), "w") as fp:
        fp.write(xml)

def save_vtu(filename, M):
    xml="""<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian" header_type="UInt64">
    <UnstructuredGrid>
        <Piece NumberOfPoints="%d" NumberOfCells="%s">
            <Points>
                <DataArray Name="Points" NumberOfComponents="3" type="Float32" format="binary">
                %s
                </DataArray>
            </Points>
            
            <PointData>
                <DataArray Name="Displacement" NumberOfComponents="3" type="Float32" format="binary">
                    %s
                </DataArray>
                <DataArray Name="Force" NumberOfComponents="3" type="Float32" format="binary">
                    %s
                </DataArray>
            </PointData>
            
            <Cells>
                <DataArray type="UInt32" Name="connectivity" format="binary">
                    %s
                </DataArray>
                <DataArray type="UInt32" Name="offsets" format="binary">
                    %s
                </DataArray>
                <DataArray type="UInt8" Name="types" format="binary">
                    %s
                </DataArray>
            </Cells>
        </Piece>
  </UnstructuredGrid>
</VTKFile>
"""
    def to_ascii(a):
        return " ".join(str(i) for i in a.ravel())

    def to_binary(a):
        byts = base64.b64encode(np.array(a.nbytes, np.uint64).tobytes() + a.tobytes())
        return repr(byts)[2:-1]

    T = M.T
    n_tets = M.T.shape[0]

    xml = xml % (
        M.R.shape[0], M.T.shape[0], to_binary(M.R.astype("float32")),
        to_binary(M.U.astype("float32")),
        to_binary(M.f.astype("float32")),
        to_binary(T.astype("uint32")),
        to_binary((np.arange(n_tets) * 4 + 4).astype("uint32")),
        to_binary((np.ones(n_tets, dtype=np.uint8) * 10).astype("uint8")),
    )
    with open(ensure_file_extension(filename, ".vtu"), "w") as fp:
        fp.write(xml)

def save_gmsh(filename, M):
    """
    Export a mesh to the .msh file format which can be opened in GMsh.

    Parameters
    ----------
    filename : str
        The file where to store the mesh.
    mesh : :py:class:`~.FiniteBodyForces.FiniteBodyForces`
        The mesh to save.
    """
    nodes = M.R
    num_nodes = nodes.shape[0]
    tets = M.T+1
    num_tets = tets.shape[0]

    with open(ensure_file_extension(filename, ".msh"), "w") as fp:
        fp.write("$MeshFormat\n")
        fp.write("4.2 0 8\n")
        fp.write("$EndMeshFormat\n")
        fp.write("$Nodes\n")
        fp.write("1 %d 1 %d\n" % (num_nodes, num_nodes))
        fp.write("3 1 0 %d\n" % num_nodes)
        for i in range(num_nodes):
            fp.write(str(i+1)+"\n")
        for node in nodes:
            fp.write(" ".join(str(i) for i in node)+"\n")
        fp.write("$EndNodes\n")
        fp.write("$Elements\n")
        fp.write("1 %d 1 %d\n" % (num_tets, num_tets))
        fp.write("3 1 4 %d\n" % num_tets)
        for i, tet in enumerate(tets):
            fp.write(str(i)+" "+" ".join(str(i) for i in tet)+"\n")
        fp.write("$EndElements\n")
