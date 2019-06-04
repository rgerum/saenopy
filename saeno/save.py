import base64
import numpy as np


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
    <VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">
    <PolyData>
        <Piece NumberOfPoints="%d" NumberOfVerts="0" NumberOfLines="%d" NumberOfStrips="0" NumberOfPolys="0">
            <CellData></CellData>
            <Points>
                <DataArray Name="Points" NumberOfComponents="3" type="Float64" format="ascii">
                %s
                </DataArray>
            </Points>
            <Verts>
            </Verts>
            <Lines>
                <DataArray type="UInt32" Name="connectivity" format="ascii">
                 %s
                </DataArray>
                <DataArray type="UInt32" Name="offsets" format="ascii">
                 %s
                </DataArray>
            </Lines>
            <Strips></Strips>
            <Polys></Polys>
            <PointData>
                <DataArray Name="Displacement" NumberOfComponents="3" type="Float64" format="ascii">
                    %s
                </DataArray>
                <DataArray Name="Force" NumberOfComponents="3" type="Float64" format="ascii">
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
        return repr(base64.b64encode(a.astype("uint8")))  # [2:-1]

    xml = xml % (
    M.R.shape[0], n_lines, to_ascii(M.R), to_ascii(line_pairs), to_ascii(np.arange(n_lines) * 2 + 2), to_ascii(M.U),
    to_ascii(M.f_glo))
    with open(filename, "w") as fp:
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

    with open(filename, "w") as fp:
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
