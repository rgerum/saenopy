import base64
import numpy as np


def save_vtp(filename, M):
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
    M.R.shape[0], n_lines, to_ascii(M.R), to_ascii(line_pairs), to_ascii(np.arange(n_lines + 1) * 2 + 2), to_ascii(M.U),
    to_ascii(M.f_glo))
    with open(filename, "w") as fp:
        fp.write(xml)
