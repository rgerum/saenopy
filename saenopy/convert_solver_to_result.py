from saenopy.stack import format_glob
from saenopy.stack import Stack
from saenopy.solver import load
import natsort


"""
Option for newer saenopy version
"""
 
import os
from saenopy import Solver
from saenopy.solver import Result, Mesh
from glob import glob

liste = glob(r"../Amp*_dist*.npz")

for d in liste: 
    M = Solver.load(d)
    # add PIV deformations to mesh if applicable
    if M.U_target is not None:
        piv_mesh = Mesh(M.R,M.T, node_vars={"U_target":M.U_target})   
        Result("Converted_"+os.path.basename(d),
                        mesh_piv=[piv_mesh], 
                        solver=[M]).save()
        
    else:
        #piv_mesh = Mesh(M.R,M.T, node_vars={"U_target":np.zeros_like(M.)})   
        M.U_target = M.U
        piv_mesh = Mesh(M.R,M.T, node_vars={"U_target":M.U})   ## overwrites now.. 
        Result("Converted_"+os.path.basename(d),   
                       mesh_piv=[piv_mesh], 
                      solver=[M]).save()

        print ("No PIV found")
  
    



"""
older approaches
"""
 



filename = r"{folder}\Mark_and_Find_001\median_corrected_regularized\Pos{pos}_t{t}_mesh4um_alpha_10.npz"
stack = r"{folder}\Mark_and_Find_001\Pos{pos}_S001_t{t}_z{z}_ch00.tif"

print("glob1")
data, _ = format_glob(filename)
print("glob2")
data2, _ = format_glob(stack)
print("reformat")
data["t_int"] = data.t.astype(float)
data["pos_int"] = data.pos.astype(float)
data = data.sort_values(by='t_int')
data2["t_int"] = data2.t.astype(float)
data2["pos_int"] = data2.pos.astype(float)
for (folder, pos), datab in data.groupby(["folder", "pos_int"]):
    data2b = data2.query(f"pos_int == {pos} and folder == '{folder}'")
    stacks = []
    solver = []
    print("folder", folder, "pos", pos)

    for t, d in datab.groupby("t_int"):
        M = load(d.iloc[0].filename)
        d2 = data2b.query(f"t_int == {t}")
        d2 = d2.sort_values(by='z', key=natsort.natsort_keygen())
        #print(t, d)
        stacks.append(Stack(d2.filename, [0.2407, 0.2407, 1.0071]))
        solver.append(M)

    result = Result(f"nkoutput/{folder}/Pos{pos}_mesh4um_alpha_10.npz", stacks, time_delta=60, solver=solver)
    print("save", result.output)
    result.save()






"""
Option B
"""

# Basic approach that does not incluse all information
from glob import glob
from saenopy.solver import Mesh, Result
from saenopy.getDeformations import getStack, Stack
import saenopy
import os

# load solver files
liste = glob(r"../Amp*.npz")
for d in liste:
    M = saenopy.load(d)      
    # ad PIV data   
    piv_mesh = Mesh(M.R,M.T, node_vars={"U_target":M.U_target})   ## overwrites now.. 
    # create result object and save that
    Result("Converted_"+os.path.basename(d),   
                   mesh_piv=[piv_mesh], 
                  solver=[M]).save()
