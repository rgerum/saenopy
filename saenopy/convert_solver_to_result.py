from saenopy.gui_deformation_whole2 import Result, format_glob
from saenopy.getDeformations import getStack, Stack
from saenopy.solver import load
import natsort

filename = r"\\131.188.117.96\biophysDS\lbischof\tif_and_analysis_backup\2021-06-02-NK92-Blebb-Rock\Analysis\2b_Regularization\{folder}\Mark_and_Find_001\median_corrected_regularized\Pos{pos}_t{t}_mesh4um_alpha_10.npz"
stack = r"\\131.188.117.96\biophysDS\lbischof\tif_and_analysis_backup\2021-06-02-NK92-Blebb-Rock\{folder}\Mark_and_Find_001\Pos{pos}_S001_t{t}_z{z}_ch00.tif"

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
