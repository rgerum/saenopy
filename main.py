import numpy as np
import sys
import time
import os
from cpp.configHelper import loadDefaults, loadConfigFile, parseValue, saveConfigFile
from cpp.FiniteBodyForces import FiniteBodyForces
from cpp.VirtualBeads import VirtualBeads
from cpp.buildBeams import buildBeams, saveBeams
from cpp.buildEpsilon import saveEpsilon

__version__ = 0.4

def main():
    global CFG

    if len(sys.argv) > 2 and sys.argv[1] == "-v":
        print(__version__)
        exit()

    start = time.time()
    starttotal = time.time()

    results = {}
    results["ERROR"] = ""

    # //------ START OF MODULE loadParameters --------------------------------------///

    print("LOAD PARAMETERS")

    CFG = loadDefaults()

    if len(sys.argv) > 1:
        for a in range(1, len(sys.argv), 2):
            if sys.argv[a] == "CONFIG":
                CFG.update(loadConfigFile(sys.argv[a+1]))
                os.chdir(os.path.dirname(sys.argv[a+1]))

        for a in range(1, len(sys.argv), 2):
            CFG[sys.argv[a]] = parseValue(sys.argv[a+1])

    CFG["DATAOUT"] += "_py"
    outdir = CFG["DATAOUT"]
    indir = CFG["DATAIN"]+"/"

    if not os.path.exists(outdir):
        os.mkdir(outdir)
    else:
        print("WARNING: DATAOUT directory already exists. Overwriting old results.")

    M = FiniteBodyForces(CFG)
    B = VirtualBeads(CFG)

    # //------ END OF MODULE loadParameters --------------------------------------///

    # //------ START OF MODULE buildBeams --------------------------------------///
    print("BUILD BEAMS")

    M.computeBeams(int(np.floor(np.sqrt(int(CFG["BEAMS"]) * np.pi + 0.5))))
    saveBeams(M.s, os.path.join(outdir, "beams.dat"))

    # precompute the material model
    M.computeEpsilon()

    if CFG["SAVEEPSILON"]:
        saveEpsilon(M.epsilon, os.path.join(outdir, "epsilon.dat"), CFG)
        saveEpsilon(M.epsbar, os.path.join(outdir, "epsbar.dat"), CFG)
        saveEpsilon(M.epsbarbar, os.path.join(outdir, "epsbarbar.dat"), CFG)

    # //------ END OF MODULE buildBeams --------------------------------------///

    if CFG["BOXMESH"]:
        # // ------ START OF MODULE makeBoxmesh -------------------------------------- // /

        print("MAKE BOXMESH")

        M.makeBoxmesh()

        print(M.N_c, " coords")

        #// ------ END OF MODULE makeBoxmesh - ------------------------------------- // /

    else:
        #//------ START OF MODULE loadMesh --------------------------------------///

        print("LOAD MESH")

        M.loadMeshCoords(os.path.join(indir, CFG["COORDS"]))
        M.loadMeshTets(os.path.join(indir, CFG["TETS"]))

        #//------ End OF MODULE loadMesh --------------------------------------///

    finish = time.time()
    CFG["TIME_INITIALIZATION"] = str(finish - start)
    start = time.time()

    if CFG["MODE"] == "relaxation":
        #// ------ START OF MODULE loadBoundaryConditions -------------------------------------- // /
        print("LOAD BOUNDARY CONDITIONS")

        M.computePhi()
        M.computeConnections()

        M.loadBoundaryConditions(os.path.join(indir, CFG["BCOND"]))
        M.loadConfiguration(os.path.join(indir, CFG["ICONF"]))

        #// ------ END OF MODULE loadBoundaryConditions - ------------------------------------- // /

        #//------ START OF MODULE solveBoundaryConditions --------------------------------------///
        print("SOLVE BOUNDARY CONDITIONS")

        M.relax()

        finish = time.time()
        CFG["TIME_RELAXATION"] = finish - start
        CFG["TIME_TOTALTIME"] = finish - starttotal

        #//------ END OF MODULE solveBoundaryConditions --------------------------------------///

        #//------ START OF MODULE saveResults --------------------------------------///
        print("SAVE RESULTS")

        M.storeF(os.path.join(outdir, "F.dat"))
        M.storeRAndU(os.path.join(outdir, "R.dat"), os.path.join(outdir, "U.dat"))
        M.storeEandV(os.path.join(outdir, "RR.dat"), os.path.join(outdir, "EV.dat"))
        saveConfigFile(CFG, os.path.join(outdir, "config.txt"))

        #//------ END OF MODULE saveResults --------------------------------------///
    else:
        if CFG["FIBERPATTERNMATCHING"]:
            #//------ START OF MODULE loadStacks --------------------------------------///
            print("LOAD STACKS")

            if CFG["USESPRINTF"]:
                readStackSprintf(stacka, CFG["STACKA"], int(CFG["ZFROM"]), int(CFG["ZTO"]), int(CFG["JUMP"]))
            else:
                readStackWildcard(stacka, CFG["STACKA"], int(CFG["JUMP"]))

            sX=stacka.size()
            sY=stacka[0].size()
            sZ=stacka[0][0].size()

            B=VirtualBeads(sX,sY,sZ,double(CFG["VOXELSIZEX"]),double(CFG["VOXELSIZEY"]),double(CFG["VOXELSIZEZ"])*float(CFG["JUMP"]))
            B.allBeads(M)

            if CFG["ALLIGNSTACKS"]:

                if CFG["USESPRINTF"]:
                    readStackSprintf(stackro,str(CFG["STACKR"]),int(CFG["ZFROM"]),int(CFG["ZTO"]),int(CFG["JUMP"]))
                else:
                    readStackWildcard(stackro,str(CFG["STACKR"]),int(CFG["JUMP"]))

                stackr=stack3D()

                B.Drift = B.findDriftCoarse(stackro,stacka,float(CFG["DRIFT_RANGE"]),float(CFG["DRIFT_STEP"]))
                B.Drift = B.findDrift(stackro,stacka)
                print("Drift is ",B.Drift[0]," ", B.Drift[1]," ",B.Drift[2]," before alligning stacks")

                CFG["DRIFT_FOUNDX"]=B.Drift[0]
                CFG["DRIFT_FOUNDY"]=B.Drift[1]
                CFG["DRIFT_FOUNDZ"]=B.Drift[2]

                dx=-np.floor(B.Drift[0]/B.dX+0.5)
                dy=-np.floor(B.Drift[1]/B.dY+0.5)
                dz=-np.floor(B.Drift[2]/B.dZ+0.5)

                allignStacks(stacka,stackro,stackr,dx,dy,dz)

                if CFG["SAVEALLIGNEDSTACK"]:
                    saveStack(stackr, os.path.join(outdir, "stackr"))

                stackro.clear()


            else:
                if CFG["USESPRINTF"]:
                    readStackSprintf(stackr, str(CFG["STACKR"]),int(CFG["ZFROM"]),int(CFG["ZTO"]),int(CFG["JUMP"]))
                else:
                    readStackWildcard(stackr,str(CFG["STACKR"]),int(CFG["JUMP"]))

            #//------ End OF MODULE loadStacks --------------------------------------///

            #//------ START OF MODULE extractDeformations --------------------------------------///
            print("EXTRACT DEFORMATIONS")

            B.Drift=np.zeros(3)

            if CFG["DRIFTCORRECTION"]:
                B.Drift=B.findDriftCoarse(stackr,stacka,float(CFG["DRIFT_RANGE"]),float(CFG["DRIFT_STEP"]))
                B.Drift=B.findDrift(stackr,stacka)
                print("Drift is ", B.Drif[0], " ", B.Drift[1], " ", B.Drift[2])
            elif not CFG["BOXMESH"]:
                B.loadVbeads(os.path.join(indir, CFG["VBEADS"]))

            if CFG["INITIALGUESS"]:
                B.loadGuess(M, os.path.join(indir, CFG["UGUESS"]))
            B.findDisplacements(stackr,stacka,M,float(CFG["VB_REGPARA"]))
            if CFG["REFINEDISPLACEMENTS"]:
                M.computeConnections()
                B.refineDisplacements(stackr,stacka,M,float(CFG["VB_REGPARAREF"]))

            if CFG["SUBTRACTMEDIANDISPL"]:
                B.substractMedianDisplacements()

            B.storeUfound( os.path.join(outdir, CFG["UFOUND"]), os.path.join(outdir, CFG["SFOUND"]))
            M.storeRAndU(os.path.join(outdir, "R.dat"), os.path.join(outdir, "U.dat"))

            stacka.clear()
            stackr.clear()

            finish = time.time()
            CFG["TIME_FIBERPATTERNMATCHING"] = finish - start
            start = time.time()

            # //------ END OF MODULE extractDeformations --------------------------------------///

        else:

            # //------ START OF MODULE loadDeformations --------------------------------------///
            print("LOAD DEFORMATIONS")

            B.Drift=np.zeros(3)
            B.loadUfound(os.path.join(indir, CFG["UFOUND"]),os.path.join(indir, CFG["SFOUND"]))

            if CFG["MODE"] == "computation":
                M.loadConfiguration(os.path.join(indir, CFG["UFOUND"]))

            # //------ END OF MODULE loadDeformations --------------------------------------///

        if CFG["MODE"] == "regularization":

            #//------ START OF MODULE regularizeDeformations --------------------------------------///
            print("REGULARIZE DEFORMATIONS")

            doreg = True

            if not CFG["SCATTEREDRFOUND"]:

                B.vbead.assign(M.N_c, True)

                badbeadcount=0
                goodbeadcount=0

                for c in range(M.N_c):
                    if B.S_0[c] < float(CFG["VB_MINMATCH"]):
                        B.vbead[c]= False
                        if B.S_0[c]!=0.0:
                            badbeadcount+=1
                    else:
                        goodbeadcount+=1

                doreg = goodbeadcount>badbeadcount

            doreg = True

            if doreg:

                if CFG["BOXMESH"]:
                    pass
                else:
                    M.loadBoundaryConditions( os.path.join(indir, CFG["BCOND"]))

                M.computePhi()
                M.computeConnections()
                B.computeOutOfStack(M)
                if CFG["REGMETHOD"] == "laplace":
                    M.computeLaplace()
                B.computeConconnections(M)
                if CFG["REGMETHOD"] == "laplace":
                    B.computeConconnections_Laplace(M)

                rvec=B.relax(M)

                results["MISTFIT"]=rvec[0]
                results["L"]=rvec[1]

            else:

                print("ERROR: Stacks could not be matched onto one another. Skipped regularization.")
                results["ERROR"]=results["ERROR"]+"ERROR: Stacks could not be matched onto one another. Skipped regularization."

                if CFG["BOXMESH"]:
                    pass

                else:
                    M.loadBoundaryConditions( os.path.join(indir, CFG["BCOND"]))

                M.computePhi()
                M.computeConnections()
                B.computeOutOfStack(M)
                if CFG["REGMETHOD"] == "laplace":
                    M.computeLaplace()
                B.computeConconnections(M)
                if CFG["REGMETHOD"] == "laplace":
                    B.computeConconnections_Laplace(M)

                M.updateGloFAndK()

                results["L"]="0.0"
                results["MISFIT"]="0.0"

            #//------ END OF MODULE regularizeDeformations --------------------------------------///

        else:

            if CFG["MODE"] == "computation":

                # // ------ START OF MODULE computeResults -------------------------------------- // /
                print("COMPUTE RESULTS")

                if CFG["BOXMESH"]:
                    pass

                else:
                    M.loadBoundaryConditions( os.path.join(indir, CFG["BCOND"]))

                M.computePhi()
                M.computeConnections()

                M.updateGloFAndK()

                # // ------ END OF MODULE computeResults -------------------------------------- // /


        if CFG["MODE"] != "none":

            # //------ START OF MODULE saveResults --------------------------------------///
            print("SAVE RESULTS")

            finish = time.time()
            CFG["TIME_REGULARIZATION"]= finish - start
            CFG["TIME_TOTALTIME"]=finish - starttotal

            M.storeF(os.path.join(outdir, "F.dat"))
            M.storeFden(os.path.join(outdir, "Fden.dat"))
            M.storeRAndU(os.path.join(outdir, "R.dat"),os.path.join(outdir, "U.dat"))
            M.storeEandV(os.path.join(outdir, "RR.dat"),os.path.join(outdir, "EV.dat"))
            M.storePrincipalStressAndStiffness(os.path.join(outdir, "Sbmax.dat"),os.path.join(outdir, "Sbmin.dat"),os.path.join(outdir, "WPK.dat"))
            B.storeLocalweights(os.path.join(outdir, "weights.dat"))

            M.computeStiffening(results)
            M.computeForceMoments(results)
            results["ENERGY"]=M.E_glo

            saveConfigFile(CFG,os.path.join(outdir, "config.txt"))
            saveConfigFile(results,os.path.join(outdir, "results.txt"))

            #//------ END OF MODULE saveResults --------------------------------------///


if __name__ == "__main__":
    main()