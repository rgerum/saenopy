from pathlib import Path
from typing import Any, Union


def parseValue(value: Any) -> Union[None, int, float, str]:
    """ parse a string value to None, an int, a float or return it as a string """
    if value == "None":
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def loadConfigFile(filename: Union[str, Path]):
    """ load a config file from the given path """
    filename = Path(filename)
    results = {}
    with filename.open() as fp:
        for line in fp:
            line = line.strip()
            if len(line) == 0 or line[0] == "#":
                continue
            key, value = line.split("=")
            key = key.strip()
            value = parseValue(value.strip())
            results[key] = value
    return results


def saveConfigFile(CFG: dict, filename: Union[str, Path]):
    """ save the config to the given file """
    filename = Path(filename)
    with filename.open("w") as fp:
        for key, value in CFG.items():
            fp.write("%s = %s\n" % (key, value))


def loadDefaults():
    """ load the default config """
    CFG = {}

    CFG["CONFIG"] = ""

    # Meta
    CFG["MODE"] = "regularization"  # values: computation , regularization , relaxation
    CFG["BOXMESH"] = 1
    CFG["FIBERPATTERNMATCHING"] = 1

    # buildBeams
    CFG["BEAMS"] = 300
    CFG["EPSMAX"] = 4.0
    CFG["EPSSTEP"] = 0.000001
    CFG["K_0"] = 1.0
    CFG["D_0"] = 10000
    CFG["L_S"] = 0.0
    CFG["D_S"] = 10000
    CFG["SAVEEPSILON"] = 0

    # makeBoxmesh
    CFG["BM_GRAIN"] = 15
    CFG["BM_N"] = 20
    CFG["BM_MULOUT"] = 1
    CFG["BM_RIN"] = 0

    # loadMesh
    CFG["COORDS"] = "coords.dat"  # remove
    CFG["TETS"] = "tets.dat"  # remove

    # loadBoundaryConditions
    CFG["BCOND"] = "bcond.dat"  # remove
    CFG["ICONF"] = "iconf.dat"  # remove

    # solveBoundaryConditions
    CFG["REL_ITERATIONS"] = 300
    CFG["REL_CONV_CRIT"] = 0.01
    CFG["REL_SOLVER_STEP"] = 0.066
    CFG["REL_RELREC"] = "relrec.dat"  # remove

    # loadDeformations
    CFG["UFOUND"] = "Ufound.dat"  # remove
    CFG["SFOUND"] = "Sfound.dat"  # remove
    CFG["RFOUND"] = "Rfound.dat"  # remove

    # regularizeDeformations
    CFG["ALPHA"] = 1.0
    CFG["REGMETHOD"] = "robust"
    CFG["ROBUSTMETHOD"] = "huber"
    CFG["REG_LAPLACEGRAIN"] = 15.0
    CFG["REG_ITERATIONS"] = 100
    CFG["REG_CONV_CRIT"] = 0.01
    CFG["REG_SOLVER_STEP"] = 0.33
    CFG["REG_SOLVER_PRECISION"] = 1e-18
    CFG["REG_RELREC"] = "relrec.dat"  # remove
    CFG["REG_SIGMAZ"] = 1.0

    CFG["SCATTEREDRFOUND"] = False

    # loadStacks
    CFG["DRIFTCORRECTION"] = 1
    CFG["STACKA"] = ""
    CFG["STACKR"] = ""
    CFG["ZFROM"] = ""
    CFG["ZTO"] = ""
    CFG["USESPRINTF"] = 0
    CFG["VOXELSIZEX"] = 1.0
    CFG["VOXELSIZEY"] = 1.0
    CFG["VOXELSIZEZ"] = 1.0
    CFG["JUMP"] = 1
    CFG["ALLIGNSTACKS"] = 1
    CFG["SAVEALLIGNEDSTACK"] = 0
    CFG["DRIFT_STEP"] = 2.0
    CFG["DRIFT_RANGE"] = 30.0
    CFG["INITIALGUESS"] = ""

    # extractDeformation
    CFG["UGUESS"] = "Uguess.dat"
    CFG["VBEADS"] = "vbeads.dat"
    CFG["SUBPIXEL"] = 0.005
    CFG["VB_MINMATCH"] = 0.7
    CFG["VB_N"] = 1
    CFG["VB_SX"] = 12
    CFG["VB_SY"] = 12
    CFG["VB_SZ"] = 12
    CFG["VB_REGPARA"] = 0.01
    CFG["VB_REGPARAREF"] = 0.1
    CFG["WEIGHTEDCROSSCORR"] = 0
    CFG["REFINEDISPLACEMENTS"] = 0
    CFG["SUBTRACTMEDIANDISPL"] = 0

    CFG["FM_RMAX"] = 150e-6

    # saveResults
    CFG["DATAOUT"] = "."
    CFG["DATAIN"] = "."

    return CFG
