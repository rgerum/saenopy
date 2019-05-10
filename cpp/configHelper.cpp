
#include <map>
typedef std::map< std::string , Chameleon> config;

std::string trim(std::string const& source, char const* delims = " \t\r\n") {
  std::string result(source);
  std::string::size_type index = result.find_last_not_of(delims);
  if(index != std::string::npos)
    result.erase(++index);

  index = result.find_first_not_of(delims);
  if(index != std::string::npos)
    result.erase(0, index);
  else
    result.erase();
  return result;
}

void loadConfigFile(config& CFG, std::string const& filename, config& results) {

  std::ifstream file(filename.c_str());

  if(file.fail()){

      std::cout<<"ERROR in loadConfigFile: \""<<filename.c_str()<<"\" not found\n";
      results["ERROR"]=Chameleon(std::string(results["ERROR"])+std::string("ERROR in loadConfigFile: \"")+filename+std::string("\" not found\n"));

  }

  std::string line;
  std::string name;
  std::string value;
  int posEqual;
  while (std::getline(file,line)) {

    if (! line.length()) continue;

    if (line[0] == '#') continue;

    line=line.substr(0,line.find('#'));

    posEqual=line.find('=');
    name  = trim(line.substr(0,posEqual));
    value = trim(line.substr(posEqual+1));

    CFG[name]=Chameleon(value);

  }


}

void saveConfigFile(config& CFG, std::string const& filename) {

    std::ofstream fout(filename.c_str());

    config::iterator it;

    for(it=CFG.begin(); it!=CFG.end(); it++){

        fout<<(*it).first<<" = "<<std::string((*it).second)<<"\n";

    }

}

void loadDefaults(std::map<std::string, Chameleon>& CFG){

    CFG["CONFIG"]=Chameleon("");

    //Meta
    CFG["MODE"]=Chameleon("regularization"); //values: computation , regularization , relaxation
    CFG["BOXMESH"]=Chameleon(1);
    CFG["FIBERPATTERNMATCHING"]=Chameleon(1);

    //buildBeams
    CFG["BEAMS"]=Chameleon(300);
    CFG["EPSMAX"]=Chameleon("4.0");
    CFG["EPSSTEP"]=Chameleon("0.000001");
    CFG["K_0"]=Chameleon("1.0");
    CFG["D_0"]=Chameleon(10000);
    CFG["L_S"]=Chameleon("0.0");
    CFG["D_S"]=Chameleon(10000);
    CFG["SAVEEPSILON"]=Chameleon(0);

    //makeBoxmesh
    CFG["BM_GRAIN"]=Chameleon(15);
    CFG["BM_N"]=Chameleon(20);
    CFG["BM_MULOUT"]=Chameleon(1);
    CFG["BM_RIN"]=Chameleon(0);

    //loadMesh
    CFG["COORDS"]=Chameleon("coords.dat"); //remove
    CFG["TETS"]=Chameleon("tets.dat"); //remove

    //loadBoundaryConditions
    CFG["BCOND"]=Chameleon("bcond.dat"); //remove
    CFG["ICONF"]=Chameleon("iconf.dat"); //remove

    //solveBoundaryConditions
    CFG["REL_ITERATIONS"]=Chameleon(300);
    CFG["REL_CONV_CRIT"]=Chameleon("0.01");
    CFG["REL_SOLVER_STEP"]=Chameleon("0.066");
    CFG["REL_RELREC"]=Chameleon("relrec.dat"); //remove

    //loadDeformations
    CFG["UFOUND"]=Chameleon("Ufound.dat"); //remove
    CFG["SFOUND"]=Chameleon("Sfound.dat"); //remove
    CFG["RFOUND"]=Chameleon("Rfound.dat"); //remove

    //regularizeDeformations
    CFG["ALPHA"]=Chameleon("1.0");
    CFG["REGMETHOD"]=Chameleon("robust");
    CFG["ROBUSTMETHOD"]=Chameleon("huber");
    CFG["REG_LAPLACEGRAIN"]=Chameleon("15.0");
    CFG["REG_ITERATIONS"]=Chameleon(100);
    CFG["REG_CONV_CRIT"]=Chameleon("0.01");
    CFG["REG_SOLVER_STEP"]=Chameleon("0.33");
    CFG["REG_SOLVER_PRECISION"]=Chameleon("1e-18");
    CFG["REG_RELREC"]=Chameleon("relrec.dat"); //remove
    CFG["REG_SIGMAZ"]=Chameleon("1.0");

    //loadStacks
    CFG["DRIFTCORRECTION"]=Chameleon(1);
    CFG["STACKA"]=Chameleon("");
    CFG["STACKR"]=Chameleon("");
    CFG["ZFROM"]=Chameleon("");
    CFG["ZTO"]=Chameleon("");
    CFG["USESPRINTF"]=Chameleon(0);
    CFG["VOXELSIZEX"]=Chameleon(1.0);
    CFG["VOXELSIZEY"]=Chameleon(1.0);
    CFG["VOXELSIZEZ"]=Chameleon(1.0);
    CFG["JUMP"]=Chameleon(1);
    CFG["ALLIGNSTACKS"]=Chameleon("1");
    CFG["SAVEALLIGNEDSTACK"]=Chameleon("0");
    CFG["DRIFT_STEP"]=Chameleon("2.0");
    CFG["DRIFT_RANGE"]=Chameleon("30.0");

    //extractDeformation
    CFG["UGUESS"]=Chameleon("Uguess.dat");
    CFG["VBEADS"]=Chameleon("vbeads.dat");
    CFG["SUBPIXEL"]=Chameleon("0.005");
    CFG["VB_MINMATCH"]=Chameleon("0.7");
    CFG["VB_N"]=Chameleon(1);
    CFG["VB_SX"]=Chameleon(12);
    CFG["VB_SY"]=Chameleon(12);
    CFG["VB_SZ"]=Chameleon(12);
    CFG["VB_REGPARA"]=Chameleon("0.01");
    CFG["VB_REGPARAREF"]=Chameleon("0.1");
    CFG["WEIGHTEDCROSSCORR"]=Chameleon(0);
    CFG["REFINEDISPLACEMENTS"]=Chameleon(0);
    CFG["SUBTRACTMEDIANDISPL"]=Chameleon(0);

    CFG["FM_RMAX"]=Chameleon("150e-6");

    //saveResults
    CFG["DATAOUT"]=Chameleon("");
    CFG["DATAIN"]=Chameleon("");


}

