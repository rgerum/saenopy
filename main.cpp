
#define PI 3.141592653589793238462643383279502884197169399375105

#define VERSION 0.4
#define _GNU_SOURCE
#undef __STRICT_ANSI__

#include <float.h>
#include <QtGui>
#include <iostream>
#include <errno.h>
#include <iterator>
#include <fstream>
#include <vector>
#include <list>
#include <algorithm>
#include <map>
#include <set>
#include <tuple>
#include <unordered_map>
#include <time.h>
#include <ctime>
//#include <dirent.h>
#include <sys/stat.h>

#include "cpp/Vec3D.cpp"
#include "cpp/Mat3D.cpp"
#include "cpp/DRec.cpp"

#include "cpp/Chameleon.cpp"
#include "cpp/configHelper.cpp"

config CFG;

#include "cpp/buildEpsilon.cpp"
#include "cpp/buildBeams.cpp"

#include "cpp/tensorHelper.cpp"
#include "cpp/sparseHelper.cpp"
#include "cpp/multigridHelper.cpp"
#include "cpp/FiniteBodyForces.cpp"

#include "cpp/imageHelper.cpp"

#include "cpp/downhillSimplexHelper.cpp"
#include "cpp/stack3DHelper.cpp"

#include "cpp/VirtualBeads.cpp"


std::string IntToStr( int n ) { std::ostringstream result; result << n; return result.str(); }
std::string FloatToStr( float n ) {
    char floatStringBuffer[256];
    sprintf(floatStringBuffer, "%f", n);
    return std::string(floatStringBuffer);
}

int main(int argc,char *argv[])
{
    QCoreApplication a(argc, argv);

    if(std::string(argv[1])=="-v"){

        std::cout<<"SAENO "<<VERSION<<"\n";
        std::cin.ignore();
        exit(2);

    }

    clock_t start, finish, starttotal;
    start = clock();
    starttotal = clock();

    config results;
    results.clear();
    results["ERROR"]=Chameleon("");

    //------ START OF MODULE loadParameters --------------------------------------///
    std::cout<<"LOAD PARAMETERS                  \n";

    loadDefaults(CFG);

    if(argc>1) for(int a=1; a<argc; a=a+2) if(std::string(argv[a])=="CONFIG") loadConfigFile(CFG,argv[a+1],results);

    if(std::string(results["ERROR"])!=std::string(Chameleon(""))) {
        std::cin.ignore();
        exit(2);
    }

    if(argc>1) for(int a=1; a<argc; a=a+2){

        CFG[std::string(argv[a])]=Chameleon(argv[a+1]);

    }

    std::string outdir=std::string(CFG["DATAOUT"]);
    std::string indir=std::string(CFG["DATAIN"])+std::string("/");

    QDir dir(outdir.c_str());
    if (!dir.exists()) {
        dir.mkpath(".");
    }else{
        std::cout<<"WARNING: DATAOUT directory allready exists. Overwriting old results.   \n";
    }

    FiniteBodyForces M=FiniteBodyForces();

    VirtualBeads B=VirtualBeads();

    //------ END OF MODULE loadParameters --------------------------------------///


    //------ START OF MODULE buildBeams --------------------------------------///
    std::cout<<"BUILD BEAMS                  \n";

    buildBeams(M.s,floor(sqrt(int(CFG["BEAMS"])*PI+0.5)));
    M.N_b=M.s.size();
    std::cout<<M.N_b<<" beams were generated \n";

    M.computeEpsilon();

    if(bool(CFG["SAVEEPSILON"])){

        saveEpsilon(M.epsilon,(outdir+std::string("/epsilon.dat")).c_str());
        saveEpsilon(M.epsbar,(outdir+std::string("/epsbar.dat")).c_str());
        saveEpsilon(M.epsbarbar,(outdir+std::string("/epsbarbar.dat")).c_str());

    }

    //------ END OF MODULE buildBeams --------------------------------------///

    if(bool(CFG["BOXMESH"])){

        //------ START OF MODULE makeBoxmesh --------------------------------------///

        std::cout<<"MAKE BOXMESH                   \n";

        M.makeBoxmesh();

        std::cout<<M.N_c<<" coords  \n";

        //------ END OF MODULE makeBoxmesh --------------------------------------///

    }else{

        //------ START OF MODULE loadMesh --------------------------------------///

        std::cout<<"LOAD MESH                  \n";

        M.loadMeshCoords( (indir+std::string(CFG["COORDS"])).c_str() );
        M.loadMeshTets( (indir+std::string(CFG["TETS"])).c_str() );

        //------ End OF MODULE loadMesh --------------------------------------///

    }

    finish = clock();
    CFG["TIME_INITIALIZATION"]=FloatToStr((finish - start)/CLOCKS_PER_SEC );
    start = clock();

    if(std::string(CFG["MODE"])=="relaxation"){

        //------ START OF MODULE loadBoundaryConditions --------------------------------------///
        std::cout<<"LOAD BOUNDARY CONDITIONS                  \n";

        M.computePhi();
        M.computeConnections();

        M.loadBoundaryConditions( (indir+std::string(CFG["BCOND"])).c_str() );
        M.loadConfiguration( (indir+std::string(CFG["ICONF"])).c_str() );


        //------ END OF MODULE loadBoundaryConditions --------------------------------------///

        //------ START OF MODULE solveBoundaryConditions --------------------------------------///
        std::cout<<"SOLVE BOUNDARY CONDITIONS                  \n";

        M.updateGloFAndK();
        M.relax();

        finish = clock();
        CFG["TIME_RELAXATION"]=( (finish - start)/CLOCKS_PER_SEC );
        CFG["TIME_TOTALTIME"]=( (finish - starttotal)/CLOCKS_PER_SEC );

        //------ END OF MODULE solveBoundaryConditions --------------------------------------///

        //------ START OF MODULE saveResults --------------------------------------///
        std::cout<<"SAVE RESULTS                  \n";

        M.storeF((outdir+"/F.dat").c_str());
        M.storeRAndU((outdir+"/R.dat").c_str(),(outdir+"/U.dat").c_str());
        M.storeEandV((outdir+"/RR.dat").c_str(),(outdir+"/EV.dat").c_str());
        saveConfigFile(CFG,(outdir+"/config.txt").c_str());

        //------ END OF MODULE saveResults --------------------------------------///

    }else{

        if(bool(CFG["FIBERPATTERNMATCHING"])){


            //------ START OF MODULE loadStacks --------------------------------------///
            std::cout<<"LOAD STACKS                  \n";

            stack3D stacka;

            //std::cout<<"STACKA = "<<std::string(CFG["STACKA"])<<"\n";

            if(bool(CFG["USESPRINTF"])) readStackSprintf(stacka,std::string(CFG["STACKA"]),int(CFG["ZFROM"]),int(CFG["ZTO"]),int(CFG["JUMP"]));
            else readStackWildcard(stacka,std::string(CFG["STACKA"]),int(CFG["JUMP"]));

            //saveStack(stacka,outdir+std::string("/stacka"));

            //CFG["VOXELSIZEZ"]=double(CFG["VOXELSIZEZ"])*float(CFG["JUMP"]);

            int sX=stacka.size();
            int sY=stacka[0].size();
            int sZ=stacka[0][0].size();

            B=VirtualBeads(sX,sY,sZ,double(CFG["VOXELSIZEX"]),double(CFG["VOXELSIZEY"]),double(CFG["VOXELSIZEZ"])*float(CFG["JUMP"]));
            B.allBeads(M);

            stack3D stackro;
            stack3D stackr;

            if(bool(CFG["ALLIGNSTACKS"])){

                if(bool(CFG["USESPRINTF"])) readStackSprintf(stackro,std::string(CFG["STACKR"]),int(CFG["ZFROM"]),int(CFG["ZTO"]),int(CFG["JUMP"]));
                else readStackWildcard(stackro,std::string(CFG["STACKR"]),int(CFG["JUMP"]));

                stackr=stack3D();

                B.Drift=B.findDriftCoarse(stackro,stacka,float(CFG["DRIFT_RANGE"]),float(CFG["DRIFT_STEP"]));
                B.Drift=B.findDrift(stackro,stacka);
                std::cout<<"Drift is "<<B.Drift.x<<" "<<B.Drift.y<<" "<<B.Drift.z<<" before alligning stacks    \n";

                CFG["DRIFT_FOUNDX"]=B.Drift.x;
                CFG["DRIFT_FOUNDY"]=B.Drift.y;
                CFG["DRIFT_FOUNDZ"]=B.Drift.z;

                float dx=-floor(B.Drift.x/B.dX+0.5);
                float dy=-floor(B.Drift.y/B.dY+0.5);
                float dz=-floor(B.Drift.z/B.dZ+0.5);

                allignStacks(stacka,stackro,stackr,dx,dy,dz);

                if(CFG["SAVEALLIGNEDSTACK"]) saveStack(stackr,outdir+std::string("/stackr"));

                stackro.clear();


            }else{

                if(bool(CFG["USESPRINTF"])) readStackSprintf(stackr,std::string(CFG["STACKR"]),int(CFG["ZFROM"]),int(CFG["ZTO"]),int(CFG["JUMP"]));
                else readStackWildcard(stackr,std::string(CFG["STACKR"]),int(CFG["JUMP"]));

            }


            //saveStack(stackr,outdir+std::string("/stackr"));

            //------ End OF MODULE loadStacks --------------------------------------///

            //------ START OF MODULE extractDeformations --------------------------------------///
            std::cout<<"EXTRACT DEFORMATIONS                  \n";

            //B.computeConconnections(M);

            B.Drift=vec3D(0.0,0.0,0.0);

            //std::cout<<"\n\n\n"<<int(CFG["DRIFTCORRECTION"])<<"\n\n\n\n";

            if( bool(CFG["DRIFTCORRECTION"]) ) {

                //std::cout<<"\n\n\n\n";

                B.Drift=B.findDriftCoarse(stackr,stacka,float(CFG["DRIFT_RANGE"]),float(CFG["DRIFT_STEP"]));
                B.Drift=B.findDrift(stackr,stacka);
                std::cout<<"Drift is "<<B.Drift.x<<" "<<B.Drift.y<<" "<<B.Drift.z<<"       \n";
            }


            else if(!bool(CFG["BOXMESH"])) B.loadVbeads((indir+std::string(CFG["VBEADS"])).c_str());

            if(bool(CFG["INITIALGUESS"])) B.loadGuess(M,indir+std::string(CFG["UGUESS"]));
            B.findDisplacements(stackr,stacka,M,float(CFG["VB_REGPARA"]));
            if(bool(CFG["REFINEDISPLACEMENTS"])){
                M.computeConnections();
                B.refineDisplacements(stackr,stacka,M,float(CFG["VB_REGPARAREF"]));
            }
            if(bool(CFG["SUBTRACTMEDIANDISPL"])) B.substractMedianDisplacements();

            B.storeUfound( outdir+"/"+std::string(CFG["UFOUND"]),outdir+"/"+std::string(CFG["SFOUND"]));
            //B.storeUfound(outdir+"/Ufound1.dat",outdir+"/Sfound1.dat");
            M.storeRAndU(outdir+"/R.dat",outdir+"/U.dat");

            stacka.clear();
            stackr.clear();

            finish = clock();
            CFG["TIME_FIBERPATTERNMATCHING"]=( (finish - start)/CLOCKS_PER_SEC );
            start = clock();

            //------ END OF MODULE extractDeformations --------------------------------------///

        }else{

            //------ START OF MODULE loadDeformations --------------------------------------///
            std::cout<<"LOAD DEFORMATIONS                  \n";

            B.Drift=vec3D(0.0,0.0,0);
            B.loadUfound((indir+std::string("/")+std::string(CFG["UFOUND"])).c_str(),(indir+std::string("/")+std::string(CFG["SFOUND"])).c_str());

            if(std::string(CFG["MODE"])=="computation") M.loadConfiguration((indir+std::string(CFG["UFOUND"])).c_str());

            //------ END OF MODULE loadDeformations --------------------------------------///

        }

        if(std::string(CFG["MODE"])=="regularization"){

            //------ START OF MODULE regularizeDeformations --------------------------------------///
            std::cout<<"REGULARIZE DEFORMATIONS                  \n";

            bool doreg=true;

            if(!(bool(CFG["SCATTEREDRFOUND"]))){

                B.vbead.assign(M.N_c,true);

                int badbeadcount=0;
                int goodbeadcount=0;

                for(int c=0; c<M.N_c; c++){
                    if(B.S_0[c]<float(CFG["VB_MINMATCH"])){
                        B.vbead[c]=false;
                        if(B.S_0[c]!=0.0) badbeadcount++;
                    }else{
                        goodbeadcount++;
                    }
                }


                doreg=(goodbeadcount>badbeadcount);

            }

            doreg=true;

            if( doreg ){

                if(bool(CFG["BOXMESH"])){

                    //M.shrinkToSubmesh(1);
                    //setOuterSurf(int(CFG["BM_N"]),1,M.var,false);

                }else{

                    M.loadBoundaryConditions( (indir+std::string(CFG["BCOND"])).c_str() );

                }

                M.computePhi();
                M.computeConnections();
                B.computeOutOfStack(M);
                if(std::string(CFG["REGMETHOD"])=="laplace") M.computeLaplace();
                B.computeConconnections(M);
                if(std::string(CFG["REGMETHOD"])=="laplace") B.computeConconnections_Laplace(M);

                vec3D rvec=B.relax(M);

                results["MISTFIT"]=rvec[0];
                results["L"]=rvec[1];

            }else{

                std::cout<<"ERROR: Stacks could not be matched onto one another. Skipped regularization.\n";
                results["ERROR"]=Chameleon(std::string(results["ERROR"])+std::string("ERROR: Stacks could not be matched onto one another. Skipped regularization."));

                if(bool(CFG["BOXMESH"])){

                    //M.shrinkToSubmesh(1);
                    //setOuterSurf(int(CFG["BM_N"]),1,M.var,false);

                }else{

                    M.loadBoundaryConditions( (indir+std::string(CFG["BCOND"])).c_str() );

                }

                M.computePhi();
                M.computeConnections();
                B.computeOutOfStack(M);
                if(std::string(CFG["REGMETHOD"])=="laplace") M.computeLaplace();
                B.computeConconnections(M);
                if(std::string(CFG["REGMETHOD"])=="laplace") B.computeConconnections_Laplace(M);

                M.updateGloFAndK();

                results["L"]=Chameleon("0.0");
                results["MISFIT"]=Chameleon("0.0");

            }


            //------ END OF MODULE regularizeDeformations --------------------------------------///

        }else{

            if(std::string(CFG["MODE"])=="computation"){

                //------ START OF MODULE computeResults --------------------------------------///
                std::cout<<"COMPUTE RESULTS                \n";

                if(bool(CFG["BOXMESH"])){

                    //M.shrinkToSubmesh(1);
                    //setOuterSurf(int(CFG["BM_N"]),1,M.var,false);

                }else{

                    M.loadBoundaryConditions( (indir+std::string(CFG["BCOND"])).c_str() );

                }
                //for(int c=0; c<M.N_c; c++) M.U[c]=B.U_found[c];

                M.computePhi();
                M.computeConnections();

                M.updateGloFAndK();

                //------ END OF MODULE computeResults --------------------------------------///

            }

        }

        if(std::string(CFG["MODE"])!="none"){

            //------ START OF MODULE saveResults --------------------------------------///
            std::cout<<"SAVE RESULTS                  \n";

            //M.prolongToGrain(1);

            finish = clock();
            CFG["TIME_REGULARIZATION"]=( (finish - start)/CLOCKS_PER_SEC );
            CFG["TIME_TOTALTIME"]=( (finish - starttotal)/CLOCKS_PER_SEC );

            M.storeF((outdir+"/F.dat").c_str());
            M.storeFden((outdir+"/Fden.dat").c_str());
            M.storeRAndU((outdir+"/R.dat").c_str(),(outdir+"/U.dat").c_str());
            M.storeEandV((outdir+"/RR.dat").c_str(),(outdir+"/EV.dat").c_str());
            M.storePrincipalStressAndStiffness((outdir+"/Sbmax.dat").c_str(),(outdir+"/Sbmin.dat").c_str(),(outdir+"/WPK.dat").c_str());
            B.storeLocalweights((outdir+"/weights.dat").c_str());

            M.computeStiffening(results);
            //std::cout<<"computing forcemoments   \n";
            M.computeForceMoments(results);
            results["ENERGY"]=M.E_glo;

            saveConfigFile(CFG,(outdir+"/config.txt").c_str());
            saveConfigFile(results,(outdir+"/results.txt").c_str());

            //------ END OF MODULE saveResults --------------------------------------///

        }

    }

    return 0;

}
