#include <string>
#include <fstream>

extern config CFG;

class FiniteBodyForces{

    public:

        std::vector< vec3D > R;                                         //coordinates ,N_c*vec3D
        std::vector< std::vector<int> > T;                              //tetrahedrons ,N_T*4*int
        std::vector< double > E;                                        //volume ,N_T*double
        std::vector< double > V;                                        //volume ,N_T*double
        std::vector< bool > var;                                        //false if the coord is fixed

        std::vector< std::vector < std::vector< double > > > Phi;       //shape tensors ,N_T*4*3

        std::vector< vec3D > U;                                         //displacements ,N_c*vec3D

        std::vector< double > f_glo;                                    //global forces ,N_c*vec3D f_glo
        std::vector< double > f_ext;                                    //external forces ,N_c*vec3D f_glo
        std::vector< std::unordered_map<size_t,mat3D> > K_glo;                     //global stiffness ,N_c*3*N_c*3*double K_glo
        std::vector< std::vector< double > > Kinv_glo;                  //inverse of the global stiffness ,N_c*3*N_c*3*double Kinv_glo

        std::vector< std::unordered_map<size_t,mat3D> > Laplace;

        double E_glo;

        std::vector< std::list<size_t> > connections;

        int currentgrain;

        int64_t N_T,N_c;

        std::vector< vec3D >  s;                                        //beams
        int N_b;

        std::vector<double> epsilon;
        std::vector<double> epsbar;
        std::vector<double> epsbarbar;
        double dlmin,dlmax,dlstep;

        int toi(int,int,int);

        void makeBoxmesh();
        void shrinkToSubmesh(int grain);
        void prolongToGrain(int grain);
        void restrictToGrain(int grain, bool interpolate);

        void loadMeshCoords(std::string fcoordsname);
        void loadMeshTets(std::string ftetname);
        void loadBeams(std::string fbeamsname);
        void loadBoundaryConditions(std::string fbcondsname);
        void loadConfiguration(std::string Uname);

        void updateRedIndex();

        void computePhi();
        void computeLaplace();
        void computeEpsilon();
        void computeEpsilon(double,double,double,double);
        void computeConnections();

        void updateGloFAndK();

        void relax();

        double solve_CG();
        double solve_J();
        void solve_nCG();
        double solve_SS();
        void smoothen();

        void precond_J3(std::vector<double>& f, std::vector<double>& u);

        void mulK(std::vector<double>& u, std::vector<double>& f);
        void mulKall(std::vector<double>& u, std::vector<double>& f);

        void computeStiffening(config&);
        void computeForceMoments(config&);

        void storeRAndU(std::string Rname, std::string Uname);
        void storeF(std::string Fname);
        void storeFden(std::string Fdenname);
        void storeEandV(std::string Rname,std::string EVname);
        void storePrincipalStressAndStiffness(std::string sbname, std::string sbminname, std::string epkname);

        FiniteBodyForces(){ currentgrain=0; };

};

void FiniteBodyForces::makeBoxmesh(){

    currentgrain=1;

    int nx=int(CFG["BM_N"]);
    //int ny=nx;
    //int nz=nx;
    double dx=double(CFG["BM_GRAIN"]);
    //double dy=dx;
    //double dz=dx;

    double rin=double(CFG["BM_RIN"]);
    double mulout=double(CFG["BM_MULOUT"]);
    double rout=nx*dx*0.5;

    if(rout<rin) std::cout<<"WARNING in makeBoxmesh: Mesh BM_RIN should be smaller than BM_MULOUT*BM_GRAIN*0.5   \n";

    makeBoxmeshCoords(dx, nx, rin, mulout, R);

    N_c=R.size();
    var.assign(N_c,true);
    U.assign(N_c,vec3D(0.0,0.0,0.0));

    var.assign(N_c,false);

    //int ny=nx;
    //int nz=nx;

    makeBoxmeshTets(nx, T, currentgrain);
    N_T=T.size();

    setActiveFields(nx,currentgrain,var,true);

    Phi.assign(N_T,
        std::vector< std::vector<double> >(4,
            std::vector<double>(3,0.0)
        )
    );

    V.assign(N_T,0.0);
    E.assign(N_T,0.0);


}

void FiniteBodyForces::loadMeshCoords(std::string fcoordsname){

    string xStr,yStr,zStr,zzStr;

    //--------------  LOAD COORDS

    // find NMX

    bool check;

    std::ifstream i_file( fcoordsname.c_str() );

    if(!i_file) std::cout<<"ERROR in loadMeshCoords: File not found !!!     \nfilename was \""<<fcoordsname<<"\"";

    int NMX=-1;
      do {
          check=(i_file >> xStr >> yStr >> zStr);
          if(check) NMX++;
          //std::cout<<i_file;
          //std::cout<<"|"<<check<<"|\n";
          if (i_file.eof()) break;
      } while(true);
      //NMX--;
      i_file.close();

    // load data

    vec3D entry;
    std::ifstream j_file( fcoordsname.c_str() );

    R.assign(NMX+1,vec3D(0.0,0.0,0.0));

    //std::cout<<NMX<<std::endl;

    for (int N=0;N<=NMX;N++) {
        j_file >> xStr >> yStr >> zStr;

        R[N]=vec3D(dOfS(xStr),dOfS(yStr),dOfS(zStr));

        //if(N==3100) std::cout<<R[N].x<<" "<<R[N].y<<" "<<R[N].z<<std::endl;
        //if(N==3100) std::cout<<xStr<<" "<<yStr<<" "<<zStr<<std::endl;

        //if(R[N].x<0.0001) R[N].x=0.0;
        //if(R[N].y<0.0001) R[N].y=0.0;
        //if(R[N].z<0.0001) R[N].z=0.0;

        //printf("N=%d: x=%9.3e, y=%9.3e",N,R[N].x,R[N].y);

        //std::cout<<N<<" "<<R[N].x<<" "<<R[N].y<<" "<<R[N].z<<std::endl;
    }//N



    j_file.close();

    //std::cout<<"check";

    printf("%s read (%d entries).\n",fcoordsname.c_str(),NMX+1);

    N_c=NMX+1;

    U.assign(NMX+1,vec3D(0.0,0.0,0.0));

    //f_glo.assign(N_c*3,0.0);
    //K_glo.assign(N_c*3,std::vector<double>(N_c*3,0.0));

    var.assign(N_c,true);
    //varmask.resize(3*N_c);

    //int i;

    K_glo.resize(N_c);
    for(int i=0; i<(N_c); i++){
        K_glo[i].clear();
        K_glo[i].rehash(27);
    }

    f_glo.resize(3*N_c);
    f_ext.resize(3*N_c);

}

void FiniteBodyForces::loadMeshTets(std::string ftetsname){

    string xStr,yStr,zStr,zzStr;

    //--------------  LOAD COORDS

    // find NMX

    bool check;

    std::ifstream i_file( ftetsname.c_str() );

    if(!i_file) std::cout<<"ERROR in loadMeshTets: File not found !!!     \nfilename was \""<<ftetsname<<"\"";

    int NMX=-1;
    do {
      check=(i_file >> xStr >> yStr >> zStr >> zStr);
      if(check) NMX++;
      if (i_file.eof()) break;
    } while(true);
    //NMX--;
    i_file.close();

    // load data

    vec3D entry;
    std::ifstream j_file( ftetsname.c_str() );

    T.assign(NMX+1,std::vector<int>(4,0));

    //std::cout<<"!!!!"<<NMX<<std::endl;

    for (int N=0;N<=NMX;N++) {
        j_file >> xStr >> yStr >> zStr >> zzStr;

        T[N][0]=(int)dOfS(xStr)-1;
        T[N][1]=(int)dOfS(yStr)-1;
        T[N][2]=(int)dOfS(zStr)-1;
        T[N][3]=(int)dOfS(zzStr)-1;
        //printf("N=%d: x=%9.3e, y=%9.3e",N,R[N].x,R[N].y);

        //std::cout<<N<<" "<<T[N][0]<<" "<<T[N][1]<<" "<<T[N][2]<<" "<<T[N][3]<<std::endl;
    }//N



    j_file.close();

    //std::cout<<"check";

    printf("%s read (%d entries).\n",ftetsname.c_str(),NMX+1);

    N_T=NMX+1;

    Phi.assign(N_T,
        std::vector< std::vector<double> >(4,
            std::vector<double>(3,0.0)
        )
    );

    //f_tet.assign(N_T, std::vector< vec3D >(4, vec3D(0.0,0.0,0.0)));

    //K_tet.assign(N_T, std::vector< std::vector <mat3D> >(4,std::vector<mat3D>(4,mat3D(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0))));

    V.assign(N_T,0.0);
    E.assign(N_T,0.0);

}

void FiniteBodyForces::loadBeams(std::string fbeamsname){

    DRec3D srec=DRec3D();

    srec.read(fbeamsname.c_str());

    s=srec.data;

    N_b=s.size();

}

void FiniteBodyForces::loadBoundaryConditions(std::string fbcondsname){

    std::vector< std::vector<double> > temp;

    readFromDatFile(fbcondsname.c_str(),temp);

    int i=0;
    int iend=temp.size();

    for(i=0; i<iend; i++){


        //U[N]=vec3D(temp[i][0],temp[i][1],temp[i][2]);

        var[i]=(temp[i][3]>0.5);

        //std::cout<<temp[i][0]<<" , "<<temp[i][1]<<" , "<<temp[i][2]<<" , "<<temp[i][3]<<" \n";

        if(!var[i]) U[i]=vec3D(temp[i][0],temp[i][1],temp[i][2]);
        else{

            f_ext[3*i]=temp[i][0];
            f_ext[3*i+1]=temp[i][1];
            f_ext[3*i+2]=temp[i][2];

        }

    }


}

void FiniteBodyForces::loadConfiguration(std::string Uname){

    DRec3D Urec=DRec3D();

    Urec.read(Uname.c_str());

    for(int c=0; c<N_c; c++) U[c]=Urec.data[c];

    //N_b=s.size();


}

void FiniteBodyForces::computeConnections(){

    unsigned int tt,t1,t2,c1,c2;

    connections.resize(N_c);

    for(c1=0;c1<N_c;c1++){

        connections[c1].clear();

    }

    std::list<size_t>::iterator it;

    bool found=false;

    for(tt=0; tt<N_T; tt++){

        for(t1=0;t1<4;t1++) for(t2=0;t2<4;t2++){

            c1=T[tt][t1];
            c2=T[tt][t2];

            found=false;

            for(it=connections[c1].begin(); (it!=connections[c1].end() && !found); it++){

                if(*it==c2) found=true;

            }

            //it=connections[c1].end();
            //if(*it==c2) found=true;

            if(!found) connections[c1].push_back(c2);
            //if(!found) std::cout<<c1<<"\n";

        }

    }

    for(c1=0;c1<N_c;c1++) connections[c1].sort();

}

void FiniteBodyForces::computePhi(){


    int t;

    std::vector< std::vector <double> > X,Y,Z,Chi;

    X.assign(4,std::vector<double>(4,0.0));
    Y.assign(4,std::vector<double>(4,0.0));
    Z.assign(4,std::vector<double>(4,0.0));

    Chi.assign(4,std::vector<double>(3,0.0));

    Chi[0][0]=-1.0;
    Chi[0][1]=-1.0;
    Chi[0][2]=-1.0;

    Chi[1][0]=1.0;
    Chi[1][1]=0.0;
    Chi[1][2]=0.0;

    Chi[2][0]=0.0;
    Chi[2][1]=1.0;
    Chi[2][2]=0.0;

    Chi[3][0]=0.0;
    Chi[3][1]=0.0;
    Chi[3][2]=1.0;

    vec3D b,c,d,RR;

    mat3D A,B,Binv;

    //bool ordered=false;

    //int k,l;

    //double u,v,w,UU,VV,WW;

    for(t=0; t<N_T; t++){

        //std::cout<<t<<"\n";

        B=mat3D(

            R[T[t][1]].x-R[T[t][0]].x, R[T[t][2]].x-R[T[t][0]].x, R[T[t][3]].x-R[T[t][0]].x,
            R[T[t][1]].y-R[T[t][0]].y, R[T[t][2]].y-R[T[t][0]].y, R[T[t][3]].y-R[T[t][0]].y,
            R[T[t][1]].z-R[T[t][0]].z, R[T[t][2]].z-R[T[t][0]].z, R[T[t][3]].z-R[T[t][0]].z

        );

        //RR=(R[T[t][0]]+R[T[t][1]]+R[T[t][2]]+R[T[t][3]])*0.25;


        V[t]=abs(B.det())/6.0;

        if(V[t]!=0.0) {

            Binv=B.inv();

            Phi[t]=Chi*Binv;

        }

        if(false){

            b=R[T[t][1]]-R[T[t][0]];
            c=R[T[t][2]]-R[T[t][0]];
            d=R[T[t][3]]-R[T[t][0]];

            std::cout<<"| now at "<<t<<std::endl<<std::endl;

            std::cout<<T[t][0]<<" "<<T[t][1]<<" "<<T[t][2]<<" "<<T[t][3]<<" "<<std::endl;

            R[T[t][0]].print2();
            R[T[t][1]].print2();
            R[T[t][2]].print2();
            R[T[t][3]].print2();

            std::cout<<std::endl;

            std::cout<<std::endl;

            b.print2();
            c.print2();
            d.print2();

            std::cout<<std::endl;

            B.print();

            std::cout<<Phi[t][0][0]<<" "<<Phi[t][0][1]<<" "<<Phi[t][0][2]<<std::endl;
            std::cout<<Phi[t][1][0]<<" "<<Phi[t][1][1]<<" "<<Phi[t][1][2]<<std::endl;
            std::cout<<Phi[t][2][0]<<" "<<Phi[t][2][1]<<" "<<Phi[t][2][2]<<std::endl;
            std::cout<<Phi[t][3][0]<<" "<<Phi[t][3][1]<<" "<<Phi[t][3][2]<<std::endl;

            std::cout<<V[t]<<std::endl<<"-------"<<std::endl;

        }


    }



}

void FiniteBodyForces::computeLaplace(){

    Laplace.resize(N_c);
    for(int i=0; i<(N_c); i++){
        Laplace[i].clear();
        Laplace[i].rehash(27);
    }

    mat3D Idmat=mat3D(1.0,0.0,0.0 ,0.0,1.0,0.0 ,0.0,0.0,1.0);

    for(size_t c=0; c<N_c; c++){

        for(std::list<size_t>::iterator it=connections[c].begin(); it!=connections[c].end(); it++) if(*it!=c){

            double r=norm(R[c]-R[*it]);

            Laplace[c][*it]+=Idmat*(-1.0/r);
            Laplace[c][c]+=Idmat*(1.0/r);


            //Laplace[c][*it].print();

        }


    }


}

void FiniteBodyForces::computeEpsilon(){

    //double k0=0.9;
    double k1=double(CFG["K_0"]);


    double ds0=double(CFG["D_0"]);
    double s1=double(CFG["L_S"]);
    double ds1=double(CFG["D_S"]);


    buildEpsilon(epsilon,epsbar,epsbarbar,k1,ds0,s1,ds1);

    dlmin=-1.0;
    dlmax=double(CFG["EPSMAX"]);
    dlstep=double(CFG["EPSSTEP"]);

}

void FiniteBodyForces::computeEpsilon(double k1, double ds0, double s1, double ds1){

    buildEpsilon(epsilon,epsbar,epsbarbar,k1,ds0,s1,ds1);

    dlmin=-1.0;
    dlmax=double(CFG["EPSMAX"]);
    dlstep=double(CFG["EPSSTEP"]);

}

void FiniteBodyForces::updateGloFAndK(){

    bool countEnergy;

    E_glo=0.0;

    int64_t tt=0;

    mat3D F;

    double dEdsbar,dEdsbarbar,deltal,sstarsstar,epsilon_b,epsbar_b,epsbarbar_b,dli;

    vec3D s_bar;

    int t,b,t1,t2,li,c1,c2;

    std::vector< std::vector<double> > u_T(3,std::vector<double>(4,0.0));
    std::vector<double> s_star=std::vector<double>(4,0.0);
    std::vector< std::vector<double> > FF(3,std::vector<double>(3,0.0));

    std::unordered_map<size_t,mat3D>::iterator itm;

    K_glo.resize(N_c);
    for(int i=0; i<(N_c); i++) for(itm=K_glo[i].begin(); itm!=K_glo[i].end(); itm++) itm->second=mat3D(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0);

    f_glo.assign(3*N_c, 0.0);

    for(tt=0; tt<N_T; tt++){

        E[tt]=0.0;

        if((tt%100)==0) std::cout<<"Updating f and K "<<(floor((tt/(N_T+0.0))*1000)+0.0)/10.0<<"%          \r";

        if( var[T[tt][0]] || var[T[tt][1]] || var[T[tt][2]] || var[T[tt][3]] ) countEnergy=true;
        else countEnergy=false;

        for(t=0; t<4; t++){

            u_T[0][t]=U[T[tt][t]].x;
            u_T[1][t]=U[T[tt][t]].y;
            u_T[2][t]=U[T[tt][t]].z;

        }

        FF=u_T*Phi[tt];

        F=mat3D(

            FF[0][0]+1.0,FF[0][1],FF[0][2],
            FF[1][0],FF[1][1]+1.0,FF[1][2],
            FF[2][0],FF[2][1],FF[2][2]+1.0

            );

        for(b=0;b<N_b;b++){

            s_bar=F*s[b];
            s_star=Phi[tt]*s[b];

            deltal=abs(s_bar)-1.0;

            li=floor( (deltal-dlmin)/dlstep );
            dli=(deltal-dlmin)/dlstep-li;

            if( li>((dlmax-dlmin)/dlstep)-2 ) {
                li=((dlmax-dlmin)/dlstep)-2;
                dli=0;
            }

            epsilon_b=(1-dli)*epsilon[li]+dli*epsilon[li+1];
            epsbar_b=(1-dli)*epsbar[li]+dli*epsbar[li+1];
            epsbarbar_b=(1-dli)*epsbarbar[li]+dli*epsbarbar[li+1];


            if(countEnergy) E_glo+=epsilon_b*V[tt]/N_b;



            E[tt]+=epsilon_b*V[tt]/N_b;

            dEdsbar=-1.0*(epsbar_b/(deltal+1.0))/(N_b+0.0)*V[tt];

            dEdsbarbar=( ((deltal+1.0)*epsbarbar_b-epsbar_b)/((deltal+1.0)*(deltal+1.0)*(deltal+1.0)) )/(N_b+0.0)*V[tt];

            for(t1=0;t1<4;t1++){


                c1=T[tt][t1];

                f_glo[3*c1+0]+=s_star[t1]*s_bar.x*(dEdsbar);
                f_glo[3*c1+1]+=s_star[t1]*s_bar.y*(dEdsbar);
                f_glo[3*c1+2]+=s_star[t1]*s_bar.z*(dEdsbar);


                for(t2=0; t2<4; t2++){

                    c2=T[tt][t2];

                    sstarsstar=s_star[t1]*s_star[t2];

                    K_glo[c1][c2]+=mat3D(
                        sstarsstar*0.5*( dEdsbarbar*s_bar.x*s_bar.x-dEdsbar ),
                        sstarsstar*0.5*( dEdsbarbar*s_bar.x*s_bar.y ),
                        sstarsstar*0.5*( dEdsbarbar*s_bar.x*s_bar.z ),

                        sstarsstar*0.5*( dEdsbarbar*s_bar.y*s_bar.x ),
                        sstarsstar*0.5*( dEdsbarbar*s_bar.y*s_bar.y-dEdsbar),
                        sstarsstar*0.5*( dEdsbarbar*s_bar.y*s_bar.z ),

                        sstarsstar*0.5*( dEdsbarbar*s_bar.z*s_bar.x ),
                        sstarsstar*0.5*( dEdsbarbar*s_bar.z*s_bar.y ),
                        sstarsstar*0.5*( dEdsbarbar*s_bar.z*s_bar.z-dEdsbar )
                    );
                }
            }
        }

        if(false){

            F.print();

            std::cout<<std::endl;

            R[T[tt][3]].print2();

            std::cout<<std::endl;

            std::cout<<Phi[tt][0][0]<<" "<<Phi[tt][0][1]<<" "<<Phi[tt][0][2]<<std::endl;
            std::cout<<Phi[tt][1][0]<<" "<<Phi[tt][1][1]<<" "<<Phi[tt][1][2]<<std::endl;
            std::cout<<Phi[tt][2][0]<<" "<<Phi[tt][2][1]<<" "<<Phi[tt][2][2]<<std::endl;
            std::cout<<Phi[tt][3][0]<<" "<<Phi[tt][0][1]<<" "<<Phi[tt][3][2]<<std::endl<<std::endl;

            std::cout<<dEdsbar<<" "<<dEdsbarbar<<std::endl;

        }

    }


}

void FiniteBodyForces::relax(){

    int i=0;
    int i_max=int(CFG["REL_ITERATIONS"]);
    double du;

    std::string outdir=std::string(CFG["DATAOUT"]);
    std::string relrecname=outdir+std::string("/")+std::string(CFG["REL_RELREC"]);

    DRec3D relrec=DRec3D();

    std::clock_t start, finish;
    start = clock();

    for( i=0; i<i_max; i++){


        du=solve_CG();
        //if(du<double(CFG["REL_CONV_CRIT"]) && i>9) i=i_max;
        updateGloFAndK();

        double ff=0.0;

        for(int ii=0; ii<N_c; ii++) if(var[ii]) ff+=norm(vec3D( f_glo[3*ii], f_glo[3*ii+1], f_glo[3*ii+2] ));

        std::cout<<"Newton "<<i<<": du="<<du<<"  Energy="<<E_glo<<"  Residuum="<<ff<<"          \n";

        relrec.record(vec3D(du,E_glo,ff));
        relrec.store(relrecname.c_str());


        if(i>6){

            int s=relrec.data.size();

            double Emean=0.0,cc=0.0;

            for(int ii=(s-1); ii>s-6; ii--){

                Emean+=relrec.data[ii][1];
                cc=cc+1.0;

            }

            Emean=Emean/cc;

            double Estd=0.0;
            cc=0.0;

            for(int ii=(s-1); ii>s-6; ii--){

                Estd+=(Emean-relrec.data[ii][1])*(Emean-relrec.data[ii][1]);
                cc=cc+1.0;

            }

            Estd=sqrt(Estd)/cc;

            if( (Estd/Emean)<double(CFG["REL_CONV_CRIT"]) ) i=i_max;
        }

    }

    finish=clock();
    std::cout<<"| time for relaxation was "<<(finish-start)/1000.0<<"                      \r";

}

double FiniteBodyForces::solve_CG(){

    //b=f_glo
    //A=K_glo

    double tol=0.00001;

    int maxiter=3*N_c;

    double resid,rsnew,alpha; //alpha_0,beta_0,rho_0,rho1_0,

    std::vector<double> pp, zz, qq, rr, kk, uu, ff, fff, Ap;

    pp.assign(3*N_c,0.0);
    Ap.assign(3*N_c,0.0);
    zz.assign(3*N_c,0.0);
    qq.assign(3*N_c,0.0);
    rr.assign(3*N_c,0.0);
    kk.assign(3*N_c,0.0);
    uu.assign(3*N_c,0.0);
    ff.assign(3*N_c,0.0);
    fff.assign(3*N_c,0.0);

    plusminus(f_glo,-1.0,f_ext,ff);

    int i=0;

    for(i=0; i<N_c; i++) if(!var[i]) {

        ff[3*i]=0.0;
        ff[3*i+1]=0.0;
        ff[3*i+2]=0.0;

    }

    double normb=imul(ff,ff);

    //std::cout<<ff[0]<<" "<<ff[1]<<" "<<ff[2]<<" "<<ff[3]<<" "<<ff[4]<<" "<<ff[5]<<" "<<ff[6]<<std::endl;

    if(normb>0.0){

        mulK(uu,kk);

        plusminus(ff,-1.0,kk,rr);

        //if(normb==0.0) normb=1.0;

        //std::cout<<"computin resid fist stime"<<std::endl;

        setequal(rr,pp);

        resid=imul(pp,pp);


        //std::cout<<i<<": "<<resid<<std::endl;
        for( i = 1; i <= maxiter; i++){

            mulK(pp,Ap);

            alpha=resid/imul(pp,Ap);

            plusminus(uu,alpha,pp,uu);
            plusminus(rr,-alpha,Ap,rr);

            rsnew=imul(rr,rr);

            if( rsnew<tol*normb) break;

            plusminus(rr,rsnew/resid,pp,pp);

            resid=rsnew;

            if( (i%100)==0 ) std::cout<<i<<": "<<resid<<" alpha="<<alpha<<"\r";

        }

        int c;

        vec3D dU;

        double stepper=double(CFG["REL_SOLVER_STEP"]);
        double du=0.0;

        for(c=0; c<N_c; c++) if(var[c]){

            dU=vec3D(uu[3*c],uu[3*c+1],uu[3*c+2]);

            //if(abs(dU)>1.666) dU=dU/abs(dU)*1.666;

            U[c]+=dU*stepper;
            du+=norm(dU)*stepper*stepper;

        }

        return du;

    }else return 0.0;



}

void FiniteBodyForces::smoothen(){

    mat3D A;
    vec3D f,du;

    double ddu=0.0;

    int c;
    for(c=0; c<N_c; c++) if(var[c]){

        A=K_glo[c][c];

        f=vec3D( f_glo[3*c+0],f_glo[3*c+1],f_glo[3*c+2] );

        du=(A.inv())*f;

        U[c]+=du;

        ddu+=norm(du);

        //A.print();


    }

    std::cout<<"du="<<ddu<<"\n";


}

void FiniteBodyForces::mulK(std::vector<double>& u, std::vector<double>& f){

    int c1,c2;

    vec3D ff,uu;
    mat3D A;

    std::list<size_t>::iterator it;

    for(c1=0; c1<N_c; c1++){

        ff=vec3D(0.0,0.0,0.0);

        for(it=connections[c1].begin(); it!=connections[c1].end(); it++){

            c2=*it;


            if(var[c1]){

                uu=vec3D( u[3*c2],u[3*c2+1],u[3*c2+2] );

                A=K_glo[c1][c2];

                ff+=A*uu;

                //if(c1==c2) K_glo[c1][c2].print();

            }

        }

        f[3*c1]=ff.x;
        f[3*c1+1]=ff.y;
        f[3*c1+2]=ff.z;

    }



}

void FiniteBodyForces::mulKall(std::vector<double>& u, std::vector<double>& f){

    int c1,c2;

    vec3D ff,uu;
    mat3D A;

    std::list<size_t>::iterator it;

    for(c1=0; c1<N_c; c1++){

        ff=vec3D(0.0,0.0,0.0);

        for(it=connections[c1].begin(); it!=connections[c1].end(); it++){

            c2=*it;


            //if(var[c1]){

                uu=vec3D( u[3*c2],u[3*c2+1],u[3*c2+2] );

                A=K_glo[c1][c2];

                ff+=A*uu;

                //if(c1==c2) K_glo[c1][c2].print();

            //}

        }

        f[3*c1]=ff.x;
        f[3*c1+1]=ff.y;
        f[3*c1+2]=ff.z;

    }



}

void FiniteBodyForces::computeStiffening(config& results){

    std::vector<double> uu, Ku;

    uu.assign(3*N_c,0.0);
    Ku.assign(3*N_c,0.0);

    int i=0;

    for(i=0; i<N_c; i++) {

        uu[3*i]=U[i].x;
        uu[3*i+1]=U[i].y;
        uu[3*i+2]=U[i].z;

    }

    mulK(uu,Ku);

    double kWithStiffening=imul(uu,Ku);



    //double k0=0.9;
    double k1=double(CFG["K_0"]);


    double ds0=double(CFG["D_0"]);


    buildEpsilon(epsilon,epsbar,epsbarbar,k1,ds0,0,0);



    updateGloFAndK();

    uu.assign(3*N_c,0.0);
    Ku.assign(3*N_c,0.0);

    for(i=0; i<N_c; i++) {

        uu[3*i]=U[i].x;
        uu[3*i+1]=U[i].y;
        uu[3*i+2]=U[i].z;

    }

    mulK(uu,Ku);

    double kWithoutStiffening=imul(uu,Ku);


    results["STIFFENING"]=kWithStiffening/kWithoutStiffening;

    computeEpsilon();

}

void FiniteBodyForces::computeForceMoments(config& results){

    double rmax=double(CFG["FM_RMAX"]);

    std::vector<bool> in;

    in.assign(N_c,false);

    vec3D Rcms=vec3D(0.0,0.0,0.0);

    vec3D fsum=vec3D(0.0,0.0,0.0);

    //std::cout<<"check 01   \n";

    vec3D B=vec3D(0.0,0.0,0.0);
    vec3D B1=vec3D(0.0,0.0,0.0);
    vec3D B2=vec3D(0.0,0.0,0.0);
    mat3D A=mat3D(0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0);

    vec3D f;

    mat3D I=mat3D(1.0,0.0,0.0, 0.0,1.0,0.0, 0.0,0.0,1.0);

    for(int c=0; c<N_c; c++) if(abs(R[c])<rmax){

        in[c]=true;

        fsum+=f;

        f=vec3D( f_glo[3*c], f_glo[3*c+1], f_glo[3*c+2] );

        B1+=R[c]*norm(f);
        B2+=f*R[c].Dot(f);

        //f.print2();

        A+=I*norm(f)-Outer(f,f);

    }

    B=B1-B2;

    //std::cout<<"A = \n";
    //A.print();

    //std::cout<<"A.det = "<<A.det()<<"\n";

    //std::cout<<"A.inv = \n";
    //A.inv().print();

    //std::cout<<"\nB1 = \n";
    //B1.print2();

    //std::cout<<"\nB2 = \n";
    //B2.print2();

    //std::cout<<"\nB = \n";
    //B.print2();

    Rcms=A.inv()*B;
    //std::cout<<"\nCMS = \n";
    //Rcms.print2();

    /*
    for(int c=0; c<N_c; c++) if(abs(R[c])<rmax){

        in[c]=true;

        fn[c]=norm(vec3D( f_glo[3*c],f_glo[3*c+1],f_glo[3*c+2] ));

        Rcms+=R[c]*fn[c];

        f2s+=fn[c];

        fsum+=vec3D( f_glo[3*c],f_glo[3*c+1],f_glo[3*c+2] );

    }
    */


    results["FSUM_X"]=fsum.x;
    results["FSUM_Y"]=fsum.y;
    results["FSUM_Z"]=fsum.z;
    results["FSUMABS"]=abs(fsum);

    results["CMS_X"]=Rcms.x;
    results["CMS_Y"]=Rcms.y;
    results["CMS_Z"]=Rcms.z;

    mat3D M;
    M=mat3D(0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0);
    //std::cout<<"check 02   \n";

    double contractility=0.0;

    for(int c=0; c<N_c; c++) if(in[c]){

        vec3D RR=R[c]-Rcms;

        /*

        M.xx+=fn[c]*(RR.y*RR.y+RR.z*RR.z);
        M.yy+=fn[c]*(RR.x*RR.x+RR.z*RR.z);
        M.zz+=fn[c]*(RR.x*RR.x+RR.y*RR.y);

        M.xy+=fn[c]*(RR.x*RR.y);
        M.yx+=fn[c]*(RR.y*RR.x);

        M.yz+=fn[c]*(RR.y*RR.z);
        M.zy+=fn[c]*(RR.z*RR.y);

        M.zx+=fn[c]*(RR.z*RR.x);
        M.xz+=fn[c]*(RR.x*RR.z);

        */

        contractility+=RR.Dot(vec3D( f_glo[3*c],f_glo[3*c+1],f_glo[3*c+2] ))/abs(RR);

    }

    results["CONTRACTILITY"]=contractility;

    std::vector<vec3D> vecs;

    buildBeams(vecs, 150);

    double fmax=0.0,fmin=0.0,mmax=0.0,mmin=0.0;
    int bmax=0,bmin=0;

    for(unsigned int b=0; b<vecs.size(); b++){

        double ff=0.0,mm=0.0; //s1=0.0,

        for(int c=0; c<N_c; c++) if(in[c]){

            vec3D RR=R[c]-Rcms;
            vec3D eR=RR/abs(RR);

            ff+=( eR.Dot(vecs[b]) )*( vecs[b].Dot( vec3D( f_glo[3*c],f_glo[3*c+1],f_glo[3*c+2] )) );
            mm+=( RR.Dot(vecs[b]) )*( vecs[b].Dot( vec3D( f_glo[3*c],f_glo[3*c+1],f_glo[3*c+2] )) );

        }

        if(mm>mmax || b==0){

            bmax=b;
            fmax=ff;
            mmax=mm;

        }

        if(mm<mmin || b==0){

            bmin=b;
            fmin=ff;
            mmin=mm;

        }

    }

    vec3D vmid=vecs[bmax].CrossProduct(vecs[bmin]);
    vmid=vmid/abs(vmid);

    double fmid=0.0,mmid=0.0; //signum=0.0,

    for(int c=0; c<N_c; c++) if(in[c]){

        vec3D RR=R[c]-Rcms;
        vec3D eR=RR/abs(RR);

        fmid+=( eR.Dot(vmid) )*( vmid.Dot( vec3D( f_glo[3*c],f_glo[3*c+1],f_glo[3*c+2] )) );
        mmid+=( RR.Dot(vmid) )*( vmid.Dot( vec3D( f_glo[3*c],f_glo[3*c+1],f_glo[3*c+2] )) );

    }

    results["FMAX"]=fmax;
    results["MMAX"]=mmax;
    results["VMAX_X"]=vecs[bmax].x;
    results["VMAX_Y"]=vecs[bmax].y;
    results["VMAX_Z"]=vecs[bmax].z;

    results["FMID"]=fmid;
    results["MMID"]=mmid;
    results["VMID_X"]=vmid.x;
    results["VMID_Y"]=vmid.y;
    results["VMID_Z"]=vmid.z;

    results["FMIN"]=fmin;
    results["MMIN"]=mmin;
    results["VMIN_X"]=vecs[bmin].x;
    results["VMIN_Y"]=vecs[bmin].y;
    results["VMIN_Z"]=vecs[bmin].z;

    results["POLARITY"]=fmax/contractility;

    /*
    //std::cout<<"check 03   \n";

    int N=3;

    int I,J;
    complexT **A,**x;
    double **evec;
    A=NRmatrix<complexT>(1,N,1,N);
    x=NRmatrix<complexT>(1,N,1,N);
    evec=NRmatrix<double>(1,N,1,N);

    for (I=1;I<=N;I++) {
    for (J=1;J<=N;J++) {
      A[I][J]=complexT(M[I][J],0.0);
    }//J
    }//I

    //std::cout<<"check 04   \n";

    double *eval;
    eval=NRvector<double>(1,3);
    DiagHermMatrix(A,x,eval,N);

    //cprimat("C-Eigenvectors:\n",x,3);

    for (I=1;I<=N;I++) {
    for (J=1;J<=N;J++) {
      evec[I][J]=real(x[I][J]);
    }//J
    }//I

    //std::cout<<"check 05   \n";

    //evec[K][I] is the vector component K of eigenvector nr. I

    free_NRmatrix<complexT>(A,1,N,1,N);
    free_NRmatrix<complexT>(x,1,N,1,N);

    double shape=max(eval[1],max(eval[2],eval[3]))/min(eval[1],min(eval[2],eval[3]));

    results["SHAPE"]=shape;

    //std::cout<<"check 06   \n";

    vec3D vv1=vec3D(evec[1][1],evec[2][1],evec[3][1]);
    vec3D vv2=vec3D(evec[1][2],evec[2][2],evec[3][2]);
    vec3D vv3=vec3D(evec[1][3],evec[2][3],evec[3][3]);

    vv1=vv1/abs(vv1);
    vv2=vv2/abs(vv2);
    vv3=vv3/abs(vv3);

    double ff1=0.0,ff2=0.0,ff3=0.0,contractility=0.0;

    int s1,s2,s3;


    //std::cout<<"check 07   \n";

    for(int c=0; c<N_c; c++) if(in[c]){

        vec3D RR=R[c]-Rcms;

        if(RR.Dot(vv1)>0) s1=1.0;
        else s1=-1.0;

        if(RR.Dot(vv2)>0) s2=1.0;
        else s2=-1.0;

        if(RR.Dot(vv3)>0) s3=1.0;
        else s3=-1.0;

        ff1+=s1*vv1.Dot( vec3D( f_glo[3*c],f_glo[3*c+1],f_glo[3*c+2] ));
        ff2+=s2*vv2.Dot( vec3D( f_glo[3*c],f_glo[3*c+1],f_glo[3*c+2] ));
        ff3+=s3*vv3.Dot( vec3D( f_glo[3*c],f_glo[3*c+1],f_glo[3*c+2] ));

        contractility+=RR.Dot(vec3D( f_glo[3*c],f_glo[3*c+1],f_glo[3*c+2] ))/abs(RR);

    }

    results["CONTRACTILITY"]=contractility;


    //std::cout<<"check 08   \n";

    std::vector<double> values;
    std::vector<vec3D> vecs;
    std::vector<int> indices;

    values.clear();
    indices.clear();

    values.push_back(ff1);
    vecs.push_back(vv1);
    indices.push_back(0);

    values.push_back(ff2);
    vecs.push_back(vv2);
    indices.push_back(1);

    values.push_back(ff3);
    vecs.push_back(vv3);
    indices.push_back(2);



    //std::cout<<"check 09   \n";

    BubbleSort_IVEC_using_associated_dVals(indices,values);

    double f1=values[2];
    vec3D v1=vecs[indices[2]];
    results["F1"]=f1;
    results["V1_X"]=v1.x;
    results["V1_Y"]=v1.y;
    results["V1_Z"]=v1.z;

    double f2=values[1];
    vec3D v2=vecs[indices[1]];
    results["F2"]=f2;
    results["V2_X"]=v2.x;
    results["V2_Y"]=v2.y;
    results["V2_Z"]=v2.z;

    double f3=values[0];
    vec3D v3=vecs[indices[0]];
    results["F3"]=f3;
    results["V3_X"]=v3.x;
    results["V3_Y"]=v3.y;
    results["V3_Z"]=v3.z;

    double polarity=f1/(f1+f2+f3);
    results["POLARITY"]=polarity;

    */
}

void FiniteBodyForces::storePrincipalStressAndStiffness(std::string sbname, std::string sbminname, std::string epkname){

    DRec3D sbrec = DRec3D();
    DRec3D sbminrec = DRec3D();
    DRec3D epkrec = DRec3D();


    for(int tt=0; tt<N_T; tt++){

        mat3D F;

        double dEdsbar,dEdsbarbar,deltal; //sstarsstar

        vec3D s_bar;

        int li;

        std::vector< std::vector<double> > u_T(3,std::vector<double>(4,0.0));

        std::vector< std::vector<double> > FF(3,std::vector<double>(3,0.0));

        std::vector< std::vector< std::vector< std::vector<double> > > > K(3,
            std::vector< std::vector< std::vector<double> > >(3,
                std::vector< std::vector<double> >(3,
                    std::vector<double>(3,0.0)
                )
            )
        );

        mat3D P=mat3D(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0);

        double E=0.0;

        double dlbmax=-1.0,dlbmin=4.0;
        int bmax=0,bmin=0;

        if((tt%100)==0) std::cout<<"computing principal stress and stiffness "<<(floor((tt/(N_T+0.0))*1000)+0.0)/10.0<<"%          \r";

        for(int t=0; t<4; t++){

            u_T[0][t]=U[T[tt][t]].x;
            u_T[1][t]=U[T[tt][t]].y;
            u_T[2][t]=U[T[tt][t]].z;

        }

        FF=u_T*Phi[tt];

        F=mat3D(

            FF[0][0]+1.0,FF[0][1],FF[0][2],
            FF[1][0],FF[1][1]+1.0,FF[1][2],
            FF[2][0],FF[2][1],FF[2][2]+1.0

            );

        for(int b=0; b<N_b; b++){

            s_bar=F*s[b];

            deltal=abs(s_bar)-1.0;

            if(deltal>dlbmax) {
                bmax=b;
                dlbmax=deltal;
            }

            if(deltal<dlbmin) {
                bmin=b;
                dlbmin=deltal;
            }

            li=lround( (deltal-dlmin)/dlstep );

            if( li>((dlmax-dlmin)/dlstep) ) {
                li=((dlmax-dlmin)/dlstep)-1;
            }

            E+=epsilon[li]/(N_b+0.0);

            P+=outerprod(s[b],s_bar)*epsbar[li]*(1.0/(deltal+1.0)/(N_b+0.0));

            dEdsbar=-1.0*(epsbar[li]/(deltal+1.0))/(N_b+0.0);

            dEdsbarbar=( ((deltal+1.0)*epsbarbar[li]-epsbar[li])/((deltal+1.0)*(deltal+1.0)*(deltal+1.0)) )/(N_b+0.0);

            for(int i=0;i<3;i++) for(int j=0;j<3;j++) for(int k=0;k<3;k++) for(int l=0;l<3;l++){

                K[i][j][k][l]+=dEdsbarbar*s[b][i]*s[b][k]*s_bar[j]*s_bar[l];
                if(j==l) K[i][j][k][l]-=dEdsbar*s[b][i]*s[b][k];

            }


        }

        double p=(P*s[bmax]).Dot(s[bmax]);

        double kk=0;

        for(int i=0;i<3;i++) for(int j=0;j<3;j++) for(int k=0;k<3;k++) for(int l=0;l<3;l++){

            kk+=K[i][j][k][l]*s[bmax][i]*s[bmax][j]*s[bmax][k]*s[bmax][l];

        }

        if(false){

            s_bar=F*s[bmax];

            deltal=abs(s_bar)-1.0;

            li=lround( (deltal-dlmin)/dlstep );

            if( li>((dlmax-dlmin)/dlstep) ) {
                li=((dlmax-dlmin)/dlstep)-1;
            }

            std::cout<<"now at "<<tt<<" E="<<E<<" p="<<p<<" kk="<<kk<<" kdia="<<(K[0][0][0][0]+K[1][1][1][1]+K[2][2][2][2])<<" \n";
            std::cout<<"eps="<<epsilon[li]<<" epsbar="<<epsbar[li]<<" epsbarbar="<<epsbarbar[li]<<" \n";

        }

        sbrec.record(F*s[bmax]);
        sbminrec.record(F*s[bmin]);
        epkrec.record(vec3D(E,p,kk));

    }


    sbrec.store(sbname.c_str());
    sbminrec.store(sbminname.c_str());
    epkrec.store(epkname.c_str());

}

void FiniteBodyForces::storeRAndU(std::string Rname, std::string Uname){

    DRec3D Rrec=DRec3D();
    DRec3D Urec=DRec3D();

    int c;

    for(c=0; c<N_c; c++){
        Rrec.record(R[c]);
        Urec.record(U[c]);
    }

    Rrec.store(Rname.c_str());
    Urec.store(Uname.c_str());


}

void FiniteBodyForces::storeF(std::string Fname){

    DRec3D Frec=DRec3D();

    int c;

    for(c=0; c<N_c; c++){ Frec.record(vec3D( f_glo[3*c],f_glo[3*c+1],f_glo[3*c+2] )); }

    Frec.store(Fname.c_str());

}

void FiniteBodyForces::storeFden(std::string Fdenname){

    std::vector<double> Vr;
    Vr.assign(N_c,0.0);

    for(int64_t tt=0; tt<N_T; tt++) for(int t=0; t<4; t++) Vr[T[tt][t]]+=V[tt]*0.25;

    DRec3D Frec=DRec3D();

    for(int c=0; c<N_c; c++){ Frec.record(vec3D( f_glo[3*c]/Vr[c],f_glo[3*c+1]/Vr[c],f_glo[3*c+2]/Vr[c] )); }

    Frec.store(Fdenname.c_str());

}

void FiniteBodyForces::storeEandV(std::string Rname,std::string EVname){

    DRec3D Rrec=DRec3D();
    DRecXY EVrec=DRecXY();

    vec3D N;

    int t;

    for(t=0; t<N_T; t++){

        N=(R[T[t][0]]+R[T[t][1]]+R[T[t][2]]+R[T[t][3]])*0.25;

        Rrec.record( N );

        EVrec.record( E[t], V[t] ) ;


    }

    Rrec.store(Rname.c_str());
    EVrec.store(EVname.c_str());

}



