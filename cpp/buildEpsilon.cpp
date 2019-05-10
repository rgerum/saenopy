extern config CFG;

void saveEpsilon(std::vector<double> epsilon, const char* fname){

    DRecXY epsrec=DRecXY();

    int imax=ceil( (double(CFG["EPSMAX"])+1.0)/double(CFG["EPSSTEP"]) );

    double lambda;

    for(int i=0; i<imax; i++){

        lambda=(i*double(CFG["EPSSTEP"]))-1.0;

        epsrec.record(lambda,epsilon[i]);

    }

    epsrec.store(fname);

}

void buildEpsilon(std::vector<double>& epsilon, std::vector<double>& epsbar, std::vector<double>& epsbarbar, double k1, double ds0, double s1, double ds1){

    std::cout<<k1<<" "<<ds0<<" "<<s1<<" "<<ds1<<std::endl;

    int imax=ceil( (double(CFG["EPSMAX"])+1.0)/double(CFG["EPSSTEP"]) );

    epsilon.assign(imax,0.0);
    epsbar.assign(imax,0.0);
    epsbarbar.assign(imax,0.0);

    double lambda;

    int i;

    for(i=0; i<imax; i++){

        lambda=(i*double(CFG["EPSSTEP"]))-1.0;

        epsbarbar[i]=0;

        if(lambda>0) {

            //epsbarbar[i]+=k1-k0;

            epsbarbar[i]+=(k1);

            if(lambda>s1 && ds1>0.0){

                epsbarbar[i]+=(k1)*(exp((lambda-s1)/ds1)-1.0);

            }

        }else{

            //epsbarbar[i]+=(k1)*exp( ((lambda)/ds0) );

            if(ds0!=0.0) epsbarbar[i]+=(k1)*exp( ((lambda)/ds0) );
            else epsbarbar[i]+=(k1);

        }

        if(epsbarbar[i]>10e10) epsbarbar[i]=10e10;

        //if(lambda<1.0) std::cout<<i<<": lambda="<<lambda<<"   epsbarbar="<<epsbarbar[i]<<std::endl;

    }

    double sum=0.0;

    for(i=0; i<imax; i++){
        sum+=epsbarbar[i]*double(CFG["EPSSTEP"]);
        epsbar[i]=sum;
    }

    int imid=ceil(1.0/double(CFG["EPSSTEP"]));

    double off=epsbar[imid];

    for(i=0; i<imax; i++) epsbar[i]-=off;

    //for(i=0; i<(imax/5*2); i=i+100) std::cout<<i<<": lambda="<<(i*double(CFG["EPSSTEP"]))-1.0<<"   epsbar="<<epsbar[i]<<std::endl;

    sum=0.0;

    for(i=0; i<imax; i++){
        sum+=epsbar[i]*double(CFG["EPSSTEP"]);
        epsilon[i]=sum;
    }

    off=epsilon[imid];
    for(i=0; i<imax; i++) epsilon[i]-=off;

    //for(i=0; i<(imax/5*2); i=i+100) std::cout<<i<<": lambda="<<(i*double(CFG["EPSSTEP"]))-1.0<<"   epsilon="<<epsilon[i]<<std::endl;

    //saveEpsilon(epsbarbar,"epsbarbar.dat");
    //saveEpsilon(epsbar,"epsbar.dat");
    //saveEpsilon(epsilon,"epsilon.dat");

}
