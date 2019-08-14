void makeBoxmeshCoords(float dx, int nx, float rin, float mulout, std::vector<vec3D>& R){

    int ny=nx;
    int nz=nx;
    float dy=dx;
    float dz=dx;

    float rout=nx*dx*0.5;

    int N_c=nx*nx*nx;

    R.assign(N_c,vec3D(0.0,0.0,0.0));

    std::vector<int> tet=std::vector<int>(4,0);

    int x,y,z,i;

    float X,Y,Z;

    for(x=0; x<nx; x++) for(y=0; y<ny; y++) for(z=0; z<nz; z++){

        i=x+nx*y+nx*ny*z;

        X=x*dx-(nx-1)*dx*0.5;
        Y=y*dy-(ny-1)*dy*0.5;
        Z=z*dz-(nz-1)*dz*0.5;

        double f=max(abs(X),max(abs(Y),abs(Z)));

        double mul=max(1.0, ((f-rin)/(rout-rin)+1.0)*(mulout-1.0)+1.0 );

        R[i]=vec3D(X*mul,Y*mul,Z*mul);


    }

}

void makeBoxmeshTets(int nx, std::vector< std::vector<int> >& T, int grain=1){

    int ny=nx;
    int nz=nx;
    T.clear();

    std::vector<int> tet=std::vector<int>(4,0);

    int x,y,z,i,i1,i2,i3,i4,i5,i6,i7,i8;

    for(x=0; x<nx; x+=grain) for(y=0; y<ny; y+=grain) for(z=0; z<nz; z+=grain){

        i=x+nx*y+nx*ny*z;

        if(x>0 && y>0 && z>0){

            i1=i;
            i2=(x-0)+nx*(y-grain)+nx*ny*(z-0);
            i3=(x-grain)+nx*(y-grain)+nx*ny*(z-0);
            i4=(x-grain)+nx*(y-0)+nx*ny*(z-0);
            i5=(x-0)+nx*(y-0)+nx*ny*(z-grain);
            i6=(x-grain)+nx*(y-0)+nx*ny*(z-grain);
            i7=(x-grain)+nx*(y-grain)+nx*ny*(z-grain);
            i8=(x-0)+nx*(y-grain)+nx*ny*(z-grain);

            //tet[0]=i1; tet[1]=i2; tet[2]=i8; tet[3]=i3; T.push_back(tet);
            //tet[0]=i1; tet[1]=i3; tet[2]=i6; tet[3]=i4; T.push_back(tet);
            //tet[0]=i1; tet[1]=i5; tet[2]=i6; tet[3]=i8; T.push_back(tet);
            //tet[0]=i3; tet[1]=i6; tet[2]=i7; tet[3]=i8; T.push_back(tet);
            //tet[0]=i1; tet[1]=i8; tet[2]=i6; tet[3]=i3; T.push_back(tet);

            tet[0]=i1; tet[1]=i2; tet[2]=i3; tet[3]=i8; T.push_back(tet);
            tet[0]=i1; tet[1]=i3; tet[2]=i4; tet[3]=i6; T.push_back(tet);
            tet[0]=i1; tet[1]=i5; tet[2]=i8; tet[3]=i6; T.push_back(tet);
            tet[0]=i3; tet[1]=i6; tet[2]=i8; tet[3]=i7; T.push_back(tet);
            tet[0]=i1; tet[1]=i8; tet[2]=i3; tet[3]=i6; T.push_back(tet);

        }

    }

}

void setOuterSurf(int nx, int grain, std::vector<bool>& var, bool val){

    int n=nx;

    int x,y,z;

    x=0;

    for(int y=0; y<n; y+=grain) for(int z=0; z<n; z+=grain) var[x+n*y+n*n*z]=val;

    x=grain*(n-1)/grain;

    for(int y=0; y<n; y+=grain) for(int z=0; z<n; z+=grain) var[x+n*y+n*n*z]=val;




    y=0;

    for(int x=0; x<n; x+=grain) for(int z=0; z<n; z+=grain) var[x+n*y+n*n*z]=val;

    y=grain*(n-1)/grain;

    for(int x=0; x<n; x+=grain) for(int z=0; z<n; z+=grain) var[x+n*y+n*n*z]=val;



    z=0;

    for(int x=0; x<n; x+=grain) for(int y=0; y<n; y+=grain) var[x+n*y+n*n*z]=val;

    z=grain*(n-1)/grain;

    for(int x=0; x<n; x+=grain) for(int y=0; y<n; y+=grain) var[x+n*y+n*n*z]=val;


}

void setActiveFields(int nx, int grain, std::vector<bool>& var, bool val){

    int ny=nx;
    int nz=nx;

    int x,y,z,i;

    for(x=0; x<nx; x+=grain) for(y=0; y<ny; y+=grain) for(z=0; z<nz; z+=grain){

        i=x+nx*y+nx*ny*z;

        var[i]=val;

    }


}

int toi(int n,int x, int y, int z){

    return x+n*y+n*n*z;

}

void makeInterpolationMatrix(int n, int grain, std::vector< std::unordered_map<size_t,mat3D> >& I ){

    //nx=CFG["BM_N"];

    I.clear();

    I.resize(n*n*n);

    mat3D Ones=mat3D(1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0);

    for(int x=0; x<n; x+=2*grain) for(int y=0; y<n; y+=2*grain) for(int z=0; z<n; z+=2*grain){

        //std::cout<<x<<" "<<y<<" "<<z<<" \n";

        //std::cout<<"+0 +0 +0 \n";

        I[toi(n,x,y,z)][toi(n,x,y,z)]=Ones;

        //std::cout<<"+1 +0 +0 \n";

        if( x+2*grain < n ){

            I[toi(n,x+grain,y,z)][toi(n,x,y,z)]=Ones*0.5;
            I[toi(n,x+grain,y,z)][toi(n,x+2*grain,y,z)]=Ones*0.5;

        }

        //std::cout<<"+0 +1 +0 \n";

        if( y+2*grain < n ){

            I[toi(n,x,y+grain,z)][toi(n,x,y,z)]=Ones*0.5;
            I[toi(n,x,y+grain,z)][toi(n,x,y+2*grain,z)]=Ones*0.5;

        }

        //std::cout<<"+0 +0 +1 \n";

        if( z+2*grain < n ){

            I[toi(n,x,y,z+grain)][toi(n,x,y,z)]=Ones*0.5;
            I[toi(n,x,y,z+grain)][toi(n,x,y,z+2*grain)]=Ones*0.5;

        }

        //std::cout<<"+1 +1 +0 \n";

        if( (x+2*grain < n) && (y+grain < n) ){

            I[toi(n,x+grain,y+grain,z)][toi(n,x,y,z)]=Ones*0.25;
            I[toi(n,x+grain,y+grain,z)][toi(n,x+2*grain,y,z)]=Ones*0.25;
            I[toi(n,x+grain,y+grain,z)][toi(n,x,y+2*grain,z)]=Ones*0.25;
            I[toi(n,x+grain,y+grain,z)][toi(n,x+2*grain,y+2*grain,z)]=Ones*0.25;

        }

        //std::cout<<"+0 +1 +1 \n";

        if( (y+2*grain < n) && (z+2*grain < n) ){

            I[toi(n,x,y+grain,z+grain)][toi(n,x,y,z)]=Ones*0.25;
            I[toi(n,x,y+grain,z+grain)][toi(n,x,y,z+2*grain)]=Ones*0.25;
            I[toi(n,x,y+grain,z+grain)][toi(n,x,y+2*grain,z)]=Ones*0.25;
            I[toi(n,x,y+grain,z+grain)][toi(n,x,y+2*grain,z+2*grain)]=Ones*0.25;

        }

        //std::cout<<"+1 +0 +1 \n";

        if( (x+2*grain < n) && (z+2*grain < n) ){

            I[toi(n,x+grain,y,z+grain)][toi(n,x,y,z)]=Ones*0.25;
            I[toi(n,x+grain,y,z+grain)][toi(n,x,y,z+2*grain)]=Ones*0.25;
            I[toi(n,x+grain,y,z+grain)][toi(n,x+2*grain,y,z)]=Ones*0.25;
            I[toi(n,x+grain,y,z+grain)][toi(n,x+2*grain,y,z+2*grain)]=Ones*0.25;

        }

        //std::cout<<"+1 +1 +1 \n";

        if( (x+2*grain < n) && (y+2*grain < n) && (z+2*grain < n) ){

            I[toi(n,x+grain,y+grain,z+grain)][toi(n,x,y,z)]=Ones*0.125;
            I[toi(n,x+grain,y+grain,z+grain)][toi(n,x,y,z+2*grain)]=Ones*0.125;
            I[toi(n,x+grain,y+grain,z+grain)][toi(n,x+2*grain,y,z)]=Ones*0.125;
            I[toi(n,x+grain,y+grain,z+grain)][toi(n,x+2*grain,y,z+2*grain)]=Ones*0.125;
            I[toi(n,x+grain,y+grain,z+grain)][toi(n,x,y+2*grain,z)]=Ones*0.125;
            I[toi(n,x+grain,y+grain,z+grain)][toi(n,x,y+2*grain,z+2*grain)]=Ones*0.125;
            I[toi(n,x+grain,y+grain,z+grain)][toi(n,x+2*grain,y+2*grain,z)]=Ones*0.125;
            I[toi(n,x+grain,y+grain,z+grain)][toi(n,x+2*grain,y+2*grain,z+2*grain)]=Ones*0.125;

        }

    }

}

void restricValues(int nx, int grain, std::vector< vec3D >& U){

    int N_c=nx*nx*nx;

    std::vector< vec3D > Ucoarse;

    std::vector< std::unordered_map<size_t,mat3D> > I;

    makeInterpolationMatrix(nx,grain/2,I);

    //std::cout<<I.size()<<" !!!!   \n";

    mulT(I,U,Ucoarse);

    for(int c=0; c<N_c; c++) U[c]=Ucoarse[c]*0.125;

}

void restrictValuesMultiple(int nx, int fgrain, int tgrain, std::vector< vec3D >& U){


    int cgrain=fgrain;

    while( (2*cgrain)!=tgrain ){

        cgrain*=2;
        restricValues(nx,cgrain,U);

    }

    restricValues(nx,tgrain,U);

}

void interpolateFromSubmesh(int nx, int grain, std::vector< vec3D >& U){

    std::vector<bool> var;
    var.assign(nx*nx*nx,false);
    setActiveFields(nx,grain,var,true);

    int ny=nx;
    int nz=nx;

    vec3D meanU=vec3D();
    int mcount=0;

    int x,y,z,i,dx,dy,dz,xx,yy,zz;

    for(x=0; x<nx; x+=grain) for(y=0; y<ny; y+=grain) for(z=0; z<nz; z+=grain) if(!var[x+nx*y+nx*ny*z]){

        meanU=vec3D(0.0,0.0,0.0);
        mcount=0;

        for(dx=-grain; dx<2*grain; dx+=grain) for(dy=-grain; dy<2*grain; dy+=grain) for(dz=-grain; dz<2*grain; dz+=grain){

            xx=x+dx;
            yy=y+dy;
            zz=z+dz;

            //std::cout<<x<<" "<<y<<" "<<z<<" - "<<dx<<" "<<dy<<" "<<dz<<"\n";

            i=xx+nx*yy+nx*ny*zz;

            if(xx>-1 && xx<nx && yy>-1 && yy<nx && zz>-1 && zz<nx) if(var[i]){

                meanU+=U[i];
                //U[i].print2();
                mcount++;

            }

        }

        i=x+nx*y+nx*ny*z;

        //std::cout<<x<<" "<<y<<" "<<z<<" - "<<dx<<" "<<dy<<" "<<dz<<" "<<mcount<<" \n";

        U[i]=meanU*(1.0/(mcount+0.0));
        //U[i].print2();

        //std::cout<<"\n\n";

    }

}

