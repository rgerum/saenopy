#include <QRegExp>
#include <QStringList>
#include <QDir>
#include <QString>

extern config CFG;



//using namespace cimg_library;

typedef std::vector< std::vector< std::vector<unsigned char> > > stack3D;
typedef std::vector< std::vector< std::vector<double> > > substack;

/*
stack3D resize(const stack3D& stack1, int g){

    stack3D stack2=stack3D();

    int sX=stack1.size();
    int sY=stack1[0].size();
    int sZ=stack1[0][0].size();

    stack2.assign(sX/g,
        std::vector< std::vector< unsigned char > >(sY/g,
            std::vector< unsigned char >(sZ/g, 0 )
        )
    );

    int i,j,k,ii,jj,kk;

    int ggg=g*g*g;

    double sum=0.0;

    for(i=0; i<(sX/g); i++) for(j=0; j<(sY/g); j++) for(k=0; k<(sZ/g); k++){

        sum=0.0;

        for(ii=0; ii<g; ii++) for(jj=0; jj<g; jj++) for(kk=0; kk<g; kk++){


            sum+=stack1[g*i+ii][g*j+jj][g*k+kk]+0.0;


        }

        stack2[i][j][k]=sum/ggg;

        //std::cout<<int(stack2[i][j][k])<<std::endl;

    }

    return stack2;

}


double crosscorrelateSections(const stack3D& stack1, const stack3D& stack2, vec3D r, vec3D du, int big=1){

    int sgX=int(CFG["VB_SX"])*big;
    int sgY=int(CFG["VB_SY"])*big;
    int sgZ=int(CFG["VB_SZ"])*big;

    int sX=stack1.size();
    int sY=stack1[0].size();
    int sZ=stack1[0][0].size();

    bool in=true;

    if(r.x<1) in=false;
    if(r.y<1) in=false;
    if(r.z<1) in=false;
    if(r.x>(sX-1)) in=false;
    if((r+du).y>(sY-1)) in=false;
    if((r+du).z>(sZ-1)) in=false;
    if((r+du).x<1) in=false;
    if((r+du).y<1) in=false;
    if((r+du).z<1) in=false;
    if((r+du).x>(sX-1)) in=false;
    if((r+du).y>(sY-1)) in=false;
    if((r+du).z>(sZ-1)) in=false;

    if(in){

        int xf=floor(r.x);
        int yf=floor(r.y);
        int zf=floor(r.z);

        double fx=r.x-xf;
        double fy=r.y-yf;
        double fz=r.z-zf;

        int dxf=floor(du.x+r.x);
        int dyf=floor(du.y+r.y);
        int dzf=floor(du.z+r.z);

        double fdx=du.x+r.x-dxf;
        double fdy=du.y+r.y-dyf;
        double fdz=du.z+r.z-dzf;

        int ii,jj,kk;

        std::vector< std::vector< std::vector<double> > > substack1,substack2;

        substack1.assign(sgX,
            std::vector< std::vector< double > >(sgY,
                std::vector< double >(sgZ, 0.0 )
            )
        );

        substack2.assign(sgX,
            std::vector< std::vector< double > >(sgY,
                std::vector< double >(sgZ, 0.0 )
            )
        );

        int sum1=0;
        int sum2=0;

        int iii,jjj,kkk;

        //std::cout<<xf<<","<<yf<<","<<zf<<","<<dxf<<","<<dyf<<","<<dzf<<" | "<<sX<<","<<sY<<","<<sZ<<" \n";

        if( xf>(sgX/2) && xf<(sX-sgX/2) && yf>(sgY/2) && yf<(sY-sgY/2) && zf>(sgZ/2) && zf<(sZ-sgZ/2) && dxf>(sgX/2) && dxf<(sX-sgX/2) && dyf>(sgY/2) && dyf<(sY-sgY/2) && dzf>(sgZ/2) && dzf<(sZ-sgZ/2) ){

            for(ii=0; ii<sgX; ii++) for(jj=0; jj<sgY; jj++) for(kk=0; kk<sgZ; kk++){

                iii=ii-sgX/2;
                jjj=jj-sgY/2;
                kkk=kk-sgZ/2;

                substack1[ii][jj][kk]=(
                                        (1-fx)*(1-fy)*(1-fz)*stack1[iii+xf][jjj+yf][kkk+zf]+
                                        (fx)*(1-fy)*(1-fz)*stack1[iii+xf+1][jjj+yf][kkk+zf]+
                                        (1-fx)*(fy)*(1-fz)*stack1[iii+xf][jjj+yf+1][kkk+zf]+
                                        (1-fx)*(1-fy)*(fz)*stack1[iii+xf][jjj+yf][kkk+zf+1]+
                                        (fx)*(fy)*(1-fz)*stack1[iii+xf+1][jjj+yf+1][kkk+zf]+
                                        (1-fx)*(fy)*(fz)*stack1[iii+xf][jjj+yf+1][kkk+zf+1]+
                                        (fx)*(1-fy)*(fz)*stack1[iii+xf+1][jjj+yf][kkk+zf+1]+
                                        (fx)*(fy)*(fz)*stack1[iii+xf+1][jjj+yf+1][kkk+zf+1]
                                    );

                substack2[ii][jj][kk]=(
                                        (1-fdx)*(1-fdy)*(1-fdz)*stack2[iii+dxf][jjj+dyf][kkk+dzf]+
                                        (fdx)*(1-fdy)*(1-fdz)*stack2[iii+dxf+1][jjj+dyf][kkk+dzf]+
                                        (1-fdx)*(fdy)*(1-fdz)*stack2[iii+dxf][jjj+dyf+1][kkk+dzf]+
                                        (1-fdx)*(1-fdy)*(fdz)*stack2[iii+dxf][jjj+dyf][kkk+dzf+1]+
                                        (fdx)*(fdy)*(1-fdz)*stack2[iii+dxf+1][jjj+dyf+1][kkk+dzf]+
                                        (1-fdx)*(fdy)*(fdz)*stack2[iii+dxf][jjj+dyf+1][kkk+dzf+1]+
                                        (fdx)*(1-fdy)*(fdz)*stack2[iii+dxf+1][jjj+dyf][kkk+dzf+1]+
                                        (fdx)*(fdy)*(fdz)*stack2[iii+dxf+1][jjj+dyf+1][kkk+dzf+1]
                                    );

                sum1+=substack1[ii][jj][kk];
                sum2+=substack2[ii][jj][kk];


            }

            double sssg=(sgX*sgY*sgZ)+0.0;

            for(ii=0; ii<sgX; ii++) for(jj=0; jj<sgY; jj++) for(kk=0; kk<sgZ; kk++){

                        substack1[ii][jj][kk]-=sum1/sssg;
                        substack2[ii][jj][kk]-=sum2/sssg;


            }



            double var1=0;
            double var2=0;

            for(ii=0; ii<sgX; ii++) for(jj=0; jj<sgY; jj++) for(kk=0; kk<sgZ; kk++){

                        var1+=substack1[ii][jj][kk]*substack1[ii][jj][kk];
                        var2+=substack2[ii][jj][kk]*substack2[ii][jj][kk];


            }

            var1=sqrt(var1);
            var2=sqrt(var2);

            for(ii=0; ii<sgX; ii++) for(jj=0; jj<sgY; jj++) for(kk=0; kk<sgZ; kk++){

                substack1[ii][jj][kk]/=var1;
                substack2[ii][jj][kk]/=var2;

            }

            double S=0.0;

            for(ii=0; ii<sgX; ii++) for(jj=0; jj<sgY; jj++) for(kk=0; kk<sgZ; kk++){

                S+=substack1[ii][jj][kk]*substack2[ii][jj][kk];

            }

            return S;

        }else{

            return -1.0;

        }

    }else{

        return -1.0;

    }

}

*/

double crosscorrelateSections(const substack& substackr,const substack& substacka){

    int sgX=int(CFG["VB_SX"]);
    int sgY=int(CFG["VB_SY"]);
    int sgZ=int(CFG["VB_SZ"]);

    double S=0.0;

    for(auto ii=0; ii<sgX; ii++) for(auto jj=0; jj<sgY; jj++) for(auto kk=0; kk<sgZ; kk++){

        S+=substackr[ii][jj][kk]*substacka[ii][jj][kk];

    }

    return S;

}

substack getSubstack(const stack3D& stack1, vec3D r){

    int sgX=int(CFG["VB_SX"]);
    int sgY=int(CFG["VB_SY"]);
    int sgZ=int(CFG["VB_SZ"]);

    int sX=stack1.size();
    int sY=stack1[0].size();
    int sZ=stack1[0][0].size();

    std::vector< std::vector< std::vector<double> > > substack;

    substack.assign(sgX,
        std::vector< std::vector< double > >(sgY,
            std::vector< double >(sgZ, 0.0 )
        )
    );

    int ii,jj,kk;

    int sum=0;

    int iii,jjj,kkk;

    //std::cout<<xf<<","<<yf<<","<<zf<<","<<dxf<<","<<dyf<<","<<dzf<<" | "<<sX<<","<<sY<<","<<sZ<<" \n";


    int xf=floor(r.x);
    int yf=floor(r.y);
    int zf=floor(r.z);

    double fx=r.x-xf;
    double fy=r.y-yf;
    double fz=r.z-zf;

    if( xf>(sgX/2) && xf<(sX-sgX/2) && yf>(sgY/2) && yf<(sY-sgY/2) && zf>(sgZ/2) && zf<(sZ-sgZ/2) ){

        //std::cout<<xf<<","<<yf<<","<<zf<<", | "<<sX<<","<<sY<<","<<sZ<<" \n";

        for(ii=0; ii<sgX; ii++) for(jj=0; jj<sgY; jj++) for(kk=0; kk<sgZ; kk++){

            iii=ii-sgX/2;
            jjj=jj-sgY/2;
            kkk=kk-sgZ/2;

            substack[ii][jj][kk]=(
                                    (1-fx)*(1-fy)*(1-fz)*stack1[iii+xf][jjj+yf][kkk+zf]+
                                    (fx)*(1-fy)*(1-fz)*stack1[iii+xf+1][jjj+yf][kkk+zf]+
                                    (1-fx)*(fy)*(1-fz)*stack1[iii+xf][jjj+yf+1][kkk+zf]+
                                    (1-fx)*(1-fy)*(fz)*stack1[iii+xf][jjj+yf][kkk+zf+1]+
                                    (fx)*(fy)*(1-fz)*stack1[iii+xf+1][jjj+yf+1][kkk+zf]+
                                    (1-fx)*(fy)*(fz)*stack1[iii+xf][jjj+yf+1][kkk+zf+1]+
                                    (fx)*(1-fy)*(fz)*stack1[iii+xf+1][jjj+yf][kkk+zf+1]+
                                    (fx)*(fy)*(fz)*stack1[iii+xf+1][jjj+yf+1][kkk+zf+1]
                                );



            sum+=substack[ii][jj][kk];


        }

        double sssg=(sgX*sgY*sgZ)+0.0;

        for(ii=0; ii<sgX; ii++) for(jj=0; jj<sgY; jj++) for(kk=0; kk<sgZ; kk++){

                    substack[ii][jj][kk]-=sum/sssg;


        }



        double var=0;

        for(ii=0; ii<sgX; ii++) for(jj=0; jj<sgY; jj++) for(kk=0; kk<sgZ; kk++){

                    var+=substack[ii][jj][kk]*substack[ii][jj][kk];


        }

        var=sqrt(var);

        for(ii=0; ii<sgX; ii++) for(jj=0; jj<sgY; jj++) for(kk=0; kk<sgZ; kk++){

            substack[ii][jj][kk]/=var;

        }

        //std::cout<<substack[0][0][0]<<"              \n";

        return substack;

    }else{

        substack.assign(sgX,
            std::vector< std::vector< double > >(sgY,
                std::vector< double >(sgZ, 0.0 )
            )
        );

        return substack;

    }

}

double crosscorrelateSections(const substack& substackr,const stack3D& stack2, vec3D r){

    substack substacka=getSubstack(stack2,r);

    return crosscorrelateSections(substackr,substacka);

}

double crosscorrelateSections(const stack3D& stack1, const stack3D& stack2, vec3D r, vec3D du){

    substack substackr=getSubstack(stack1,r);
    substack substacka=getSubstack(stack2,r+du);

    return crosscorrelateSections(substackr,substacka);

}

substack operator+= (substack substack1, const substack substack2){

    int sgX=int(CFG["VB_SX"]);
    int sgY=int(CFG["VB_SY"]);
    int sgZ=int(CFG["VB_SZ"]);

    for(auto ii=0; ii<sgX; ii++) for(auto jj=0; jj<sgY; jj++) for(auto kk=0; kk<sgZ; kk++){

        substack1[ii][jj][kk]+=substack2[ii][jj][kk];

    }

    return substack1;

}

vec3D findLocalDisplacement(const substack substackr ,const stack3D& stacka, vec3D R, vec3D Ustart, DRec& Srec, double lambda){

    //double lambda=0.01;


    double subpixel=double(CFG["SUBPIXEL"]);

    std::vector<vec3D> P;

    std::vector<double> S;

    double hinit=4.0;

    P.resize(4);
    S.resize(4);

    srand ( time(NULL) );

    P[0]=vec3D((rand()%2000)/1000.0-1.0,(rand()%2000)/1000.0-1.0,(rand()%2000)/1000.0-1.0)*hinit;
    P[1]=vec3D((rand()%2000)/1000.0-1.0,(rand()%2000)/1000.0-1.0,(rand()%2000)/1000.0-1.0)*hinit;
    P[2]=vec3D((rand()%2000)/1000.0-1.0,(rand()%2000)/1000.0-1.0,(rand()%2000)/1000.0-1.0)*hinit;
    P[3]=vec3D((rand()%2000)/1000.0-1.0,(rand()%2000)/1000.0-1.0,(rand()%2000)/1000.0-1.0)*hinit;

    S[0]=crosscorrelateSections(substackr,stacka,R+P[0]+Ustart)-lambda*norm(P[0]);
    S[1]=crosscorrelateSections(substackr,stacka,R+P[1]+Ustart)-lambda*norm(P[1]);
    S[2]=crosscorrelateSections(substackr,stacka,R+P[2]+Ustart)-lambda*norm(P[2]);
    S[3]=crosscorrelateSections(substackr,stacka,R+P[3]+Ustart)-lambda*norm(P[3]);

    size_t mini=minimal(S);
    size_t maxi=maximal(S);

    bool done=false;

    vec3D P_ref;
    vec3D P_exp;
    vec3D P_con;
    vec3D P_plane;

    double S_ref;
    double S_exp;
    double S_con;

    double mx,my,mz,stdx,stdy,stdz;

    unsigned int i;

    int ii;

    for(ii=0; (ii<1000 && !done); ii++){

        mini=minimal(S);
        maxi=maximal(S);

        //std::cout<<"| new cycle mini = "<<mini<<std::endl;

        //reflect

        P_plane=vec3D(0.0,0.0,0.0);
        for(i=0; i<4; i++) if(i!=mini) P_plane+=P[i];
        P_plane=P_plane*0.333333333;

        P_ref=P[mini]+(P_plane-P[mini])*2.0;

        //B.Shift=P_ref; S_ref=B.testDeformation(stackr,stacka,M);
        S_ref=crosscorrelateSections(substackr,stacka,R+P_ref+Ustart)-lambda*norm(P_ref);

        if(S_ref>S[maxi]){

            //expand

            //std::cout<<"| expanding "<<std::endl;

            P_exp=P[mini]+(P_plane-P[mini])*3.0;

            //B.Shift=P_exp; S_exp=B.testDeformation(stackr,stacka,M);
            S_exp=crosscorrelateSections(substackr,stacka,R+P_exp+Ustart)-lambda*norm(P_exp);

            if(S_exp>S_ref){

                //std::cout<<"| took expanded "<<std::endl;

                P[mini]=P_exp;
                S[mini]=S_exp;

            }else{

                //std::cout<<"| took reflected (expanded was worse)"<<std::endl;

                P[mini]=P_ref;
                S[mini]=S_ref;

            }

        }else{

            bool bsw=false;
            for(i=0;i<4;i++) if(i!=mini) if(S_ref>S[i]) bsw=true;

            if(bsw){

                //std::cout<<"| took reflected (better than second worst)"<<std::endl;

                P[mini]=P_ref;
                S[mini]=S_ref;

            }else{

                if(S_ref>S[maxi]){

                    //std::cout<<"| took reflected (not better than second worst)"<<std::endl;

                    P[mini]=P_ref;
                    S[mini]=S_ref;

                }else{

                    P_con=P[mini]+(P_plane-P[mini])*0.5;
                    //B.Shift=P_con; S_con=B.testDeformation(stackr,stacka,M);
                    S_con=crosscorrelateSections(substackr,stacka,R+P_con+Ustart)-lambda*norm(P_con);

                    if(S_con>S[mini]){

                        //std::cout<<"| took contracted"<<std::endl;

                        P[mini]=P_con;
                        S[mini]=S_con;

                    }else{

                        //std::cout<<"| contracting myself"<<std::endl;

                        for(i=0; i<4; i++) if(i!=maxi) {

                            P[i]=(P[maxi]+P[i])*0.5;
                            //B.Shift=P[i]; S[i]=B.testDeformation(stackr,stacka,M);
                            S[i]=crosscorrelateSections(substackr,stacka,R+P[i]+Ustart)-lambda*norm(P[i]);

                        }

                    }

                }

            }

        }

        //P[0].print2();
        //P[1].print2();
        //P[2].print2();
        //P[3].print2();

        //std::cout<<"0: "<<S[0]<<" ; "<<P[0].x<<" , "<<P[0].y<<" , "<<P[0].z<<std::endl;
        //std::cout<<"1: "<<S[1]<<" ; "<<P[1].x<<" , "<<P[1].y<<" , "<<P[1].z<<std::endl;
        //std::cout<<"2: "<<S[2]<<" ; "<<P[2].x<<" , "<<P[2].y<<" , "<<P[2].z<<std::endl;
        //std::cout<<"3: "<<S[3]<<" ; "<<P[3].x<<" , "<<P[3].y<<" , "<<P[3].z<<std::endl;

        //mS=(S[0]+S[1]+S[2]+S[3])

        //std::cout<<" S_ref = "<<S_ref<<std::endl;

        mx=(P[0].x+P[1].x+P[2].x+P[3].x)*0.25;
        my=(P[0].y+P[1].y+P[2].y+P[3].y)*0.25;
        mz=(P[0].z+P[1].z+P[2].z+P[3].z)*0.25;

        stdx=sqrt( (P[0].x-mx)*(P[0].x-mx) + (P[1].x-mx)*(P[1].x-mx) + (P[2].x-mx)*(P[2].x-mx) + (P[3].x-mx)*(P[3].x-mx) )*0.25;
        stdy=sqrt( (P[0].y-my)*(P[0].y-my) + (P[1].y-my)*(P[1].y-my) + (P[2].y-my)*(P[2].y-my) + (P[3].y-my)*(P[3].y-my) )*0.25;
        stdz=sqrt( (P[0].z-mz)*(P[0].z-mz) + (P[1].z-mz)*(P[1].z-mz) + (P[2].z-mz)*(P[2].z-mz) + (P[3].z-mz)*(P[3].z-mz) )*0.25;

        //std::cout<<"stdx = "<<stdx<<" ; stdy = "<<stdy<<" ; stdz = "<<stdz<<std::endl;

        if(stdx<subpixel && stdy<subpixel && stdz<subpixel) done=true;


    }

    mini=minimal(S);
    maxi=maximal(S);

    //B.Shift=P[maxi];

   // B.Shift.print2();

   //std:cout<<S[maxi]<<std::endl;

    Srec.record(S[maxi]);

   return P[maxi]+Ustart;


}

double crosscorrelateStacks(const stack3D& stack1, const stack3D& stack2, vec3D du, int jump=-1){

    //std::cout<<"check 01 \n";

    int sX=stack1.size();
    int sY=stack1[0].size();
    int sZ=stack1[0][0].size();

    int jumpx=jump;
    int jumpy=jump;
    int jumpz=jump;

    if(jump<0){

        jumpx=sX/24;
        jumpy=sY/24;
        jumpz=sZ/24;

    }

    int sgX=(sX-jumpx)/jumpx-1;
    int sgY=(sY-jumpy)/jumpy-1;
    int sgZ=(sZ-jumpz)/jumpz-1;

    int dif=floor(du.x);
    int djf=floor(du.y);
    int dkf=floor(du.z);

    double fdi=du.x-dif;
    double fdj=du.y-djf;
    double fdk=du.z-dkf;

    int ii,jj,kk;

    std::vector< std::vector< std::vector<double> > > substack1, substack2;


    //std::cout<<"check 02 \n"<<sgX<<" "<<sgY<<" "<<sgZ<<"\n";

    substack1.assign(sgX,
        std::vector< std::vector< double > >(sgY,
            std::vector< double >(sgZ, 0.0 )
        )
    );

    substack2.assign(sgX,
        std::vector< std::vector< double > >(sgY,
            std::vector< double >(sgZ, 0.0 )
        )
    );

    int sum1=0;
    int sum2=0;

    int i1,j1,k1,i2,j2,k2;

    //std::cout<<"check 03 \n";

    for(ii=0; ii<sgX; ii++) for(jj=0; jj<sgY; jj++) for(kk=0; kk<sgZ; kk++){

        i1=jumpx*(ii+1);
        i2=jumpx*(ii+1)+dif;
        j1=jumpy*(jj+1);
        j2=jumpy*(jj+1)+djf;
        k1=jumpz*(kk+1);
        k2=jumpz*(kk+1)+dkf;

        substack1[ii][jj][kk]=(stack1[i1][j1][k1]);

        substack2[ii][jj][kk]=(
                                (1-fdi)*(1-fdj)*(1-fdk)*stack2[(i2+sX)%sX][(j2+sY)%sY][(k2+sZ)%sZ]+
                                (fdi)*(1-fdj)*(1-fdk)*stack2[(i2+1+sX)%sX][(j2+sY)%sY][(k2+sZ)%sZ]+
                                (1-fdi)*(fdj)*(1-fdk)*stack2[(i2+sX)%sX][(j2+1+sY)%sY][(k2+sZ)%sZ]+
                                (1-fdi)*(1-fdj)*(fdk)*stack2[(i2+sX)%sX][(j2+sY)%sY][(k2+1+sZ)%sZ]+
                                (fdi)*(fdj)*(1-fdk)*stack2[(i2+1+sX)%sX][(j2+1+sY)%sY][(k2+sZ)%sZ]+
                                (1-fdi)*(fdj)*(fdk)*stack2[(i2+sX)%sX][(j2+1+sY)%sY][(k2+1+sZ)%sZ]+
                                (fdi)*(1-fdj)*(fdk)*stack2[(i2+1+sX)%sX][(j2+sY)%sY][(k2+1+sZ)%sZ]+
                                (fdi)*(fdj)*(fdk)*stack2[(i2+1+sX)%sX][(j2+1+sY)%sY][(k2+1+sZ)%sZ]
                            );

        sum1+=substack1[ii][jj][kk];
        sum2+=substack2[ii][jj][kk];

    }

    double sssg=(sgX*sgY*sgZ)+0.0;

    for(ii=0; ii<sgX; ii++) for(jj=0; jj<sgY; jj++) for(kk=0; kk<sgZ; kk++){

                substack1[ii][jj][kk]-=sum1/sssg;
                substack2[ii][jj][kk]-=sum2/sssg;


    }

    double var1=0;
    double var2=0;

    for(ii=0; ii<sgX; ii++) for(jj=0; jj<sgY; jj++) for(kk=0; kk<sgZ; kk++){

                var1+=substack1[ii][jj][kk]*substack1[ii][jj][kk];
                var2+=substack2[ii][jj][kk]*substack2[ii][jj][kk];


    }

    var1=sqrt(var1);
    var2=sqrt(var2);

    for(ii=0; ii<sgX; ii++) for(jj=0; jj<sgY; jj++) for(kk=0; kk<sgZ; kk++){

        substack1[ii][jj][kk]/=var1;
        substack2[ii][jj][kk]/=var2;

    }

    double S=0.0;

    for(ii=0; ii<sgX; ii++) for(jj=0; jj<sgY; jj++) for(kk=0; kk<sgZ; kk++){

        S+=substack1[ii][jj][kk]*substack2[ii][jj][kk];

    }

    return S;

}

/*
void blur(stack3D& stack, stack3D& stack2 , int kernelsize){

    std::vector< std::vector< std::vector< double > > > kernel;

    kernel.assign(kernelsize,
        std::vector< std::vector< double > >(kernelsize,
            std::vector< double >(kernelsize, 0.0 )
        )
    );

    int i,j,k;

    double dx,dy,dz,Isum=0.0;

    for(i=0; i<kernelsize; i++) for(j=0; j<kernelsize; j++) for(k=0; k<kernelsize; k++){

        dx=floor(i-0.5*kernelsize)/kernelsize;
        dy=floor(j-0.5*kernelsize)/kernelsize;
        dz=floor(k-0.5*kernelsize)/kernelsize;

        kernel[i][j][k]=exp(-dx*dx-dy*dy-dz*dz);
        Isum+=kernel[i][j][k];

    }

    for(i=0; i<kernelsize; i++) for(j=0; j<kernelsize; j++) for(k=0; k<kernelsize; k++) kernel[i][j][k]/=Isum;


    int sX=stack.size();
    int sY=stack[0].size();
    int sZ=stack[0][0].size();

    stack2.assign(sX,
        std::vector< std::vector< unsigned char > >(sY,
            std::vector< unsigned char >(sZ, 0 )
        )
    );

    int i1,j1,k1,i2,j2,k2,ii,jj,kk,iii,jjj,kkk;

    double I=0.0;

    for(i=0; i<sX; i++){

        std::cout<<"bluring "<<floor( (i+0.0)/(sX+0.0)*10000 )/100<<"% done         \r";

        for(j=0; j<sY; j++) for(k=0; k<sZ; k++){

            i1=max(double(0),floor(i-0.5*kernelsize));
            i2=min(double(sX-1),ceil(i+0.5*kernelsize)-1);

            j1=max(double(0),floor(j-0.5*kernelsize));
            j2=min(double(sY-1),ceil(j+0.5*kernelsize)-1);

            k1=max(double(0),floor(k-0.5*kernelsize));
            k2=min(double(sZ-1),ceil(k+0.5*kernelsize)-1);

            I=0.0;

            for(ii=i1; ii<i2; ii++) for(jj=j1; jj<j2; jj++) for(kk=k1; kk<k2; kk++){


                iii=ii-i+floor(0.5*kernelsize)+1;
                jjj=jj-j+floor(0.5*kernelsize)+1;
                kkk=kk-k+floor(0.5*kernelsize)+1;

                //std::cout<<i<<","<<j<<","<<k<<" "<<ii<<","<<jj<<","<<kk<<" "<<iii<<","<<jjj<<","<<kkk<<"\n";

                I+=stack[ii][jj][kk]*kernel[iii][jjj][kkk];

            }

            stack2[i][j][k]=floor(I);

        }
    }
}

stack2D slice(stack3D& stack, double theta, double phi, int thickness){


    int sizeX=stack.size();
    int sizeY=stack[0].size();
    int sizeZ=stack[0][0].size();


    stack2D img=emptyStack2D(sizeX,sizeY);

    int x,y,z;

    mat3D M=rotMatZ(phi)*rotMatX(theta);

    //M.print();

    vec3D middle=vec3D(sizeX*0.5,sizeY*0.5,sizeZ*0.5);

    vec3D r;

    double I;

    //int xf,yf,zf;
    //double fx,fy,fz;

    double zz;

    //int i;


    for(z=0;z<thickness;z++){

    //std::cout<<z<<"\r";

        for(x=0;x<sizeX;x++){


            for(y=0;y<sizeY;y++){


                zz=-thickness*0.5+z+0.5+sizeZ*0.5;

                r=vec3D(x,y,zz)-middle;

                r=M*r;

                r=r+middle;

                //r.print2();

                int xf=floor(r.x);
                int yf=floor(r.y);
                int zf=floor(r.z);

                double fx=r.x-xf;
                double fy=r.y-yf;
                double fz=r.z-zf;

                //std::cout<<x<<" "<<y<<" "<<zz<<" | "<<xf<<" "<<yf<<" "<<zf<<"\r";

                if(xf>-1 && yf>-1 && zf>-1 && xf<(sizeX-1) && yf<(sizeY-1) && zf<(sizeZ-1) ){

                    I=(
                        (1-fx)*(1-fy)*(1-fz)*stack[xf][yf][zf]+
                        (fx)*(1-fy)*(1-fz)*stack[xf+1][yf][zf]+
                        (1-fx)*(fy)*(1-fz)*stack[xf][yf+1][zf]+
                        (1-fx)*(1-fy)*(fz)*stack[xf][yf][zf+1]+
                        (fx)*(fy)*(1-fz)*stack[xf+1][yf+1][zf]+
                        (1-fx)*(fy)*(fz)*stack[xf][yf+1][zf+1]+
                        (fx)*(1-fy)*(fz)*stack[xf+1][yf][zf+1]+
                        (fx)*(fy)*(fz)*stack[xf+1][yf+1][zf+1]
                    );


                }else I=0.0;
                //i=int(img[x][y]);



                //r.print2();
                //std::cout<<i<<" "<<I<<" "<<max(i,int(floor(I+0.5)))<<"\n";

                if(floor(I+0.5)>img[x][y]) img[x][y]=floor(I+0.5);



            }
        }
    }

    return img;

}

*/

stack2D imageFromStack(const stack3D stack, int k){

    int sizeX=stack.size();
    int sizeY=stack[0].size();
    //int sizeZ=stack[0][0].size();

    //std::cout<<"in imageFromStack sizeX="<<sizeX<<" sizeY="<<sizeY<<"\n";

    stack2D im=emptyStack2D(sizeX, sizeY);

    for(int i=0;i<sizeX;i++){
        for(int j=0;j<sizeY;j++){

            im[i][j]=stack[i][j][k];

        }
    }

    return im;

}

double mean(const stack3D& stack){

    int sX=stack.size();
    int sY=stack[0].size();
    int sZ=stack[0][0].size();

    double m=0.0;

    for(int x=0; x<sX; x++) for(int y=0; y<sY; y++) for(int z=0; z<sZ; z++) m+=stack[x][y][z];

    return m/(sX*sY*sZ+0.0);

}

double correlateImagesFromStacks(const stack3D& stack1, const stack3D& stack2, int k1, int k2, int dx, int dy, double mean1, double mean2){

    int sX=stack1.size();
    int sY=stack1[0].size();
    //int sZ=stack1[0][0].size();

    double S=0.0;

    for(int x=0; x<sX; x+=8) for(int y=0; y<sY; y+=8) {

        int x2=(x+sX+dx)%sX;
        int y2=(y+sY+dy)%sY;

        S+=(stack1[x][y][k1]-mean1)*(stack2[x2][y2][k2]-mean2);

    }

    return S;

}

void allignStacks(const stack3D& stackr, const stack3D& stackao, stack3D& stacka, double idx=0, double idy=0, double idz=0){

    int sX=stackr.size();
    int sY=stackr[0].size();
    int sZ=stackr[0][0].size();

    int safety=sZ/16;

    stacka.assign(sX,
        std::vector< std::vector< unsigned char > >(sY,
            std::vector< unsigned char >(sZ, 0 )
        )
    );


    double meanr=mean(stackr);
    double meanao=mean(stackao);

    std::vector<int> indicesz;
    std::vector<int> indicesdx;
    std::vector<int> indicesdy;
    std::vector<double> Svals;

    //int zopt=3,dxopt=0,dyopt=0;
    int r0=3,r=3;
    int ddx=3,ddy=3;

    int dx_opt=0;
    int dy_opt=0;
    int z2_opt=0;

    for(int z=0; z<sZ; z++){


        if( z<(safety+r0) ){
            r=r0+safety;
            z2_opt=z+idz;
            dx_opt=idx;
            dy_opt=idy;
        }else r=r0;

        std::cout<<"alligning stacks z="<<z<<" dzopt="<<z2_opt-z<<" dxopt="<<dx_opt<<" dyopt="<<dy_opt<<"  \r";

        //stack2D imr=imageFromStack(stackr,z);
        std::vector< std::vector<double> > ima;

        ima.assign(sX,
                std::vector< double >(sY, 0.0 )
            );

        int zoptold=floor(z2_opt+0.5);
        int dxoptold=floor(dx_opt+0.5);
        int dyptold=floor(dy_opt+0.5);

        dx_opt=0.0;
        dy_opt=0.0;
        z2_opt=0.0;

        std::vector<double> S;
        std::vector<int> dx_S,dy_S,z2_S;
        S.clear();
        dx_S.clear();
        dy_S.clear();
        z2_S.clear();

        double Stemp;
        double sumS=0.0;
        double minS=1.0e20;
        int i=0;

        for(int z2=zoptold-r; z2<zoptold+r+1; z2++){

            for(int dx=dxoptold-ddx; dx<(dxoptold+ddx+1); dx++) for(int dy=dyptold-ddy; dy<(dyptold+ddy+1); dy++){

                int zz2=z2;

                if(zz2<0) zz2=0;

                if(zz2>sZ-1) zz2=sZ-1;

                i++;
                Stemp=correlateImagesFromStacks(stackr,stackao,z,zz2,dx,dy,meanr,meanao)/(2621440.0);
                S.push_back(Stemp);
                dx_S.push_back(dx);
                dy_S.push_back(dy);
                z2_S.push_back(z2);


                //std::cout<<Stemp<<" "<<dx<<" "<<dy<<" "<<z2<<"    \n";
            }

        }

        int maxi=maximal(S);

        dx_opt=dx_S[maxi];
        dy_opt=dy_S[maxi];
        z2_opt=z2_S[maxi];

        for(int ii=0; ii<i; ii++) if( abs(dx_S[ii]-dx_opt)<2 && abs(dy_S[ii]-dy_opt)<2 && abs(z2_S[ii]-z2_opt)<2) if(S[ii]<minS) minS=S[ii];

        for(int ii=0; ii<i; ii++) if( abs(dx_S[ii]-dx_opt)<2 && abs(dy_S[ii]-dy_opt)<2 && abs(z2_S[ii]-z2_opt)<2){

            sumS+=(S[ii]-minS);

            for(int x=0; x<sX; x++) for(int y=0; y<sY; y++) {

                int xx=(x+dx_S[ii]+sX)%sX;
                int yy=(y+dy_S[ii]+sY)%sY;

                //std::cout<<xx<<" "<<xp<<" "<<yy<<" "<<yp<<" "<<z2f<<"    \n";

                ima[x][y]+=(S[ii]-minS)*stackao[xx][yy][z2_S[ii]];

            }

            //std::cout<<S[ií]<<" "<<minS<<" "<<dx_opt<<" "<<dx_S[ii]<<"    \n";

        }

        for(int x=0; x<sX; x++) for(int y=0; y<sY; y++) stacka[x][y][z]=floor(ima[x][y]/sumS+0.5);

    }

}

void saveStack(const stack3D& stack, std::string fnamebase, std::string fnameending=std::string(".bmp")){


    //int sX=stack.size();
    //int sY=stack[0].size();
    int sZ=stack[0][0].size();

    //char buf[5];

    for(int z=0; z<sZ; z++){

        stack2D im=imageFromStack(stack,z);

        //std::cout<<"in saveStack sizeX="<<im.size()<<" sizeY="<<im[0].size()<<"\n";

        QString qs;
        qs.sprintf("%05d",z);

        //sprintf(buf,"%05d",z);
        std::string fnamez=fnamebase+std::string("_z")+qs.toStdString()+fnameending;
        imwrite(im,fnamez.c_str());

    }

}

std::string renderFilename(std::string fnamebase, int z){


    std::string sd;
    char stemp[200] = "";
    snprintf(stemp, 200, fnamebase.c_str(), z); // I use the safer version of sprintf() -- snprintf()
    sd = stemp; // the contents of sd are now "This is a string!"

    return sd;

}

void readStackWildcard(stack3D& stack, std::string fstr, int jump=1){


    //std::cout<<"fstr = "<<fstr<<"\n\n";

    QString fullstring=QString::fromStdString(fstr);

    fullstring=fullstring.replace("#","*");

    QRegExp rx("(\\\\|\\/)");

    QStringList parts = fullstring.split(rx);
    QString filename = parts.at(parts.size()-1);

    QString dirstring=fullstring.left(fullstring.length()-filename.length());

    //std::cout<<"fstr was split: "<<dirstring.toStdString()<<" ---- "<<filename.toStdString()<<"\n\n";

    QDir dir=QDir(dirstring);

    if(!dir.exists()){

        std::cout<<"ERROR: Couldn't find directory "
            <<dirstring.toStdString()<<"\n";
    }

    QStringList filters;

    if(filename.length()>0) filters<<filename;

    QStringList entryList=dir.entryList(filters);

    if(entryList.length()==0 && dir.exists()){

        std::cout<<"ERROR: Couldn't find any files matching "
            <<filename.toStdString()<<" in directory "
            <<dirstring.toStdString()<<"\n";

    }
    QImage image;

    image.load(dirstring + entryList.first());

    int sX=image.width();
    int sY=image.height();
    int sZ=(entryList.length())/jump;

    std::cout<<"sizes wil be "<<sX<<"  "<<sY<<" "<<sZ<<"    \n";

    stack.assign(sX,
        std::vector< std::vector< unsigned char > >(sY,
            std::vector< unsigned char >(sZ, 0 )
        )
    );

    std::vector<bool> failed;
    failed.assign(sZ,false);

    for(int z=0; z<sZ; z++){

        std::cout<<"loading stack - "<<z<<"   \r";

        image.load(dirstring + entryList.at(z*jump));
        failed[z]=image.isNull();


        if(!failed[z]){

            for(int x=0; x<sX; x++) for(int y=0; y<sY; y++) stack[x][y][z]=(unsigned char)floor(qGray(image.pixel(x, y)));

        }else{

            std::cout<<"\n\nWARNING: The image "<<(dirstring + entryList.at(z*jump)).toStdString()<<" could not be loaded. It will be interpolated from the neighbors.\n";

        }


    }

    // first image loading failed
    if(failed[0]){
        for(int x=0; x<sX; x++) for(int y=0; y<sY; y++) stack[x][y][0]=stack[x][y][1];
    }

    // last image loading failed
    if(failed[sZ-1]){
        for(int x=0; x<sX; x++) for(int y=0; y<sY; y++) stack[x][y][sZ-1]=stack[x][y][sZ-2];
    }

    // all other fails
    for(int z=1; z<sZ-1; z++){
        if(failed[z]){
            for(int x=0; x<sX; x++) for(int y=0; y<sY; y++) stack[x][y][z]=(stack[x][y][z-1]+stack[x][y][z+1])/2;
        }
    }

}

void readStackSprintf(stack3D& stack, std::string fnamebase, int zfrom, int zto, int jump=1){

    //cimg::exception_mode(0);

    //CImg<unsigned int> image(renderFilename(fnamebase, zfrom).c_str());

    QImage image;
    image.load(renderFilename(fnamebase, zfrom).c_str());

    //image.display();

    //int sC=image.spectrum();

    int sX=image.width();
    int sY=image.height();
    int sZ=(zto-zfrom)/jump+1;

    std::cout<<"sizes wil be "<<sX<<"  "<<sY<<" "<<sZ<<"    \n";

    stack.assign(sX,
        std::vector< std::vector< unsigned char > >(sY,
            std::vector< unsigned char >(sZ, 0 )
        )
    );

    //double corr=pow(2,8-bitdepth);

    std::vector<bool> failed;
    failed.assign(sZ,false);

    for(int z=0; z<sZ; z++){

        std::string fname=renderFilename(fnamebase, zfrom+z*jump);

        std::cout<<"loading stack - "<<z<<"   \r";

        image.load(fname.c_str());
        failed[z]=image.isNull();

        if(!failed[z]){
            for(int x=0; x<sX; x++) for(int y=0; y<sY; y++) stack[x][y][z]=(unsigned char)floor(qGray(image.pixel(x, y)));

            //std::cout<<(int)stack[128][128][z]<<"\n";

        }else{

            std::cout<<"\n\nWARNING: The image "<<fname.c_str()<<" could not be loaded. It will be interpolated from the neighbors.\n";

        }


    }

    // first image loading failed
    if(failed[0]){
        for(int x=0; x<sX; x++) for(int y=0; y<sY; y++) stack[x][y][0]=stack[x][y][1];
    }

    // last image loading failed
    if(failed[sZ-1]){
        for(int x=0; x<sX; x++) for(int y=0; y<sY; y++) stack[x][y][sZ-1]=stack[x][y][sZ-2];
    }

    // all other fails
    for(int z=1; z<sZ-1; z++){
        if(failed[z]){
            for(int x=0; x<sX; x++) for(int y=0; y<sY; y++) stack[x][y][z]=(stack[x][y][z-1]+stack[x][y][z+1])/2;
        }
    }
}
