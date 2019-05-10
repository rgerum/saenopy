
// ------------------------------------------------------------------
// mat3d_class
// Version: 2011/Feb/14
// Author: Julian Steinwachs
// ------------------------------------------------------------------

#ifndef mat3D_h
#define mat3D_h

#include <math.h>

class mat3D {

public:

    // Data

    double xx,xy,xz,yx,yy,yz,zx,zy,zz;


    // Ctors

    mat3D( vec3D InX, vec3D InY, vec3D InZ ){

        xx=InX.x;
        xy=InY.x;
        xz=InZ.x;
        yx=InX.y;
        yy=InY.y;
        yz=InZ.y;
        zx=InX.z;
        zy=InY.z;
        zz=InZ.z;


    }



    mat3D(double ixx,double ixy,double ixz,double iyx,double iyy,double iyz,double izx,double izy,double izz){

        xx=ixx;
        xy=ixy;
        xz=ixz;
        yx=iyx;
        yy=iyy;
        yz=iyz;
        zx=izx;
        zy=izy;
        zz=izz;

    }



    mat3D() : xx(0),xy(0),xz(0),yx(0),yy(0),yz(0),zx(0),zy(0),zz(0) {}

    // Operator Overloads



    vec3D operator* (const vec3D& V2) const {
        return vec3D(
               xx*V2.x+xy*V2.y+xz*V2.z,
               yx*V2.x+yy*V2.y+yz*V2.z,
               zx*V2.x+zy*V2.y+zz*V2.z
               );
    }

    mat3D operator* (const mat3D& M2) const {
        return mat3D(
               xx*M2.xx+xy*M2.yx+xz*M2.zx,
               xx*M2.xy+xy*M2.yy+xz*M2.zy,
               xx*M2.xz+xy*M2.yz+xz*M2.zz,
               yx*M2.xx+yy*M2.yx+yz*M2.zx,
               yx*M2.xy+yy*M2.yy+yz*M2.zy,
               yx*M2.xz+yy*M2.yz+yz*M2.zz,
               zx*M2.xx+zy*M2.yx+zz*M2.zx,
               zx*M2.xy+zy*M2.yy+zz*M2.zy,
               zx*M2.xz+zy*M2.yz+zz*M2.zz
               );
    }


    mat3D operator- (const mat3D& M2) const {
        return mat3D(
               xx-M2.xx,
               xy-M2.xy,
               xz-M2.xz,
               yx-M2.yx,
               yy-M2.yy,
               yz-M2.yz,
               zx-M2.zx,
               zy-M2.zy,
               zz-M2.zz
               );
    }

    mat3D operator* (const double c) const {
        return mat3D(
               c*xx,
               c*xy,
               c*xz,
               c*yx,
               c*yy,
               c*yz,
               c*zx,
               c*zy,
               c*zz
               );
    }

    void operator+= ( const mat3D& M2 ) {
        xx += M2.xx;
        yx += M2.yx;
        zx += M2.zx;
        xy += M2.xy;
        yy += M2.yy;
        zy += M2.zy;
        xz += M2.xz;
        yz += M2.yz;
        zz += M2.zz;
  }

    vec3D operator[] ( int i ) {
        if ( i == 0 ) return vec3D(xx,xy,xz);
        else if ( i == 1 ) return vec3D(yx,yy,yz);
        else return vec3D(zx,zy,zz);
  }

    //Functions

    void print(){

        std::cout<<xx<<" , "<<xy<<" , "<<xz<<std::endl;
        std::cout<<yx<<" , "<<yy<<" , "<<yz<<std::endl;
        std::cout<<zx<<" , "<<zy<<" , "<<zz<<std::endl<<std::endl;

    }

    mat3D transpose(){

        return mat3D(xx,yx,zx,xy,yy,zy,xz,yz,zz);

    }

    double det(){

        return xx*yy*zz
                +xy*yz*zx
                +xz*zy*yx
                -xx*yz*zy
                -yy*xz*zx
                -zz*xy*yx;

    }

    mat3D inv(){

        double deter=det();
        double txx=(yy*zz-yz*zy)/deter;
        double txy=(xz*zy-xy*zz)/deter;
        double txz=(xy*yz-yy*xz)/deter;
        double tyx=(yz*zx-yx*zz)/deter;
        double tyy=(xx*zz-xz*zx)/deter;
        double tyz=(xz*yx-xx*yz)/deter;
        double tzx=(yx*zy-yy*zx)/deter;
        double tzy=(xy*zx-xx*zy)/deter;
        double tzz=(xx*yy-xy*yx)/deter;

        return mat3D(txx,txy,txz,tyx,tyy,tyz,tzx,tzy,tzz);

    }

};


mat3D Outer( const vec3D &V1 , const vec3D &V ) {
    return mat3D(V.x*V1.x,V.y*V1.x,V.z*V1.x,V.x*V1.y,V.y*V1.y,V.z*V1.y,V.x*V1.z,V.y*V1.z,V.z*V1.z);
}


mat3D rotMatX(double phi){

    return mat3D(1.0,0.0,0.0,
                 0.0,cos(phi),-sin(phi),
                 0.0,sin(phi),cos(phi));


}

mat3D rotMatY(double phi){

    return mat3D(cos(phi),0.0,sin(phi),
                 0.0,1.0,0.0,
                 -sin(phi),0.0,cos(phi));

}

mat3D rotMatZ(double phi){

    return mat3D(cos(phi),-sin(phi),0.0,
                 sin(phi),cos(phi),0.0,
                 0.0,0.0,1.0);


}

mat3D outerprod(vec3D a, vec3D b){

    return mat3D(a.x*b.x,a.x*b.y,a.x*b.z,
           a.y*b.x,a.y*b.y,a.y*b.z,
           a.z*b.x,a.z*b.y,a.z*b.z);


}

#endif






