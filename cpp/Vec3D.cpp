

// ------------------------------------------------------------------
// vec3d_class
// Version: 2009/Oct/01
// ------------------------------------------------------------------

#ifndef vec3D_h
#define vec3D_h

#include <math.h>
#include <stdio.h>


#define PI 3.141592653589793238462643383279502884197169399375105

class vec3D {

public:

    // Data

    double x, y, z;


    // Ctors

    vec3D( double InX, double InY, double InZ ) : x( InX ), y( InY ), z( InZ ) {}
    vec3D( ) : x(0), y(0), z(0) {}

    // Operator Overloads

    bool operator== (const vec3D& V2) const {
        return (x == V2.x && y == V2.y && z == V2.z);
  }

    vec3D operator+ (const vec3D& V2) const {
        return vec3D( x + V2.x,  y + V2.y,  z + V2.z);
  }

    vec3D operator- (const vec3D& V2) const {
        return vec3D( x - V2.x,  y - V2.y,  z - V2.z);
  }

  vec3D operator- ( ) const {
        return vec3D(-x, -y, -z);
  }

    vec3D operator/ (double S ) const {
        double fInv = 1.0 / S;
        return vec3D (x * fInv , y * fInv, z * fInv);
  }

    vec3D operator/ (const vec3D& V2) const {
        return vec3D (x / V2.x,  y / V2.y,  z / V2.z);
  }

    vec3D operator* (const vec3D& V2) const {
        return vec3D (x * V2.x,  y * V2.y,  z * V2.z);
  }

    vec3D operator* (double S) const {
        return vec3D (x * S,  y * S,  z * S);
  }

    void operator+= ( const vec3D& V2 ) {
        x += V2.x;
        y += V2.y;
        z += V2.z;
  }

    void operator-= ( const vec3D& V2 ) {
        x -= V2.x;
        y -= V2.y;
        z -= V2.z;
  }

  void operator*= ( const double s ) {
        x *= s;
        y *= s;
        z *= s;
  }

    void operator/= ( const double s ) {
        x /= s;
        y /= s;
        z /= s;
  }


    double operator[] ( int i ) {
        if ( i == 0 ) return x;
        else if ( i == 1 ) return y;
        else return z;
  }

    // Functions


  double Dot( const vec3D &V1 ) const {
        return V1.x*x + V1.y*y + V1.z*z;
  }

  vec3D CrossProduct( const vec3D &V2 ) const {
    return vec3D(
      y * V2.z  -  z * V2.y,
      z * V2.x  -  x * V2.z,
      x * V2.y  -  y * V2.x
    );
  }

  // ------------------------------------------------------------------


  void MakeFromPolar(double r, double theta, double phi) {
    double ct=cos(theta);
    double st=sin(theta);
    double cp=cos(phi);
    double sp=sin(phi);
    x=r*st*cp;
    y=r*st*sp;
    z=r*ct;
  }



  double radius(void) {
    return sqrt(x*x+y*y+z*z);
  }

  double polarAngle(void) {
    return acos(z/sqrt(x*x+y*y+z*z));
  }

  double azimuthalAngle(void) {
    return atan2(y,x);
  }

   void print(void) {
    printf("(x = %.3f, y = %.3f, z = %.3f)\n",x,y,z);
    printf("(r = %.3f, theta = %.3f pi, phi = %.3f pi)\n",
           radius(),polarAngle()/PI,azimuthalAngle()/PI);
  }

  void print9(void) {
    printf("(x = %.9f, y = %.9f, z = %.9f)\n",x,y,z);
    printf("(r = %.9f, theta = %.9f pi, phi = %.9f pi)\n",
           radius(),polarAngle()/PI,azimuthalAngle()/PI);
  }

   void print2(void) {
    printf("(%.3f , %.3f ,  %.3f)\n",x,y,z);
    //printf("(r = %.3f, theta = %.3f pi, phi = %.3f pi)\n",
           //radius(),polarAngle()/pi,azimuthalAngle()/PI);
  }

  void RotateAroundXAxis(double alpha) {
    double s=sin(alpha);
    double c=cos(alpha);
    double y2=c*y-s*z;
    double z2=s*y+c*z;
    y=y2; z=z2;
  }

   void RotateAroundYAxis(double alpha) {
    double s=sin(alpha);
    double c=cos(alpha);
    double x2=c*x+s*z;
    double z2=c*z-s*x;
    x=x2; z=z2;
  }

   void RotateAroundZAxis(double alpha) {
    double s=sin(alpha);
    double c=cos(alpha);
    double x2=c*x-s*y;
    double y2=s*x+c*y;
    x=x2; y=y2;
  }

   void Rotate2Angles(double dTheta, double dPhi) {

    double theta1=polarAngle();
    double phi1=azimuthalAngle();

    RotateAroundZAxis(-phi1);
    RotateAroundYAxis(-theta1);

    RotateAroundYAxis(dTheta);
    RotateAroundZAxis(dPhi);

    RotateAroundZAxis(phi1);
    RotateAroundYAxis(theta1);

  }

  void ChangeLengthTo(double rNew) {
    double factor=rNew/sqrt(x*x+y*y+z*z);
    x*=factor;
    y*=factor;
    z*=factor;
  }

  // ------------------------------------------------------------------


}; // end of class


double norm(vec3D a) {
  return ( a.x*a.x + a.y*a.y + a.z*a.z );
}

double abs(vec3D a) {
  return sqrt( a.x*a.x + a.y*a.y + a.z*a.z );
}

double scalarProd(vec3D a, vec3D b) {
  return ( a.x*b.x + a.y*b.y + a.z*b.z );
}


#endif

/*
void Test_vec3D(void) {
  vec3D a,b,c;

  a=vec3D(1.0,2.0,3.0);
  b=vec3D(4.0,5.0,6.0);
  c=(a+b)/2.0;
  c.print();
}
*/



