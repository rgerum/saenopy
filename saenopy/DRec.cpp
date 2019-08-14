
#ifndef _DREC_
#define _DREC_

// ------------------------------------------------------------------
// DRec.h
// Version: 2009/Oct/25
// ------------------------------------------------------------------

//#include "RandomNumbers.h"
#include <vector>
#include <algorithm>
#include <sstream>

using namespace std;

double dOfS(string s) {
  istringstream iss (s,istringstream::in);
  double x;
  iss >> x;
    return x;
}

class DRecXY {

public:

    // ------------------------------------------------------------------

    vector<double> xx;
    vector<double> yy;

    // ------------------------------------------------------------------

    DRecXY() {
        xx.clear();
        yy.clear();
    };

    // ------------------------------------------------------------------

    void record(double x, double y) {
        xx.push_back(x);
        yy.push_back(y);
    };

    // ------------------------------------------------------------------

  void reset(void) {
        xx.clear();
        yy.clear();
    };


    void store(const char* fname) {

        unsigned int N;
        char line[81];

    ofstream a_file(fname);

    for (N=0;N<xx.size();N++) {
          sprintf(line,"%e %e\n",xx[N],yy[N]);
          a_file<<line;
        }//N

        a_file.close();
        printf("%s stored.\n",fname);

    };


  void readFromFile(const char* fname) {

    string xStr,yStr;
    // find NMX

    ifstream i_file(fname);
    int NMX=-1;
        do {
            NMX++;
            i_file >> xStr;
            if (i_file.eof()) break;

              //if( (NMX%1000000)==0 ) std::cout<<NMX<<std::endl;
        } while(true);
        NMX--;
        i_file.close();

    int N;

    reset();

    ifstream j_file(fname);

    if(!j_file) std::cout<<"ERROR in readFromFile: \""<<fname<<"\" not found !!!     \n";

      for (N=1;N<=NMX;N++) {
          j_file >> xStr >> yStr;
          record(dOfS(xStr),dOfS(yStr));
      }//N

      j_file.close();

  };

};

// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


class DRec {

public:

    // ------------------------------------------------------------------

    vector<double> data;

    // ------------------------------------------------------------------

    DRec() {
        data.clear();
    };

    // ------------------------------------------------------------------

    void record(double val) {
        data.push_back(val);
    };

    // ------------------------------------------------------------------

    void reset() {
        data.clear();
    };

    // ------------------------------------------------------------------


// void readline(const char* fname) {

//    reset();
//    string xStr;

//	  // find NMX

//    ifstream i_file(fname);
//    int NMX=-1;
//	  do {
//		  NMX++;
//		  i_file >> xStr;
//		  if (i_file.eof()) break;

//            //if( (NMX%1000000)==0 ) std::cout<<NMX<<std::endl;
//	  } while(true);
//	  NMX--;
//	  i_file.close();

//    // load data

//    ifstream j_file(fname);

//    for (int N=0;N<=NMX;N++) {
//      j_file >> xStr;
//      record(dOfS(xStr));

//      //if( (N%1000000)==0 ) std::cout<<N<<std::endl;

//		//printf("N=%d: x=%9.3e, y=%9.3e",N,xx[N],yy[N]);ww();
//    }//N

//    j_file.close();

//    printf("%s read (%d entries).\n",fname,NMX+1);

//	}

 void read(const char* fname) {

       reset();
       string xStr,yStr;

       // find NMX

     ifstream i_file(fname);

     if(!i_file) std::cout<<"ERROR in read: \""<<fname<<"\" not found !!!     \n";

     int NMX=-1;
       do {
           NMX++;
           i_file >> xStr >> yStr;
           if (i_file.eof()) break;

           //if( (NMX%10000)==0 ) std::cout<<NMX<<std::endl;

       } while(true);
       NMX--;
       i_file.close();

     // load data

     ifstream j_file(fname);

     for (int N=0;N<=NMX;N++) {
       j_file >> yStr;
       record(dOfS(yStr));
         //printf("N=%d: x=%9.3e, y=%9.3e",N,xx[N],yy[N]);ww();
           //if( (N%10000)==0 ) std::cout<<N<<std::endl;
     }//N

     j_file.close();

     printf("%s read (%d entries).\n",fname,NMX+1);
     };


    void store(const char* fname) {

        unsigned int N;
        char line[81];

    ofstream a_file(fname);

    for (N=0;N<data.size();N++) {
          sprintf(line,"%e\n",data[N]);
          a_file<<line;
        }//N

        a_file.close();
        printf("%s stored.\n",fname);

    };



}; // end of class


class DRec3D {

public:

    vector<vec3D> data;

    DRec3D() {
    data.clear();
    };


    void record(vec3D val) {
        data.push_back(val);
    };

    void record(double x,double y,double z) {
        data.push_back(vec3D(x,y,z));
    };


  void reset(void) {
        data.clear();
    };

  void store(const char* fname) {

        unsigned int N;
        char line[81];

    ofstream a_file(fname);

    for (N=0;N<data.size();N++) {
          sprintf(line,"%e %e %e\n",data[N].x,data[N].y,data[N].z);
          a_file<<line;
        }//N

        a_file.close();
        printf("%s stored.\n",fname);

    };



void read(const char* fname) {

    reset();
    string xStr,yStr,zStr;

      // find NMX

    ifstream i_file(fname);

    if(!i_file) std::cout<<"ERROR in readFromDatFile: \""<<fname<<"\" not found !!!     \n";

    int NMX=-1;
      do {
          NMX++;
          i_file >> xStr >> yStr >> zStr;
          //cout << xStr << " " << yStr << " " << zStr << "\n";ww();
          if (i_file.eof()) break;
      } while(true);
      NMX--;
      //printf("NMX=%d",NMX);ww();
      i_file.close();

    // load data

    vec3D entry;
    ifstream j_file(fname);

    for (int N=0;N<=NMX;N++) {
      j_file >> xStr >> yStr >> zStr;
      entry=vec3D(dOfS(xStr),dOfS(yStr),dOfS(zStr));
      record(entry);
    }//N

    j_file.close();

    printf("%s read (%d entries).\n",fname,NMX+1);

    };

}; // end of class


#endif
