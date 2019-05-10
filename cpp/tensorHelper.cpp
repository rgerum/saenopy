#include<vector>
#include<iostream>

double operator* (const std::vector< double> &v1, const std::vector<  double> &v2)
{

    if(v1.size()!=v2.size()) std::cout<<"error: wrong size for scalar product"<<std::endl;

    double scalProd(0);
    std::vector<double>::const_iterator i1=v1.begin();
    std::vector<double>::const_iterator i2=v2.begin();
    while( i1!=v1.end() )
    {
        scalProd=scalProd + ( (*i1)*(*i2) );
        i1++;
        i2++;
    }
    return scalProd;
}

/*

double operator* (const std::vector< double> &v1, const vec3D &v2)
{

    if(v1.size()!=3) std::cout<<"error: wrong size for scalar product"<<std::endl;

    double scalProd(0);
    int i;
    for(i=0; i<3; i++){
        scalProd=scalProd + v1[i]*v2[i];
    }
    return scalProd;
}

double operator* (const vec3D &v1, const std::vector< double> &v2)
{

    if(v2.size()!=3) std::cout<<"error: wrong size for scalar product"<<std::endl;

    double scalProd(0);
    int i;
    for(i=0; i<3; i++){
        scalProd=scalProd + v1[i]*v2[i];
    }
    return scalProd;
}

*/

std::vector<double> operator* (const std::vector <std::vector< double > > &M, const std::vector<  double> &v)
{

    if(v.size()!=M[0].size()) std::cout<<"error: wrong size for matrix multiplication"<<std::endl;

    int i_max=v.size();
    int j_max=M.size();

    int i,j;

    std::vector<double> v2(j_max,0.0);

    for(i=0;i<i_max;i++) for(j=0;j<j_max;j++){

        v2[j]+=M[j][i]*v[i];

    }

    return v2;

}

std::vector<double> operator* (const std::vector <std::vector< double > > &M, const vec3D &v)
{

    if(3!=M[0].size()) std::cout<<"error: wrong size for matrix multiplication"<<std::endl;

    int j_max=M.size();

    int j;

    std::vector<double> v2(j_max,0.0);

    for(j=0;j<j_max;j++){

        v2[j]=M[j][0]*v.x+M[j][1]*v.y+M[j][2]*v.z;

    }

    return v2;

}

std::vector< std::vector<double> > operator* (const std::vector <std::vector< double > > &M, const mat3D &m)
{

    if(3!=M[0].size()) std::cout<<"error: wrong size for matrix multiplication"<<std::endl;

    int i_max=M.size();

    int i;

    std::vector< std::vector<double> > M2(i_max,std::vector<double>(3,0.0));

    for(i=0;i<i_max;i++){


        M2[i][0]=M[i][0]*m.xx+M[i][1]*m.yx+M[i][2]*m.zx;
        M2[i][1]=M[i][0]*m.xy+M[i][1]*m.yy+M[i][2]*m.zy;
        M2[i][2]=M[i][0]*m.xz+M[i][1]*m.yz+M[i][2]*m.zz;



    }

    return M2;

}

std::vector< std::vector<double> > operator* (const std::vector <std::vector< double > > &M1, const std::vector <std::vector< double > > &M2)
{

    if(M1.size()!=M2[0].size()) std::cout<<"error: wrong size for matrix multiplication"<<std::endl;

    int i_max=M1.size();
    int j_max=M1[0].size();
    int k_max=M2[0].size();

    int i,j,k;

    std::vector< std::vector<double> > M3(i_max,std::vector<double>(k_max,0.0));

    for(i=0;i<i_max;i++){
        for(j=0;j<j_max;j++){


            for(k=0;k<k_max;k++){

                M3[i][k]+=M1[i][j]*M2[j][k];


            }
        }
    }

    return M3;

}

void mul(const std::vector <std::vector< double > > &M1, const std::vector <std::vector< double > > &M2, std::vector <std::vector< double > > &M3){


    if(M1.size()!=M2[0].size()) std::cout<<"error: wrong size for matrix multiplication"<<std::endl;

    int i_max=M1.size();
    int j_max=M1[0].size();
    int k_max=M2[0].size();

    int i,j,k;

    for(i=0;i<i_max;i++){
        for(k=0;k<k_max;k++){

            M3[i][k]=0.0;

            for(j=0;j<j_max;j++){

                M3[i][k]+=M1[i][j]*M2[j][k];


            }
        }
    }


}

void mul(const std::vector <std::vector< double > > &M, const std::vector<  double> &v,  std::vector<  double> &v2){

    //if(v.size()!=M[0].size()) std::cout<<"error: wrong size for matrix multiplication"<<std::endl;

    int i_max=v.size();
    int j_max=M.size();

    int i,j;



    //std::vector<double> v2(j_max,0.0);

    for(j=0;j<j_max;j++){

        v2[j]=0.0;

        for(i=0;i<i_max;i++){

            v2[j]+=M[j][i]*v[i];

        }

    }

}

void emul(const std::vector < double  > &v1, const std::vector<  double> &v2,  std::vector<  double> &v){

    int i;
    int i_max=v1.size();

    for(i=0;i<i_max;i++) v[i]=v1[i]*v2[i];


}

double imul(const std::vector < double  > &v1, const std::vector<  double> &v2){

    double e=0.0;
    int i;

    int i_max=v1.size();

    for(i=0;i<i_max;i++) {
        e+=v1[i]*v2[i];

        //if(v1[i]*v2[i]!=v1[i]*v2[i]) {

            //std::cout<<i<<": "<<v1[i]<<" "<<v2[i]<<std::endl;

            //std::cin>>i;

        //}
    }

    return e;


}

void setequal(const std::vector < double  > &v1, std::vector<  double> &v2){

    int i;
    int i_max=v1.size();
    for(i=0;i<i_max;i++) v2[i]=v1[i];

}

void plusminus(const std::vector < double  > &v1, double d, const std::vector < double  > &v2, std::vector<  double> &v){

    //v=v1+d*v2

    int i=0;
    int i_max=v1.size();
    for(; i<i_max-4; i+=4) {
        v[i]=v1[i]+d*v2[i];
        v[i+1]=v1[i+1]+d*v2[i+1];
        v[i+2]=v1[i+2]+d*v2[i+2];
        v[i+3]=v1[i+3]+d*v2[i+3];
    }
    for(; i<i_max; i++) v[i]=v1[i]+d*v2[i];

}

void readFromDatFile(std::string fname, std::vector <std::vector< double > > &M){

    M.clear();

    char stemp[200] = "";

    size_t i = -1;

    //bool lineended=false;

    std::string word;
    std::ifstream infile(fname.c_str());

    if(!infile) std::cout<<"ERROR in readFromDatFile: \""<<fname.c_str()<<"\" not found !!!     \n";

    while ( getline(infile, word) )
    {

        i++;

        M.push_back(std::vector<double>());
        M[i].clear();

        strcpy(stemp, word.c_str());

        char * str = stemp;
        char * pch;
        //printf ("Splitting string \"%s\" into tokens:\n",str);
        pch = strtok (str," ");
        while (pch != NULL)
        {
            M[i].push_back(atof(pch));
            //printf ("%s\n",pch);
            pch = strtok (NULL, " ,;");
        }


        /*
        if(word!=""){

            if(lineended){

                //std::cout<<"},\n{";

                i++;

                M.push_back(std::vector<double>());
                M[i].clear();

            }

            //std::cout << "\"" <<  dOfS(word) << "\" , ";

            M[i].push_back(dOfS(word));

            if(word.find("\n")!=std::string::npos){

                lineended=true;

            }else{

                lineended=false;
            }

        }*/

    }

    std::cout<<fname<<" read ("<<M.size()<<" x "<<M[0].size()<<" entries)   \n";

}

void writeToDatFile(std::string fname,const std::vector <std::vector< double > > &M){


    std::ofstream fout(fname.c_str());

    int i_max=M.size();
    int j_max=M[0].size();

    int i,j;

    for(j=0;j<j_max;j++){

        for(i=0;i<i_max;i++){

            fout<<M[i][j];

        }

        fout<<"\n";

    }

}

