void mul(std::vector< std::unordered_map<size_t,mat3D> >& M , std::vector< vec3D >& a, std::vector< vec3D >& b){

    int N_c=M.size();

    int c1;

    std::unordered_map<size_t,mat3D>::iterator it;

    for(c1=0; c1<N_c; c1++){

        b[c1]=vec3D(0.0,0.0,0.0);

        for(it=M[c1].begin(); it!=M[c1].end(); it++) b[c1]+=(*it).second*a[(*it).first];

        //if(norm(b[c1])>0) b[c1].print2();

    }

}

mat3D dot(std::unordered_map<size_t,mat3D>& C, std::unordered_map<size_t,mat3D>& R){

    mat3D result=mat3D(0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0);

    for(std::unordered_map<size_t,mat3D>::iterator it=C.begin(); it!=C.end(); it++) if(R.find((*it).first)!=R.end()) result+=C[(*it).first]*R[(*it).first];

    return result;

}

void mulT(std::vector< std::unordered_map<size_t,mat3D> >& M , std::vector< vec3D >& a, std::vector< vec3D >& b){


    int N_c=M.size();

    int c1;

    std::unordered_map<size_t,mat3D>::iterator it;

    //std::cout<<"check inside mulT "<<N_c<<"       \n";

    b.resize(N_c);

    for(c1=0; c1<N_c; c1++) {

        //std::cout<<c1<<"\n";

        b[c1]=vec3D(0.0,0.0,0.0);

    }

    for(c1=0; c1<N_c; c1++) for(it=M[c1].begin(); it!=M[c1].end(); it++) {

        //std::cout<<c1<<" "<<(*it).first<<"\n";

        b[(*it).first]+=(*it).second*a[c1];

    }


}

