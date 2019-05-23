
void buildBeams(std::vector<vec3D>& beams, int N){

    //std::vector<vec3D> beams=std::vector<vec3D>();

    beams.clear();

    double theta,phi;

    vec3D newbeam;

    double pi=3.141592653589793238462643383279502884197169399375105;

    int jmax,j;

    for(int i=0; i<N; i++){

        theta=(2*pi/N)*i;

        jmax=floor(N*sin(theta)+0.5);

        for(j=0; j<jmax; j++){

            phi=(2*pi/jmax)*j;

            newbeam.MakeFromPolar(1.0,theta,phi);

            beams.push_back(newbeam);

        }

    }

    //return beams;

}

void saveBeams(std::vector<vec3D>& beams,const char* fname){

    DRec3D beamsrec=DRec3D();

    beamsrec.data=beams;

    beamsrec.store(fname);

}
