size_t minimal(const std::vector<double> &S){

    int i;
    int i_max=S.size();

    double min=S[0];
    size_t mini=0;

    for(i=1; i<i_max; i++){

        if(S[i]<min) { mini=i; min=S[i]; }

    }

    return mini;

}

size_t maximal(const std::vector<double> &S){

    int i;
    int i_max=S.size();

    double max=S[0];
    size_t maxi=0;

    for(i=1; i<i_max; i++){

        if(S[i]>max) { maxi=i; max=S[i]; }

    }

    return maxi;

}



