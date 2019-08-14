
//using namespace cimg_library;

typedef std::vector< std::vector<unsigned int> > stack2D;

/*
stack2D imToVecvec(const CImg<unsigned int>& im, int sizeX, int sizeY, int ch=0){

    stack2D vecvec;

    vecvec.assign(sizeX,std::vector<unsigned int>(sizeY,0));

    int x,y;

    for(x=0;x<sizeX;x++){
        for(y=0;y<sizeY;y++){

            vecvec[x][y]=im(x,y,0,ch);

        }
    }

    return vecvec;

}
*/
stack2D resizeStack2D(stack2D im, int grain){

    int newSizeX=im.size()/grain;
    int newSizeY=im[1].size()/grain;

    stack2D im_new;

    im_new.assign(newSizeX,std::vector<unsigned int>(newSizeY,0));

    int x,y,dx,dy;

    float sum;

    for(x=0;x<newSizeX;x++){
        for(y=0;y<newSizeY;y++){

            sum=0;

            for(dx=0;dx<grain;dx++){
                for(dy=0;dy<grain;dy++){

                    sum+=im[grain*x+dx][grain*y+dy]/(grain*grain+0.0);

                }

            //sum/=grain*grain;

            im_new[x][y]=lround(sum);

            }

        }

    }

    return im_new;

}
/*
stack2D imread(const char* filename){

    CImg<unsigned int> cim(1024,1024,1,1);
    cim.load(filename);
    stack2D im;
    im=imToVecvec(cim,1024,1024);

    return im;

}
*/

void imwrite(stack2D im,const char* filename){


//    std::cout<<"in imwrite sizeX="<<im.size()<<" sizeY="<<im[0].size()<<"\n";

    int sizeX=im.size();
    int sizeY=im[0].size();

    //std::cout<<filename<<" size: "<<sizeX<<" "<<sizeY<<"\n";

    QImage img(sizeX,sizeY,QImage::Format_RGB888);
    img.fill(0);

    int x,y;

    for(x=0;x<sizeX;x++){
        for(y=0;y<sizeY;y++){

            img.setPixel(x,y,qRgb( im[x][y], im[x][y], im[x][y] ));

            //if(im[x][y]!=0) std::cout << "check";

        }
    }

    img.save(filename);


}

stack2D emptyStack2D(int sizeX, int sizeY){

    stack2D stack;

    stack.assign(sizeX,std::vector<unsigned int>(sizeY,0));

    //std::cout<<"in emptyStack2D sizeX="<<stack.size()<<" sizeY="<<stack[0].size()<<"\n";

    return stack;

}

float getMean(stack2D& I){

    int sizeX=I.size();
    int sizeY=I[1].size();

    int i,j;

    int sum=0;

    for(i=0;i<sizeX;i++){
        for(j=0;j<sizeY;j++){

            sum+=I[i][j];

        }
    }

    return sum/(sizeX*sizeY+0.0);

}

float getVariance(stack2D& I, float mean){

    int sizeX=I.size();
    int sizeY=I[1].size();

    int i,j;

    float sum=0;

    for(i=0;i<sizeX;i++){
        for(j=0;j<sizeY;j++){

            sum+=(I[i][j]-mean)*(I[i][j]-mean);

        }
    }

    return sqrt(sum/(sizeX*sizeY+0.0));

}

stack2D smoothen(stack2D& im){


    int sizeX=im.size();
    int sizeY=im[1].size();

    stack2D im2=emptyStack2D(sizeX,sizeY);

    int kernel[5][5]={  {0,1,1,1,0},
                        {1,1,2,1,1},
                        {1,2,3,2,1},
                        {1,1,2,1,1},
                        {0,1,1,1,0} };

    int x,y,dx,dy;

    //int i_new,i_norm;

    for(x=0;x<sizeX;x++){
        for(y=0;y<sizeY;y++){

            for(dx=-2;dx<3;dx++){
                for(dy=-2;dy<3;dy++){
                    if( (x+dx)>-1 && (y+dy)>-1 && (x+dx)<(sizeX) && (y+dy)<(sizeY) ){

                        im2[x+dx][y+dy]+=kernel[dx+2][dy+2]*im[x+dx][y+dy];



                    }
                }
            }

        }
    }

    return im2;

}

float crosscorrelateImages(stack2D& im1, stack2D& im2){

    float mean1=getMean(im1);
    float mean2=getMean(im2);

    float var1=getVariance(im1,mean1);
    float var2=getVariance(im2,mean2);

    int sizeX=im1.size();
    int sizeY=im1[1].size();

    int i,j;

    float sum=0;

    for(i=0;i<sizeX;i++){
        for(j=0;j<sizeY;j++){

            sum+=((im1[i][j]-mean1)*(im2[i][j]-mean2))/(var1*var2);

        }
    }

    return sum/(sizeX*sizeY+0.0);
}

/*

stack2D applyDeformation(const stack2D im1, MGField2D& U){

    int sizeX=im1.size();
    int sizeY=im1[1].size();

    stack2D im2=emptyStack2D(sizeX,sizeY);

    int offset=0;

    int x1,x2,y1,y2;

    //float rmin=0;

    //char intensity;

    //int r;

    int dx,dy;

    bool filled[sizeX][sizeY];

    //std::cout << std::endl;

    for(x2=0+offset;x2<sizeX+offset;x2++){
        for(y2=0+offset;y2<sizeY+offset;y2++){

            dx=lround(real(U.field[x2][y2]));
            dy=lround(imag(U.field[x2][y2]));

            //dx=0;
            //dy=0;

            //std::cout << dx << " " << dy;

            if(y2-offset+dy<sizeY && y2-offset+dy>0 && x2-offset+dx<sizeX && x2-offset+dx>0){

                //std::cout << "check";

                im2[x2-offset+dx][y2-offset+dy]=im1[x2-offset][y2-offset];

                filled[x2-offset+dx][y2-offset+dy]=true;

            }

            //std::cout << std::endl;

        }
        //std::cout << x2 << std::endl;
    }

    //std::cout << "finished first round" << std::endl;

    int sum;



    for(x1=1+offset;x1<sizeX+offset-1;x1++){
        for(y1=1+offset;y1<sizeY+offset-1;y1++){

            if(!filled[x1-offset][y1-offset]){

                //std::cout << x1 << " " << y1;

                sum=0;

                sum+=int(im2[x1-offset-1][y1-offset-1]);
                sum+=int(im2[x1-offset][y1-offset-1]);
                sum+=int(im2[x1-offset+1][y1-offset-1]);
                sum+=int(im2[x1-offset-1][y1-offset]);
                sum+=int(im2[x1-offset+1][y1-offset]);
                sum+=int(im2[x1-offset-1][y1-offset+1]);
                sum+=int(im2[x1-offset][y1-offset+1]);
                sum+=int(im2[x1-offset+1][y1-offset+1]);

                sum/=8;

                im2[x1-offset][y1-offset]=sum;

            }

        }

        //std::cout << x1-offset << std::endl;
    }

    return im2;

}

stack2D applyDeformationInverse(const stack2D im1, MGField2D& U){

    int sizeX=im1.size();
    int sizeY=im1[1].size();

    stack2D im2=emptyStack2D(sizeX,sizeY);

    int offset=0;

    int x1,x2,y1,y2;

    //float rmin=0;

    //char intensity;

    //int r;

    float dx,dy;

    //std::cout << std::endl;

    for(x2=0+offset;x2<sizeX+offset;x2++){
        for(y2=0+offset;y2<sizeY+offset;y2++){

            dx=real(U.field[x2][y2]);
            dy=imag(U.field[x2][y2]);

            int dxf=floor(dx);

            int dyf=floor(dy);

            float fdx=dx-dxf;
            float fdy=dy-dyf;

            //dx=0;
            //dy=0;

            //std::cout << dx << " " << dy;

            if( (y2+dyf+1)<sizeY && (y2+dyf)>0 && (x2+dxf+1)<sizeX && (x2+dxf)>0){

                //std::cout << "check";


                im2[x2][y2]=((1-fdx)*(1-fdy)*im1[x2+dxf][y2+dyf]
                            +fdx*(1-fdy)*im1[x2+dxf+1][y2+dyf]
                            +(1-fdx)*fdy*im1[x2+dxf][y2+dyf+1]
                            +fdx*fdy*im1[x2+dxf+1][y2+dyf+1]);

            }

            //std::cout << std::endl;

        }
        //std::cout << x2 << std::endl;
    }

    //std::cout << "finished first round" << std::endl;

    int sum;

    return im2;

}


float testDeformation(stack2D& im1, stack2D& im2, float mean1, float mean2, float var1, float var2, int x, int y, float dx, float dy){

    int dxf=floor(dx);

    int dyf=floor(dy);

    float fdx=dx-dxf;
    float fdy=dy-dyf;

    if((x+dxf)>0 && (y+dyf)>0 && (y+dyf+5)<im2[1].size() && (x+dxf+5)<im2.size()){

        float r1[4][4];
        float r2[4][4];

        int xi,yj;

        int sum1=0;
        int sum2=0;

        for(xi=0;xi<4;xi++){
            for(yj=0;yj<4;yj++){

                r1[xi][yj]=(im1[x+xi][y+yj]);
                r2[xi][yj]=(

                            (1-fdx)*(1-fdy)*im2[x+xi+dxf][y+yj+dyf]
                            +fdx*(1-fdy)*im2[x+xi+dxf+1][y+yj+dyf]
                            +(1-fdx)*fdy*im2[x+xi+dxf][y+yj+dyf+1]
                            +fdx*fdy*im2[x+xi+dxf+1][y+yj+dyf+1]

                        );

                sum1+=r1[xi][yj];
                sum2+=r2[xi][yj];

            }
        }



        for(xi=0;xi<4;xi++){
            for(yj=0;yj<4;yj++){

                r1[xi][yj]-=sum1/16.0;
                r2[xi][yj]-=sum2/16.0;

            }
        }

        float var1=0;
        float var2=0;

        for(xi=0;xi<4;xi++){
            for(yj=0;yj<4;yj++){

                var1+=r1[xi][yj]*r1[xi][yj];
                var2+=r2[xi][yj]*r2[xi][yj];

            }
        }

        var1=sqrt(var1);
        var2=sqrt(var2);

        for(xi=0;xi<4;xi++){
            for(yj=0;yj<4;yj++){

                r1[xi][yj]/=var1;
                r2[xi][yj]/=var2;

            }
        }



        float S=0;

        for(xi=0;xi<4;xi++){
            for(yj=0;yj<4;yj++){

                S+=r1[xi][yj]*r2[xi][yj];

            }
        }

        //std::cout << "inside bounds" << std::endl;

        if( !(S!=S) ) return S;
        else return 0.0;

    }else{

        //std::cout << "out of bounds" << std::endl;
        return 0.0;

    }
}



float OverlapAt2D(MGField2D& U, stack2D& im1_g, stack2D& im2_g, float mean1, float mean2, float var1, float var2, int i, int j, vec2D du){

    int offset=0;

    int dx=lround(real(U.field[i][j]+du)/(U.grain/4));
    int dy=lround(imag(U.field[i][j]+du)/(U.grain/4));

    int x=4*(i-offset);
    int y=4*(j-offset);

    //if(x+dx>0 && y+dy>0 && y+dy+4<U.meshpoints*4 && x+dx+4<U.meshpoints*4)
    return testDeformation(im1_g, im2_g,mean1,mean2,var1,var2,x-2,y-2,dx,dy);
    //else{

        //std::cout << "out of bounds" << std::endl;

        //return 0;

    //}

}

*/

