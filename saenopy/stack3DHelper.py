import matplotlib.pyplot as plt
import numpy as np
from typing import Sequence

# using namespace cimg_library

"""
stack3D resize(const stack3D& stack1, int g){

    stack3D stack2=stack3D()

    int sX=stack1.size()
    int sY=stack1[0].size()
    int sZ=stack1[0][0].size()

    stack2.assign(sX/g,
        std::vector< std::vector< unsigned char > >(sY/g,
            std::vector< unsigned char >(sZ/g, 0 )
        )
    )

    int i,j,k,ii,jj,kk

    int ggg=g*g*g

    double sum=0.0

    for(i=0; i<(sX/g); i++) for(j=0; j<(sY/g); j++) for(k=0; k<(sZ/g); k++){

        sum=0.0

        for(ii=0; ii<g; ii++) for(jj=0; jj<g; jj++) for(kk=0; kk<g; kk++){


            sum+=stack1[g*i+ii][g*j+jj][g*k+kk]+0.0


        }

        stack2[i][j][k]=sum/ggg

        # std::cout<<int(stack2[i][j][k])<<std::endl

    }

    return stack2

}


double crosscorrelateSections(const stack3D& stack1, const stack3D& stack2, vec3D r, vec3D du, int big=1){

    int sgX=int(CFG["VB_SX"])*big
    int sgY=int(CFG["VB_SY"])*big
    int sgZ=int(CFG["VB_SZ"])*big

    int sX=stack1.size()
    int sY=stack1[0].size()
    int sZ=stack1[0][0].size()

    bool in=true

    if(r[0]<1) in=false
    if(r[1]<1) in=false
    if(r[2]<1) in=false
    if(r[0]>(sX-1)) in=false
    if((r+du)[1]>(sY-1)) in=false
    if((r+du)[2]>(sZ-1)) in=false
    if((r+du)[0]<1) in=false
    if((r+du)[1]<1) in=false
    if((r+du)[2]<1) in=false
    if((r+du)[0]>(sX-1)) in=false
    if((r+du)[1]>(sY-1)) in=false
    if((r+du)[2]>(sZ-1)) in=false

    if(in){

        int xf=floor(r[0])
        int yf=floor(r[1])
        int zf=floor(r[2])

        double fx=r[0]-xf
        double fy=r[1]-yf
        double fz=r[2]-zf

        int dxf=floor(du[0]+r[0])
        int dyf=floor(du[1]+r[1])
        int dzf=floor(du[2]+r[2])

        double fdx=du[0]+r[0]-dxf
        double fdy=du[1]+r[1]-dyf
        double fdz=du[2]+r[2]-dzf

        int ii,jj,kk

        std::vector< std::vector< std::vector<double> > > substack1,substack2

        substack1.assign(sgX,
            std::vector< std::vector< double > >(sgY,
                std::vector< double >(sgZ, 0.0 )
            )
        )

        substack2.assign(sgX,
            std::vector< std::vector< double > >(sgY,
                std::vector< double >(sgZ, 0.0 )
            )
        )

        int sum1=0
        int sum2=0

        int iii,jjj,kkk

        # std::cout<<xf<<","<<yf<<","<<zf<<","<<dxf<<","<<dyf<<","<<dzf<<" | "<<sX<<","<<sY<<","<<sZ<<" \n"

        if( xf>(sgX/2) && xf<(sX-sgX/2) && yf>(sgY/2) && yf<(sY-sgY/2) && zf>(sgZ/2) && zf<(sZ-sgZ/2) && dxf>(sgX/2) && dxf<(sX-sgX/2) && dyf>(sgY/2) && dyf<(sY-sgY/2) && dzf>(sgZ/2) && dzf<(sZ-sgZ/2) ){

            for(ii=0; ii<sgX; ii++) for(jj=0; jj<sgY; jj++) for(kk=0; kk<sgZ; kk++){

                iii=ii-sgX/2
                jjj=jj-sgY/2
                kkk=kk-sgZ/2

                substack1[ii][jj][kk]=(
                                        (1-fx)*(1-fy)*(1-fz)*stack1[iii+xf][jjj+yf][kkk+zf]+
                                        (fx)*(1-fy)*(1-fz)*stack1[iii+xf+1][jjj+yf][kkk+zf]+
                                        (1-fx)*(fy)*(1-fz)*stack1[iii+xf][jjj+yf+1][kkk+zf]+
                                        (1-fx)*(1-fy)*(fz)*stack1[iii+xf][jjj+yf][kkk+zf+1]+
                                        (fx)*(fy)*(1-fz)*stack1[iii+xf+1][jjj+yf+1][kkk+zf]+
                                        (1-fx)*(fy)*(fz)*stack1[iii+xf][jjj+yf+1][kkk+zf+1]+
                                        (fx)*(1-fy)*(fz)*stack1[iii+xf+1][jjj+yf][kkk+zf+1]+
                                        (fx)*(fy)*(fz)*stack1[iii+xf+1][jjj+yf+1][kkk+zf+1]
                                    )

                substack2[ii][jj][kk]=(
                                        (1-fdx)*(1-fdy)*(1-fdz)*stack2[iii+dxf][jjj+dyf][kkk+dzf]+
                                        (fdx)*(1-fdy)*(1-fdz)*stack2[iii+dxf+1][jjj+dyf][kkk+dzf]+
                                        (1-fdx)*(fdy)*(1-fdz)*stack2[iii+dxf][jjj+dyf+1][kkk+dzf]+
                                        (1-fdx)*(1-fdy)*(fdz)*stack2[iii+dxf][jjj+dyf][kkk+dzf+1]+
                                        (fdx)*(fdy)*(1-fdz)*stack2[iii+dxf+1][jjj+dyf+1][kkk+dzf]+
                                        (1-fdx)*(fdy)*(fdz)*stack2[iii+dxf][jjj+dyf+1][kkk+dzf+1]+
                                        (fdx)*(1-fdy)*(fdz)*stack2[iii+dxf+1][jjj+dyf][kkk+dzf+1]+
                                        (fdx)*(fdy)*(fdz)*stack2[iii+dxf+1][jjj+dyf+1][kkk+dzf+1]
                                    )

                sum1+=substack1[ii][jj][kk]
                sum2+=substack2[ii][jj][kk]


            }

            double sssg=(sgX*sgY*sgZ)+0.0

            for(ii=0; ii<sgX; ii++) for(jj=0; jj<sgY; jj++) for(kk=0; kk<sgZ; kk++){

                        substack1[ii][jj][kk]-=sum1/sssg
                        substack2[ii][jj][kk]-=sum2/sssg


            }



            double var1=0
            double var2=0

            for(ii=0; ii<sgX; ii++) for(jj=0; jj<sgY; jj++) for(kk=0; kk<sgZ; kk++){

                        var1+=substack1[ii][jj][kk]*substack1[ii][jj][kk]
                        var2+=substack2[ii][jj][kk]*substack2[ii][jj][kk]


            }

            var1=sqrt(var1)
            var2=sqrt(var2)

            for(ii=0; ii<sgX; ii++) for(jj=0; jj<sgY; jj++) for(kk=0; kk<sgZ; kk++){

                substack1[ii][jj][kk]/=var1
                substack2[ii][jj][kk]/=var2

            }

            double S=0.0

            for(ii=0; ii<sgX; ii++) for(jj=0; jj<sgY; jj++) for(kk=0; kk<sgZ; kk++){

                S+=substack1[ii][jj][kk]*substack2[ii][jj][kk]

            }

            return S

        }else{

            return -1.0

        }

    }else{

        return -1.0

    }

}

"""


def norm(x):
    return np.sum(x**2)


def crosscorrelateSections(substackr: np.ndarray, substacka: np.ndarray) -> float:
    return np.sum(substackr * substacka)


def getSubstack(stack1, r, sgX, sgY, sgZ):
    sX, sY, sZ = stack1.shape

    substack = np.zeors([sgX, sgY, sgZ])

    # int ii,jj,kk

    sum = 0

    # int iii,jjj,kkk

    # std::cout<<xf<<","<<yf<<","<<zf<<","<<dxf<<","<<dyf<<","<<dzf<<" | "<<sX<<","<<sY<<","<<sZ<<" \n"

    xf, yf, zf = r.astype(int)

    fdi = r[0] - xf
    fdj = r[1] - yf
    fdk = r[2] - zf

    if (sgX / 2) < xf < (sX - sgX / 2) and (sgY / 2) < yf < (sY - sgY / 2) and (sgZ / 2) < zf < (sZ - sgZ / 2):

        # std::cout<<xf<<","<<yf<<","<<zf<<", | "<<sX<<","<<sY<<","<<sZ<<" \n"

        x0 = xf - sgX//2
        x1 = x0 - sgX
        y0 = yf - sgY//2
        y1 = y0 - sgY
        z0 = zf - sgZ//2
        z1 = z0 - sgZ

        substack = (
                (1 - fdi) * (1 - fdj) * (1 - fdk) * stack1[x0:x1, y0:y1, z0:z1] +
                (fdi) * (1 - fdj) * (1 - fdk) * stack1[x0+1:x1+1, y0:y1, z0:z1] +
                (1 - fdi) * (fdj) * (1 - fdk) * stack1[x0:x1, y0+1:y1, z0:z1] +
                (1 - fdi) * (1 - fdj) * (fdk) * stack1[x0:x1, y0:y1, z0+1:z1] +
                (fdi) * (fdj) * (1 - fdk) * stack1[x0+1:x1+1, y0+1:y1+1, z0:z1] +
                (1 - fdi) * (fdj) * (fdk) * stack1[x0:x1, y0+1:y1+1, z0+1:z1+1] +
                (fdi) * (1 - fdj) * (fdk) * stack1[x0+1:x1+1, y0:y1, z0+1:z1+1] +
                (fdi) * (fdj) * (fdk) * stack1[x0+1:x1+1, y0+1:y1+1, z0+1:z1+1:]

        )

        substack -= np.mean(substack)

        substack /= np.linalg.norm(substack)
        return substack

    else:
        substack[:] = 0

        return substack


def crosscorrelateSections(stack1, stack2, r, du=None):
    if du is None:
        substacka = getSubstack(stack2, r)
        return crosscorrelateSections(stack1, substacka)
    else:
        substackr = getSubstack(stack1, r)
        substacka = getSubstack(stack2, r + du)
        return crosscorrelateSections(substackr, substacka)


def findLocalDisplacement(substackr: np.ndarray, stacka: np.ndarray, R: np.ndarray, Ustart: np.ndarray, Srec: np.ndarray, lambd: float, subpixel: float) -> np.ndarray:
    # double lambda=0.01

    # subpixel=double(CFG["SUBPIXEL"])

    hinit = 4.0

    P = np.random.uniform(-hinit, hinit, size=4)
    S = np.random.uniform(-hinit, hinit, size=4)

    S = [crosscorrelateSections(substackr, stacka, R + P[i] + Ustart) - lambd * norm(P[i]) for i in range(4)]

    mini = np.argmin(S)
    maxi = np.argmax(S)

    done = False

    for ii in range(1000):
        if done:
            break

        mini = np.argmin(S)
        maxi = np.argmax(S)

        # std::cout<<"| new cycle mini = "<<mini<<std::endl

        # reflect

        P_plane = np.zeros(3)
        for i in range(4):
            if i != mini:
                P_plane += P[i]
        P_plane = P_plane * 0.333333333

        P_ref = P[mini] + (P_plane - P[mini]) * 2.0

        # B.Shift=P_ref; S_ref=B.testDeformation(stackr,stacka,M)
        S_ref = crosscorrelateSections(substackr, stacka, R + P_ref + Ustart) - lambd * norm(P_ref)

        if S_ref > S[maxi]:

            # expand

            # std::cout<<"| expanding "<<std::endl

            P_exp = P[mini] + (P_plane - P[mini]) * 3.0

            # B.Shift=P_exp; S_exp=B.testDeformation(stackr,stacka,M)
            S_exp = crosscorrelateSections(substackr, stacka, R + P_exp + Ustart) - lambd * norm(P_exp)

            if S_exp > S_ref:

                # std::cout<<"| took expanded "<<std::endl

                P[mini] = P_exp
                S[mini] = S_exp

            else:

                # std::cout<<"| took reflected (expanded was worse)"<<std::endl

                P[mini] = P_ref
                S[mini] = S_ref


        else:

            bsw = False
            for i in range(4):
                if i != mini:
                    if S_ref > S[i]:
                        bsw = True

            if bsw:

                # std::cout<<"| took reflected (better than second worst)"<<std::endl

                P[mini] = P_ref
                S[mini] = S_ref

            else:

                if S_ref > S[maxi]:

                    # std::cout<<"| took reflected (not better than second worst)"<<std::endl

                    P[mini] = P_ref
                    S[mini] = S_ref

                else:

                    P_con = P[mini] + (P_plane - P[mini]) * 0.5
                    # B.Shift=P_con; S_con=B.testDeformation(stackr,stacka,M)
                    S_con = crosscorrelateSections(substackr, stacka, R + P_con + Ustart) - lambd * norm(P_con)

                    if S_con > S[mini]:

                        # std::cout<<"| took contracted"<<std::endl

                        P[mini] = P_con
                        S[mini] = S_con

                    else:

                        # std::cout<<"| contracting myself"<<std::endl

                        for i in range(4):
                            if i != maxi:
                                P[i] = (P[maxi] + P[i]) * 0.5
                                # B.Shift=P[i]; S[i]=B.testDeformation(stackr,stacka,M)
                                S[i] = crosscorrelateSections(substackr, stacka, R + P[i] + Ustart) - lambd * norm(P[i])

        # P[0].print2()
        # P[1].print2()
        # P[2].print2()
        # P[3].print2()

        # std::cout<<"0: "<<S[0]<<" ; "<<P[0][0]<<" , "<<P[0][1]<<" , "<<P[0][2]<<std::endl
        # std::cout<<"1: "<<S[1]<<" ; "<<P[1][0]<<" , "<<P[1][1]<<" , "<<P[1][2]<<std::endl
        # std::cout<<"2: "<<S[2]<<" ; "<<P[2][0]<<" , "<<P[2][1]<<" , "<<P[2][2]<<std::endl
        # std::cout<<"3: "<<S[3]<<" ; "<<P[3][0]<<" , "<<P[3][1]<<" , "<<P[3][2]<<std::endl

        # mS=(S[0]+S[1]+S[2]+S[3])

        # std::cout<<" S_ref = "<<S_ref<<std::endl

        mx = (P[0][0] + P[1][0] + P[2][0] + P[3][0]) * 0.25
        my = (P[0][1] + P[1][1] + P[2][1] + P[3][1]) * 0.25
        mz = (P[0][2] + P[1][2] + P[2][2] + P[3][2]) * 0.25

        stdx = np.sqrt(
            (P[0][0] - mx) * (P[0][0] - mx) + (P[1][0] - mx) * (P[1][0] - mx) + (P[2][0] - mx) * (P[2][0] - mx) + (
                        P[3][0] - mx) * (P[3][0] - mx)) * 0.25
        stdy = np.sqrt(
            (P[0][1] - my) * (P[0][1] - my) + (P[1][1] - my) * (P[1][1] - my) + (P[2][1] - my) * (P[2][1] - my) + (
                        P[3][1] - my) * (P[3][1] - my)) * 0.25
        stdz = np.sqrt(
            (P[0][2] - mz) * (P[0][2] - mz) + (P[1][2] - mz) * (P[1][2] - mz) + (P[2][2] - mz) * (P[2][2] - mz) + (
                        P[3][2] - mz) * (P[3][2] - mz)) * 0.25

        # std::cout<<"stdx = "<<stdx<<" ; stdy = "<<stdy<<" ; stdz = "<<stdz<<std::endl

        if stdx < subpixel and stdy < subpixel and stdz < subpixel:
            done = True

    mini = np.argmin(S)
    maxi = np.argmax(S)

    # B.Shift=P[maxi]

    #  B.Shift.print2()

    # std:cout<<S[maxi]<<std::endl

    Srec.record(S[maxi])

    return P[maxi] + Ustart


def crosscorrelateStacks_old(stack1: np.ndarray, stack2: np.ndarray, du: np.ndarray, jump=-1) -> float:
    # std::cout<<"check 01 \n"

    sX, sY, sZ = stack1.shape

    jumpx = jump
    jumpy = jump
    jumpz = jump

    if jump < 0:
        jumpx = sX / 24
        jumpy = sY / 24
        jumpz = sZ / 24

    sgX = (sX - jumpx) / jumpx - 1
    sgY = (sY - jumpy) / jumpy - 1
    sgZ = (sZ - jumpz) / jumpz - 1

    dif, djf, dkf = du.astype(int)

    fdi = du[0] - dif
    fdj = du[1] - djf
    fdk = du[2] - dkf

    # std::cout<<"check 02 \n"<<sgX<<" "<<sgY<<" "<<sgZ<<"\n"
    substack1 = np.zeros((sgX, sgY, sgZ))
    substack2 = np.zeros((sgX, sgY, sgZ))

    sum1 = 0
    sum2 = 0
    # std::cout<<"check 03 \n"

    for ii in range(sgX):
        for jj in range(sgY):
            for kk in range(sgZ):
                i1 = jumpx * (ii + 1)
                i2 = jumpx * (ii + 1) + dif
                j1 = jumpy * (jj + 1)
                j2 = jumpy * (jj + 1) + djf
                k1 = jumpz * (kk + 1)
                k2 = jumpz * (kk + 1) + dkf

                substack1[ii][jj][kk] = (stack1[i1][j1][k1])

                substack2[ii][jj][kk] = (
                        (1 - fdi) * (1 - fdj) * (1 - fdk) * stack2[(i2 + sX) % sX][(j2 + sY) % sY][(k2 + sZ) % sZ] +
                        (fdi) * (1 - fdj) * (1 - fdk) * stack2[(i2 + 1 + sX) % sX][(j2 + sY) % sY][(k2 + sZ) % sZ] +
                        (1 - fdi) * (fdj) * (1 - fdk) * stack2[(i2 + sX) % sX][(j2 + 1 + sY) % sY][(k2 + sZ) % sZ] +
                        (1 - fdi) * (1 - fdj) * (fdk) * stack2[(i2 + sX) % sX][(j2 + sY) % sY][(k2 + 1 + sZ) % sZ] +
                        (fdi) * (fdj) * (1 - fdk) * stack2[(i2 + 1 + sX) % sX][(j2 + 1 + sY) % sY][(k2 + sZ) % sZ] +
                        (1 - fdi) * (fdj) * (fdk) * stack2[(i2 + sX) % sX][(j2 + 1 + sY) % sY][(k2 + 1 + sZ) % sZ] +
                        (fdi) * (1 - fdj) * (fdk) * stack2[(i2 + 1 + sX) % sX][(j2 + sY) % sY][(k2 + 1 + sZ) % sZ] +
                        (fdi) * (fdj) * (fdk) * stack2[(i2 + 1 + sX) % sX][(j2 + 1 + sY) % sY][(k2 + 1 + sZ) % sZ]
                )

                sum1 += substack1[ii][jj][kk]
                sum2 += substack2[ii][jj][kk]

    sssg = (sgX * sgY * sgZ) + 0.0

    substack1 -= sum1 / sssg
    substack2 -= sum1 / sssg
    #for ii in range(sgX):
    #    for jj in range(sgY):
    #        for kk in range(sgZ):
    #            substack1[ii][jj][kk] -= sum1 / sssg
    #            substack2[ii][jj][kk] -= sum2 / sssg

    var1 = np.linalg.norm(substack1)
    var2 = np.linalg.norm(substack2)

    #var1 = 0
    #var2 = 0
    #for ii in range(sgX):
    #    for jj in range(sgY):
    #        for kk in range(sgZ):
    #            var1 += substack1[ii][jj][kk] * substack1[ii][jj][kk]
    #            var2 += substack2[ii][jj][kk] * substack2[ii][jj][kk]
    #
    #var1 = np.sqrt(var1)
    #var2 = np.sqrt(var2)

    substack1 /= var1
    substack2 /= var2

    #for ii in range(sgX):
    #    for jj in range(sgY):
    #        for kk in range(sgZ):
    #            substack1[ii][jj][kk] /= var1
    #            substack2[ii][jj][kk] /= var2

    S = np.sum(substack1 * substack2)
    #S = 0.0
    #
    #for ii in range(sgX):
    #    for jj in range(sgY):
    #        for kk in range(sgZ):
    #            S += substack1[ii][jj][kk] * substack2[ii][jj][kk]

    return S


def crosscorrelateStacks(stack1: np.ndarray, stack2: np.ndarray, du: np.ndarray, jump: int = -1) -> float:

    if jump < 0:
        jumpx, jumpy, jumpz = np.array(stack1.shape) // 24
    else:
        jumpx = jumpy = jumpz = jump

    substack1 = stack1[::jumpx, ::jumpy, ::jumpz]
    substack2 = getShiftedInterpolatedStack(stack2, du, [jumpx, jumpy, jumpz])

    # subtract the mean
    substack1 -= np.mean(substack1)
    substack2 -= np.mean(substack2)

    # normalize
    substack1 /= np.linalg.norm(substack1)
    substack2 /= np.linalg.norm(substack2)

    # return the correlation
    return np.sum(substack1 * substack2)


def getShiftedInterpolatedStack(stack2: np.ndarray, du: Sequence, jump: Sequence) -> np.ndarray:
    du_int = du.astype(int)

    fdi, fdj, fdk = du - du_int
    jumpx, jumpy, jumpz = jump

    stack2 = np.roll(stack2, du_int)
    substack2 = (
            (1 - fdi) * (1 - fdj) * (1 - fdk) * stack2[::jumpx, ::jumpy, ::jumpz] +
            (fdi) * (1 - fdj) * (1 - fdk) * stack2[1::jumpx, ::jumpy, ::jumpz] +
            (1 - fdi) * (fdj) * (1 - fdk) * stack2[::jumpx, 1::jumpy, ::jumpz] +
            (1 - fdi) * (1 - fdj) * (fdk) * stack2[::jumpx, ::jumpy, 1::jumpz] +
            (fdi) * (fdj) * (1 - fdk) * stack2[1::jumpx, 1::jumpy, ::jumpz] +
            (1 - fdi) * (fdj) * (fdk) * stack2[::jumpx, 1::jumpy, 1::jumpz] +
            (fdi) * (1 - fdj) * (fdk) * stack2[1::jumpx, ::jumpy, 1::jumpz] +
            (fdi) * (fdj) * (fdk) * stack2[1::jumpx, 1::jumpy, 1::jumpz]

    )
    return substack2

"""
void blur(stack3D& stack, stack3D& stack2 , int kernelsize){

    std::vector< std::vector< std::vector< double > > > kernel

    kernel.assign(kernelsize,
        std::vector< std::vector< double > >(kernelsize,
            std::vector< double >(kernelsize, 0.0 )
        )
    )

    int i,j,k

    double dx,dy,dz,Isum=0.0

    for(i=0; i<kernelsize; i++) for(j=0; j<kernelsize; j++) for(k=0; k<kernelsize; k++){

        dx=floor(i-0.5*kernelsize)/kernelsize
        dy=floor(j-0.5*kernelsize)/kernelsize
        dz=floor(k-0.5*kernelsize)/kernelsize

        kernel[i][j][k]=exp(-dx*dx-dy*dy-dz*dz)
        Isum+=kernel[i][j][k]

    }

    for(i=0; i<kernelsize; i++) for(j=0; j<kernelsize; j++) for(k=0; k<kernelsize; k++) kernel[i][j][k]/=Isum


    int sX=stack.size()
    int sY=stack[0].size()
    int sZ=stack[0][0].size()

    stack2.assign(sX,
        std::vector< std::vector< unsigned char > >(sY,
            std::vector< unsigned char >(sZ, 0 )
        )
    )

    int i1,j1,k1,i2,j2,k2,ii,jj,kk,iii,jjj,kkk

    double I=0.0

    for(i=0; i<sX; i++){

        std::cout<<"bluring "<<floor( (i+0.0)/(sX+0.0)*10000 )/100<<"% done         \r"

        for(j=0; j<sY; j++) for(k=0; k<sZ; k++){

            i1=max(double(0),floor(i-0.5*kernelsize))
            i2=min(double(sX-1),ceil(i+0.5*kernelsize)-1)

            j1=max(double(0),floor(j-0.5*kernelsize))
            j2=min(double(sY-1),ceil(j+0.5*kernelsize)-1)

            k1=max(double(0),floor(k-0.5*kernelsize))
            k2=min(double(sZ-1),ceil(k+0.5*kernelsize)-1)

            I=0.0

            for(ii=i1; ii<i2; ii++) for(jj=j1; jj<j2; jj++) for(kk=k1; kk<k2; kk++){


                iii=ii-i+floor(0.5*kernelsize)+1
                jjj=jj-j+floor(0.5*kernelsize)+1
                kkk=kk-k+floor(0.5*kernelsize)+1

                # std::cout<<i<<","<<j<<","<<k<<" "<<ii<<","<<jj<<","<<kk<<" "<<iii<<","<<jjj<<","<<kkk<<"\n"

                I+=stack[ii][jj][kk]*kernel[iii][jjj][kkk]

            }

            stack2[i][j][k]=floor(I)

        }
    }
}

stack2D slice(stack3D& stack, double theta, double phi, int thickness){


    int sizeX=stack.size()
    int sizeY=stack[0].size()
    int sizeZ=stack[0][0].size()


    stack2D img=emptyStack2D(sizeX,sizeY)

    int x,y,z

    mat3D M=rotMatZ(phi)*rotMatX(theta)

    # M.print()

    vec3D middle=vec3D(sizeX*0.5,sizeY*0.5,sizeZ*0.5)

    vec3D r

    double I

    # int xf,yf,zf
    # double fx,fy,fz

    double zz

    # int i


    for(z=0;z<thickness;z++){

    # std::cout<<z<<"\r"

        for(x=0;x<sizeX;x++){


            for(y=0;y<sizeY;y++){


                zz=-thickness*0.5+z+0.5+sizeZ*0.5

                r=vec3D(x,y,zz)-middle

                r=M*r

                r=r+middle

                # r.print2()

                int xf=floor(r[0])
                int yf=floor(r[1])
                int zf=floor(r[2])

                double fx=r[0]-xf
                double fy=r[1]-yf
                double fz=r[2]-zf

                # std::cout<<x<<" "<<y<<" "<<zz<<" | "<<xf<<" "<<yf<<" "<<zf<<"\r"

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
                    )


                }else I=0.0
                # i=int(img[x][y])



                # r.print2()
                # std::cout<<i<<" "<<I<<" "<<max(i,int(floor(I+0.5)))<<"\n"

                if(floor(I+0.5)>img[x][y]) img[x][y]=floor(I+0.5)



            }
        }
    }

    return img

}

"""


def imageFromStack(stack, k: int):
    return stack[:, :, k]


def mean(stack):
    return np.mean(stack)


def correlateImagesFromStacks(stack1: np.ndarray, stack2: np.ndarray, k1: int, k2: int, dx: float, dy: float, mean1: float, mean2: float) -> float:
    return np.sum((stack1[:, :, k1] - mean1) * np.roll(stack2[:, :, k2] - mean2, (dx, dy)))


def allignStacks(stackr: np.ndarray, stackao: np.ndarray, idx: float = 0, idy: float = 0, idz: float = 0) -> np.ndarray:
    sX, sY, sZ = stackr.shape

    safety = sZ / 16

    stacka = np.zeros((sX, sY, sZ), dtype=np.uint8)

    meanr = np.mean(stackr)
    meanao = np.mean(stackao)

    r0 = 3
    r = 3
    ddx = 3
    ddy = 3

    dx_opt = 0
    dy_opt = 0
    z2_opt = 0

    # iterate over all z slices
    for z in range(sZ):
        if z < (safety + r0):
            r = r0 + safety
            z2_opt = z + idz
            dx_opt = idx
            dy_opt = idy
        else:
            r = r0

        print("alligning stacks z=", z, " dzopt=", z2_opt - z, " dxopt=", dx_opt, " dyopt=", dy_opt, "  \r", end="")

        # stack2D imr=imageFromStack(stackr,z)
        ima = np.zeros((sX, sY), dtype=float)

        zoptold = int(np.floor(z2_opt + 0.5))
        dxoptold = int(np.floor(dx_opt + 0.5))
        dyptold = int(np.floor(dy_opt + 0.5))

        S = []
        d_S = []

        sumS = 0.0
        minS = 1.0e20
        i = 0

        # try different offsets
        for z2 in range(zoptold - r, zoptold + r + 1):
            for dx in range(dxoptold - ddx, dxoptold + ddx + 1):
                for dy in range(dyptold - ddy, dyptold + ddy + 1):
                    # keep the z shift between zero and the maximum
                    zz2 = z2
                    if zz2 < 0:
                        zz2 = 0
                    if zz2 > sZ - 1:
                        zz2 = sZ - 1
                    # correlate the images at the provided offsets
                    i += 1
                    Stemp = correlateImagesFromStacks(stackr, stackao, z, zz2, dx, dy, meanr, meanao) / (2621440.0)
                    # store the correlation and the offset
                    S.append(Stemp)
                    d_S.append([dx, dy, z2])

        # find the best offset
        maxi = np.argmax(S)
        dx_opt, dy_opt, z2_opt = d_S[maxi]

        # iterate over all tested offsets to find the minimum correlation
        for ii in range(i):
            if abs(d_S[ii][0] - dx_opt) < 2 and abs(d_S[ii][1] - dy_opt) < 2 and abs(d_S[ii][2] - z2_opt) < 2:
                if S[ii] < minS:
                    minS = S[ii]

        # iterate again over all tested offsets and sum over all valid z slices
        for ii in range(i):
            if abs(d_S[ii][0] - dx_opt) < 2 and abs(d_S[ii][1] - dy_opt) < 2 and abs(d_S[ii][2] - z2_opt) < 2:
                factor = S[ii] - minS
                sumS += factor
                ima += factor * np.roll(stackao[:, :, d_S[ii][2]], (d_S[ii][0] + sX, d_S[ii][1] + sY))

        # assign the weighted averages to the stack
        stacka[:, :, z] = np.floor(ima / sumS + 0.5)

    return stacka


def saveStack(stack, fnamebase: str, fnameending: str = ".bmp"):
    # int sX=stack.size()
    # int sY=stack[0].size()
    sZ = stack.shape[2]

    # char buf[5]

    for z in range(sZ):
        im = imageFromStack(stack, z)

        # std::cout<<"in saveStack sizeX="<<im.size()<<" sizeY="<<im[0].size()<<"\n"

        qs = "%05d" % z

        # sprintf(buf,"%05d",z)
        fnamez = fnamebase + "_z" + qs + fnameending
        plt.imwrite(im, fnamez)


def renderFilename(fnamebase: str, z: int) -> str:
    return fnamebase % z


def readStackWildcard(fstr: str, jump: int = 1):
    # std::cout<<"fstr = "<<fstr<<"\n\n"

    fullstring = fstr

    # fullstring = fullstring.replace("#", "*")

    import os
    from pathlib import Path

    dirstring, filename = os.path.split(fullstring)

    dir = Path(dirstring)

    if not dir.exists():
        print("ERROR: Couldn't find directory", dirstring.toStdString())

    entryList = list(Path(dirstring).glob(filename))

    if len(entryList) == 0 and dir.exists():
        print("ERROR: Couldn't find any files matching", filename, "in directory", dirstring)

    image = plt.imread(dirstring + entryList[0])

    sX = image.shape[1]
    sY = image.shape[0]
    sZ = len(entryList) / jump

    print("sizes wil be", sX, sY, sZ)

    stack = np.zeros((sY, sX, sZ), dtype=np.uint8)

    failed = np.zeros(sZ, dtype=bool)

    for z in range(sZ):
        print("loading stack -", z, "\r", end="")

        try:
            image = plt.imread(dirstring + entryList[z * jump])
        except IOError:
            failed[z] = True

        if not failed[z]:
            stack[:, :, z] = image
        else:
            print("\n\nWARNING: The image ", (dirstring + entryList[z * jump]),
                  "could not be loaded. It will be interpolated from the neighbors.")

    #  first image loading failed
    if failed[0]:
        stack[:, :, 0] = stack[:, :, 1]

    #  last image loading failed
    if failed[-1]:
        stack[:, :, -1] = stack[:, :, -2]

    #  all other fails
    for z in range(1, sZ - 1):
        if failed[z]:
            stack[:, :, z] = (stack[:, :, z - 1] + stack[:, :, z + 1]) / 2

    return stack


def readStackSprintf(fnamebase: str, zfrom: int, zto: int, jump: int = 1):
    # cimg::exception_mode(0)

    # CImg<unsigned int> image(renderFilename(fnamebase, zfrom).c_str())

    image = plt.imread(renderFilename(fnamebase, zfrom))

    # image.display()

    # int sC=image.spectrum()

    sX = image.shape[1]
    sY = image.shape[0]
    sZ = (zto - zfrom) / jump + 1

    print("sizes wil be", sX, sY, sZ)

    stack = np.zeros((sX, sY, sZ), dtype=np.uint8)
    failed = np.zeros(sZ, dtype=bool)

    for z in range(sZ):
        print("loading stack -", z, "\r", end="")

        try:
            image = plt.imread(renderFilename(fnamebase, zfrom + z * jump))
        except IOError:
            failed[z] = True

        if not failed[z]:
            stack[:, :, z] = image
        else:
            print("\n\nWARNING: The image ", renderFilename(fnamebase, zfrom + z * jump),
                  "could not be loaded. It will be interpolated from the neighbors.")

    #  first image loading failed
    if failed[0]:
        stack[:, :, 0] = stack[:, :, 1]

    #  last image loading failed
    if failed[-1]:
        stack[:, :, -1] = stack[:, :, -2]

    #  all other fails
    for z in range(1, sZ - 1):
        if failed[z]:
            stack[:, :, z] = (stack[:, :, z - 1] + stack[:, :, z + 1]) / 2

    return stack
