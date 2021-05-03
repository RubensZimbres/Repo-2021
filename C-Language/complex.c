#include <stdio.h>
#include <complex.h>
#include <math.h>

#define __STDC_WANT_LIBEXT1__ 1

int main(){

    #ifdef __STD_NO_COMPLEX__
        printf('No complex');
        exit(1);
    #else
        printf("supported\n");
    #endif
    double complex cx =1.0 + 3.0*I;
    double complex cy =9.0 - 4.0*I;
    double complex result=cx+cy;

    printf("\n%.2f%+.2fi\n",creal(result),cimag(result));
}   
