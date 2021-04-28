#include <stdio.h>
#include <stdlib.h>

void square(int  * const x);

int main(){
    int num=9;
    square(&num);
    printf("Square on given is:\n %d",num);
    return 0;
}

void square(int * const x){

    *x=(*x) * (*x); /// resultado usa mesmo espaço na memória

}
