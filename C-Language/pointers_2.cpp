#include <stdio.h>
#include <stdlib.h>

void square(int  * x);

int main(){
    ///int num=9;
    int *num =(int*)malloc(sizeof(int)); ///memory allocation
    *num=4;
    square(num);
    printf("Square on given is:\n %d",*num);
    return 0;
}

void square(int * x){

    *x=(*x) * (*x); /// resultado usa mesmo espaço na memória

}
