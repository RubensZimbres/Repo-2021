#include <stdio.h>
#include <setjmp.h>
#include <stdlib.h>

jmp_buf buf;

int main(){
    int i = setjmp(buf);
    ///start loop

    if (i!=0) exit(0); /// breaks loop

    longjmp(buf,42); /// back to start

    printf("achieved");
    return 0;
}
