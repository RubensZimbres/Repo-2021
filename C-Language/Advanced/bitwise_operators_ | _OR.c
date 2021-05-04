#include <stdio.h>

int main(){
    short int w1=147;
    /// 10010011
    short int w2=61;
    /// 00111101
    short int w3=0;
    w3=w1 | w2;
    /// 128 64 32 16 8 4 2 1
    /// 10111111
    printf("%d",w3);
    return 0;
}
