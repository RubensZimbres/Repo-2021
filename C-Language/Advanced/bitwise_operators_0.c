#include <stdio.h>

int main(){
    short int w1=25;
    /// 0000000000011001
    short int w2=77;
    /// 0000000001001101
    short int w3=0;
    w3=w1 & w2;
    /// 128 64 32 16 8 4 2 1
    /// 0000000000001001 = 9
    printf("%d",w3);
    return 0;
}
