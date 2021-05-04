#include <stdio.h>

int main(){
    int w1=3;
    /// 00000011
    int w2=0;

    w2=w1<<1;
    /// 00000110
    printf("\nShift left one bit = %d\n",w2);
    return 0;
}
