#include <stdio.h>

int main(){
    short int w1=147;
    /// 10010011
    short int w2=61;
    /// 00111101
    short int temp=0;
    temp=w1;
    w1=w2;
    w2=temp;

    // goes to

    w1^=w2;
    w2^=w1;
    w1^=w2;

    return 0;
}
