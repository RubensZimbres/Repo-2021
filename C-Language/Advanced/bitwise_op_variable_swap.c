#include <stdio.h>

int main(){
    short int w1=4;
    /// 00000100
    short int w2=2;
    /// 00000010
    printf("\nbefore w1 =%d and w2 = %d\n",w1,w2);
    
    //short int temp=0;
    //temp=w1;
    //w1=w2;
    //w2=temp;

    // goes to

    w1^=w2;
    w2^=w1;
    w1^=w2;
    printf("\nafter w1 =%d and w2 = %d\n",w1,w2);
    return 0;
}
