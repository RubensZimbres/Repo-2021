#include <stdio.h>

int main(){
    signed int w1=2;
    /// 00000100
    signed int w2=0;

    w2=~(w1);

    printf("\nw2 complement after filpping = %d\n",w2);
    return 0;
}
