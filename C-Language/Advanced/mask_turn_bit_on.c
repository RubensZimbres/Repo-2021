#include <stdio.h>

int main(){
    int flags=15;
    /// 0000 1111
    int mask=182;
    /// 1011 0110

    //// turn on
    flags = flags | mask;
    /// 1011 1111

    /// turn off
    flags = flags & ~mask;

    printf("\nMask applied = %d\n",flags);
    return 0;
}
