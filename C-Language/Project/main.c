# include <stdio.h>

/// global
int i=5;

void foo (void);

int main (void){
    printf("%i\n",i);
    foo();
    printf("%i\n",i);
    return 0;
}
