# include <stdio.h>

int myGlobal=0;

int multiply(int x, int y)
{   int result;
    int myLocal;
    result = x*y;
    return result;
}

int main(void)
{   int result;
    result=multiply(2,3);
    printf("\n%i\n",result);
    
    return 0;
}
