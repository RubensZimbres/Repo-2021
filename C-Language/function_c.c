# include <stdio.h>

int myGlobal=0;

int multiply(int x, int y);
int AbsValue(int a);

int main(void)
{   int result;
    result=multiply(-3,-3);
    printf("\n%i\n",result);
    
    return 0;
}

int multiply(int x, int y)
{   int result;
    int myLocal;
    result = AbsValue(x)*AbsValue(y);
    return result;
}

int AbsValue(int a)
{
    if (a < 0)
    a=-a;
    return a;
}
