#include <stdio.h>
#include <math.h>


int main()
{   int i;
    int posicao[8];
    int states=2;
    int multi[8];
    int regra=30;
    int inputs[8];
    int outputs[8];
    int saida[8];
    int binary[8];
    int dno;
    for (i = 0; i <8; i++)
    {
    posicao[i] = i;
    }

    for (i = 0; i <8; i++)
    {
    multi[i] = pow(i,states);
    }

    for (i = 0; i <8; i++)
    {
    outputs[i]=regra/multi[i];
    saida[i]=outputs[i]%2;
    return 0;
    }



    for (i = 0; i <8; i++)
    {
    long bno=0,remainder,f=1;
    dno=posicao[i];
    while(dno != 0)
    {
         remainder = dno % 2;
         bno = bno + remainder * f;
         f = f * 10;
         dno = dno / 2;
         binary[i]=bno;
    }
    }
    


}
