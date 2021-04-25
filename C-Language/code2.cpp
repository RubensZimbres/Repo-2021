#include <stdio.h>
#include <math.h>


int main()
{   fflush( stdout );
    int i;
    int in[3]={0,1,1};
    int arr[3];
    int posicao[8];
    int states=2;
    int multi[8];
    int regra=232;
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
    multi[i] = pow(states,i);
    }

    for (i = 0; i <8; i++)
    {
    outputs[i]=regra/multi[i];
    saida[i]=outputs[i]%2;
    }


    printf("\n%d%d%d%d%d%d%d%d\n",saida[0],saida[1],saida[2],saida[3],saida[4],saida[5],saida[6],saida[7]);
    return 0;
    
}
