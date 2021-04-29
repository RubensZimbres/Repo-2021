#include <stdio.h>
#include <math.h>


int main()
{   
    int i;
    int posicao[8];
    int states=2;
    int multi[8];
    int regra=0;
    int *pregra=&regra;
    while (regra<=255){
    int outputs[8];
    int saida[8];
    printf("Which CA rule ?");
    scanf("%d",&regra);
    for (i = 0; i <8; i++)
    {
    posicao[i] = i;
    multi[i] = pow(states,i);
    outputs[i]=*pregra/multi[i];
    saida[i]=outputs[i]%2;
    }


    printf("\n%d%d%d%d%d%d%d%d\n",saida[0],saida[1],saida[2],saida[3],saida[4],saida[5],saida[6],saida[7]);
    }
    return 0;
    
}
