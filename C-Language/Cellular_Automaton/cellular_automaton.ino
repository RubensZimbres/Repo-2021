#include <stdio.h>
#include <math.h>
#include<stdlib.h> 

int ca_final[6]={1,0,0,1,0,1};

void setup() 
{
  Serial.begin(9600);
}


void loop(){
    delay(6000);
    for (int p=0;p<6;p++) {
      Serial.print(ca_final[p],DEC);


    }
    
    Serial.println();

    int array[8][3]={{0, 0, 0},
       {0, 0, 0},
       {0, 0, 0},
       {0, 0, 0},
       {0, 0, 0},
       {0, 0, 0},
       {0, 0, 0},
       {0, 0, 0}};

    int input1=0;
    int dno=0;
    int remainder=0;
    int g=0;
    
    for (int j=0;j<8;++j)
    {
    dno=j;
    g=2;
    while(g>=0)
    {   
        remainder = dno % 2;
        dno = dno / 2;
        array[j][g] = remainder;
        g=g-1;
    }
    }

    int m=0;
    m=m+1;
    int k=0;
    int n = 5;
    int split_array[6][3]; 

    split_array[0][0]=ca_final[n];
    split_array[0][1]=ca_final[0];
    split_array[0][2]=ca_final[1];
    split_array[n][0]=ca_final[(int)(n-1)];
    split_array[n][1]=ca_final[n];
    split_array[n][2]=ca_final[0];
    for (k=1;k<n;k++){
        split_array[k][0]=ca_final[(int)(k-1)];
        split_array[k][1]=ca_final[k];
        split_array[k][2]=ca_final[(int)(k+1)];
    }
    //////////////////////////////////////////
    int states=2;
    int regra=30;
    int saida[8]={0,0,0,0,0,0,0,0};

    for (int f = 0; f <8; ++f){
        saida[f]=(int)(regra/(pow(states,f)))%2;
        }
    ///int saida[8]={0,1,1,1,1,0,0,0};
    ///////////////////////////////////////////

    int iter=0;
    int iter2=0;

    for (int ii = 0; ii <6; ii++){
        for (iter2=0;iter2<8;++iter2){
            if (split_array[ii][0]==array[iter2][0] && split_array[ii][1]==array[iter2][1] && split_array[ii][2]==array[iter2][2]){
                ca_final[ii]=saida[iter2];
    
    }
    }
    }
}
