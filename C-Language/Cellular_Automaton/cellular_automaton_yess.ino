#include <stdio.h>
#include <math.h>
#include<stdlib.h> 

int ca_final[6]={1,0,0,1,0,1};

void setup() 
{
  Serial.begin(9600);
}


void loop(){
    delay(3000);
    for (int p=0;p<6;p++) {
      Serial.print(ca_final[p],DEC);


    }
    
    Serial.println();

    int array[8][3]={{0, 0, 0},
       {0, 0, 1},
       {0, 1, 0},
       {0, 1, 1},
       {1, 0, 0},
       {1, 0, 1},
       {1, 1, 0},
       {1, 1, 1}};
    
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
    int i=0;
    int posicao[8];
    int states=2;
    int multi[8];
    int regra=30;
    
    int outputs[8];
    int saida[8]={0,1,1,1,1,0,0,0};

    
    int iter=0;
    int iter2=0;

    for (int ii = 0; ii <6; ii++){
        for (iter2=0;iter2<6;++iter2){
            if (split_array[ii][0]==array[iter2][0] && split_array[ii][1]==array[iter2][1] && split_array[ii][2]==array[iter2][2]){
                ca_final[ii]=saida[iter2];
    
    }
    
    }
    }
}
