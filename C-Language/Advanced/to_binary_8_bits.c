#include<stdio.h>    
#include<stdlib.h> 

int main()
{  
    int a[8];
    int input1;
    int *dno=&input1;
    int remainder;
    int g=0;
    printf("Enter the number to convert: ");    
    scanf("%d",&input1);    

    while(g<8)
    {   
        remainder = *dno % 2;
        *dno = *dno / 2;
        printf("%d\n",*dno);
        a[g] = remainder;
        g=g+1;
    }

    printf("\nBinary of Given Number is:\n%d%d%d%d%d%d%d%d\n",a[7],a[6],a[5],a[4],a[3],a[2],a[1],a[0]);    
    return 0;  
} 
