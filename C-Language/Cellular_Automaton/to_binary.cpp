#include<stdio.h>    
#include<stdlib.h>  
int main()
{  
    int a[3];
    int input1;
    int *dno=&input1;
    int bno=0;
    int remainder;
    int g=0;
    printf("Enter the number to convert: ");    
    scanf("%d",&input1);    

    while(g<3)
    {   
        remainder = *dno % 2;
        *dno = *dno / 2;
        bno = remainder;
        a[g]=bno;
        g=g+1;
    }

    printf("\nBinary of Given Number is:\n%d%d%d\n",a[2],a[1],a[0]);    
    return 0;  
}  
