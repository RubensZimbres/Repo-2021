#include<stdio.h>    
#include<stdlib.h>  
int main()
{  
    int a[8][3],i,dno,bno=0,remainder,f=1,g=0;
    for (i=0;i<8;i=i+1){
    ///printf("Enter the number to convert: ");    
    ///scanf("%d",&dno);    
    dno=i;
    g=2;
    while(g>=0)
    {   
        remainder = dno % 2;
        dno = dno / 2;
        bno = remainder;
        a[i][g]=bno;
        g=g-1;
    }

    printf("\nBinary of Given Number is:\n%d%d%d\n",a[i][0],a[i][1],a[i][2]);    
    }
    return 0;  
}  
