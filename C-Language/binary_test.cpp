#include<stdio.h>    
#include<stdlib.h>  
int main()
{  
    int a[3],dno,bno=0,remainder,f=1,g=0;
    printf("Enter the number to convert: ");    
    scanf("%d",&dno);    

    while(g<3)
    {
        remainder = dno % 2;
        bno = bno + remainder * f;
        a[g]=bno;
        f = f * 10;
        dno = dno / 2;
        g=g+1;
    }

    printf("\nBinary of Given Number is=%d%d%d",a[0],a[1],a[2]);    
    return 0;  
}  
