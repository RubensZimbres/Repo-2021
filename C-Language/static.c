# include <stdio.h>

int fun(){
    static int count=0;
    int localvar=0;
    printf("automatic= %d, static = %d\n",localvar,count);
    count++;
    localvar++;
    return count;
}

int main(){
    for (int i=0; i<5; i++){
        fun();
    }
}
