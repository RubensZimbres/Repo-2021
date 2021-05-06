#include <stdio.h>
#include <stdlib.h>
#include "StringFunctions.h"


int main(){
    char temp[] = "rubens";
    char temp2[]="apple";
    char temp3[100];
    printf("Number of 'p' in apples is %d\n",numberOfCharactersInString(temp2,'p'));
    removeNonAlphaCharacters(temp);
    return 0;
}