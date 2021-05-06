#include <stdio.h>
#include "StringFunctions.h"

/*
str – string to search
searchCharacter – character to look for
return type - int : count for the number of times that character was found
*/
int numberOfCharactersInString(char *str, char searchCharacter)
{
   int i = 0, frequency = 0;

   for(i = 0; str[i] != '\0'; ++i)
   {
       if(searchCharacter == str[i])
           ++frequency;
   }

   return frequency;
}

/*
source - source string
return type - int : 0 on success
*/
int removeNonAlphaCharacters(char *source)
{
    int i = 0, j = 0;

    for(i = 0; source[i] != '\0'; ++i)
    {
        while (!( (source[i] >= 'a' && source[i] <= 'z') || (source[i] >= 'A' && source[i] <= 'Z') || source[i] == '\0') )
        {
            for(j = i; source[j] != '\0'; ++j)
            {
                source[j] = source[j+1];
            }
            source[j] = '\0';
        }
    }

    return 0;
}

/*
source - source string
return type - int : length of string
*/
int lengthOfString(char *source)
{
    int length = 0;

    for(length = 0; source[length] != '\0'; ++length);

    return length;
}

/*
str1 – string to concatenate to (resulting string)
str2 – second string to concatenate from
return type - int : 0 on success
*/
int strConcat(char *str1, char *str2)
{
    int i = 0, j = '\0';

    // calculate the length of string str1 and store it in i
    for(i = 0; str1[i] != '\0'; ++i);

    for(j = 0; str2[j] != '\0'; ++j, ++i)
    {
        str1[i] = str2[j];
    }
    str1[i] = '\0';
    return 0;
}


/*
source – string to copy from
destination – second string to copy to
return type - int : 0 on success
*/
int strCopy(char *source, char *destination)
{
    int i = 0;

    for(i = 0; source[i] != '\0'; ++i)
    {
        destination[i] = source[i];
    }

    destination[i] = '\0';

    return 0;
}

/*
source - source string
from - starting index from where you want to get substring
n - number of characters to be copied in substring
target - target string in which you want to store targe string
return type - int : 0 on success
*/
int substring(char *source, int from, int n, char *target)
{
    int length = 0,i = 0;

    //get string length
    for(length=0;source[length]!='\0';length++);

    if(from>length){
        printf("Starting index is invalid.\n");
        return 1;
    }

    if((from+n)>length){
        //get substring till end
        n=(length-from);
    }

    //get substring in target
    for(i=0;i<n;i++){
        target[i]=source[from+i];
    }
    target[i]='\0'; //assign null at last

    return 0;
}

