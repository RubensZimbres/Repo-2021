Install C/C++ complement


################################################################

#include <stdio.h>
#include <string.h>

void changePosition(char *ch1, char *ch2)
{
    char tmp;
    tmp = *ch1;
    *ch1 = *ch2;
    *ch2 = tmp;
}
void charPermu(char *cht, int stno, int endno)
{
   int i;
   if (stno == endno)
     printf("%s  ", cht);
   else
   {
       for (i = stno; i <= endno; i++)
       {
          changePosition((cht+stno), (cht+i));
          charPermu(cht, stno+1, endno);
          changePosition((cht+stno), (cht+i)); 
       }
   }
}
 
int main()
{
    char str[] = "abcd";
   printf("\n\n Pointer : Generate permutations of a given string :\n"); 
   printf("--------------------------------------------------------\n"); 
    int n = strlen(str);
    printf(" The permutations of the string are : \n");
    charPermu(str, 0, n-1);
     printf("\n\n");
    return 0;
}

##########################################

Terminal / Configure Default Bluid Task

Back to .cpp / Run Build Task

./ test
