#include <stdio.h>

int sumOfRange(int);

int main() {
   int n1 = 0;
   int sum = 0;

   scanf("%d", &n1);

   sum = sumOfRange(n1);

   printf("\n The sum 1 to %d : %d\n\n", n1, sum);

   return (0);
}

int sumOfRange(int n1) {
   int result = 0;

   if (n1 == 1) {
      return 1;
   }
   else {
     result = n1 + sumOfRange(n1 - 1);
   }

   return result;
}
