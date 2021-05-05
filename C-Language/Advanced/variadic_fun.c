#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

int addingNumbers(int nHowMany, ...);

int main(void) {

  printf( "\n\n Variadic functions: \n\n" );

  printf( "\n 10 + 20 = %d ", addingNumbers( 2, 10, 20 )  );
  printf( "\n 10 + 20 + 30 = %d ", addingNumbers( 3, 10, 20, 30 )  );
  printf( "\n 10 + 20 + 30 + 40 = %d ", addingNumbers( 4, 10, 20, 30, 40 )  );

  printf( "\n\n" );

  return 1;
}

int addingNumbers(int nHowMany, ...) {
   int nSum = 0;

   va_list intArgumentPointer;
   va_start(intArgumentPointer, nHowMany); /// last variable

    for( int i = 0; i < nHowMany; i++ ) {
        nSum += va_arg(intArgumentPointer, int);
    }

    va_end(intArgumentPointer);

    return nSum;
}
