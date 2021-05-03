#include <stdio.h>
int x = 15;
void first(void);
void second(void);
 
int main(void)  {
  extern int x;
 
  printf("x in main() is %d\n", x);
  first();
  printf("x in main() is %d\n", x);
  second();
  printf("x in main() is %d\n", x);
  return 0;
}
 
 
void first(void) {
  int x;
  x = 25;
  printf("x in first() is %d\n", x);
}
 
void second(void) {
  x = 35;
  printf("x in second() is %d\n", x);
}
