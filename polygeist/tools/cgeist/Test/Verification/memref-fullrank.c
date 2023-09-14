// RUN: cgeist %s -S -memref-fullrank -O0 | FileCheck %s

#include <stdio.h>

int f(int A[10][20]) {
  int i, j, sum = 0;
#pragma scop
  for (i = 0; i < 10; i++)
    for (j = 0; j < 20; j++)
      sum += A[i][j];
#pragma endscop
  return sum;
}

int g(int A[10][20]) {
  int c = f(A);
  printf("%d\n", c);

  return 0;
}

int main() {
  int A[10][20];
  return g(A);
}

// CHECK: func @main()
// CHECK: %[[VAL0:.*]] = memref.alloca() : memref<10x20xi32>
// CHECK: %{{.*}} = call @g(%[[VAL0]]) : (memref<10x20xi32>) -> i32

// CHECK: func @g(%arg0: memref<10x20xi32>) -> i32

// CHECK: func @f(%arg0: memref<10x20xi32>) -> i32
