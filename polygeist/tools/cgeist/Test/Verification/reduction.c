// RUN: cgeist %s --function=reduction_gemm | FileCheck %s
// XFAIL: *
void reduction_gemm() {
  int i, j, k;
  int A[1024][1024];
  int B[1024][1024];
  int C[1024][1024];

#pragma scop
  for (i = 0; i < 1024; i++)
    for (j = 0; j < 1024; j++)
      for (k = 0; k < 1024; k++)
        C[i][j] += A[i][k] * B[k][j];
#pragma endscop
}

// RUN: cgeist %s --function=reduction_bicg | FileCheck %s
// XFAIL: *
void reduction_bicg() {
  int i, j;
  int A[100][200];
  int r[100];
  int s[200];
  int p[200];
  int q[100];

#pragma scop
  for (i = 0; i < 100; i++) {
    for (j = 0; j < 200; j++) {
      s[j] = s[j] + r[i] * A[i][j];
    }
  }
#pragma endscop
}

// RUN: cgeist %s --function=reduction_sum | FileCheck %s
// XFAIL: *
void reduction_sum() {
  int sum = 0;
  int A[100];
#pragma scop
  for (int i = 0; i < 100; i++)
    sum += A[i];
#pragma endscop
}

