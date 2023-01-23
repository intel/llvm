// RUN: cgeist -S -O0 %s | FileCheck %s
// RUN: cgeist -S -O1 %s | FileCheck %s --check-prefix=OPT1

void foo(int A[10]) {
#pragma scop
  for (int i = 0; i < 10; ++i)
    A[i] = A[i] * 2;
#pragma endscop
}

// CHECK-LABEL: func @main()
// CHECK: call @foo
// OPT1-LABEL: func @main()
// OPT1-NOT: call @foo
int main() {
  int A[10];
  foo(A);
  return 0;
}
