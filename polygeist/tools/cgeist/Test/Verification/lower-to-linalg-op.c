// RUN: cgeist %s -S | FileCheck %s

#pragma lower_to(copy_op, "memref.copy") "input"(a), "output"(b)
void copy_op(int b[3][3], int a[3][3]) {
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      b[i][j] = a[i][j];
}

int main() {
  int a[3][3];
  int b[3][3];
  // CHECK: memref.copy {{.*}}, {{.*}} : memref<3x3xi32> to memref<3x3xi32>
  copy_op(a, b);
  return 0;
}
