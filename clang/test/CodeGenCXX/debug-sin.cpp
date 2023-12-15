// Ensure that declarations obtained by a using declaration are generated
// sin() is declared through cmath with: using ::sin;

// RUN: %clang -emit-llvm -S -g %s -o %t.ll
// RUN: FileCheck %s < %t.ll

// CHECK: DISubprogram(name: "sin",

#include <math.h>

int main() {
  float f;

  f=sin(1.32);
  return 0;
}
