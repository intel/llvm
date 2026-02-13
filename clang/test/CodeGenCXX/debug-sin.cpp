////////////////////////////////////////////////////////////////////////////////////////

// With default options, ensure that declarations obtained by a using declaration have
// debug info generated.  sin() is declared through cmath with: using ::sin;
//
// Also ensure that no debug info for sin() is generated if -fno-system-debug is used.

// Debug info for math library functions is not generated on Windows
// UNSUPPORTED: system-windows

// RUN: %clang                   -emit-llvm -S -g %s     -o %t.default.ll
// RUN: %clang -fno-system-debug -emit-llvm -S -g %s     -o %t.no_system_debug.ll

// Check for debug info for "sin" with default option
// RUN: FileCheck --check-prefix=CHECK-DEFAULT         %s < %t.default.ll

// No debug information for "sin" should be generated with -fno-system-debug
// RUN: FileCheck --check-prefix=CHECK-NO-SYSTEM-DEBUG %s < %t.no_system_debug.ll

// CHECK-DEFAULT:             DISubprogram(name: "sin",
// CHECK-NO-SYSTEM-DEBUG-NOT: DISubprogram(name: "sin",

////////////////////////////////////////////////////////////////////////////////////////


#include <math.h>

int main() {
  float f;

  f=sin(1.32);
  return 0;
}
