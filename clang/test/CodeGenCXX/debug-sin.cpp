////////////////////////////////////////////////////////////////////////////////////////

// With default options, ensure that declarations obtained by a using declaration have
// debug info generated.  sin() is declared through cmath with: using ::sin;
//
// Also ensure that no debug info for sin() is generated if -fno-system-debug is used.

// RUN: %clang                   -emit-llvm -S -g %s     -o %t.default.ll
// RUN: %clang -fno-system-debug -emit-llvm -S -g %s     -o %t.no_system_debug.ll

//// FIXME
// We want to run this command on both Windows and Linux
//
//    FileCheck --check-prefix=CHECK-DEFAULT         %s < %t.default.ll
//
// but debug info for "sin" is not generated on Windows.  Use the following
// line to workaround Windows.  If Windows compiler is updated to be the
// same as Linux this check will need to be updated.
//
// RUN: FileCheck %if system-windows %{ --check-prefix=CHECK-NO-SYSTEM-DEBUG %} %else %{ --check-prefix=CHECK-DEFAULT %} %s < %t.default.ll

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
