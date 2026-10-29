// FPBuiltinFnSelection also runs as part of the codegen pipeline, which builds
// its own TargetLibraryInfo. Check that the alternate math library from
// -faltmathlib= reaches it, so the llvm.fpbuiltin.* calls left over after the
// optimization pipeline still find an implementation instead of failing with
// "no suitable implementation was found".

// REQUIRES: x86-registered-target

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -S -o - \
// RUN:   -ffp-builtin-accuracy=high -faltmathlib=SVMLAltMathLibrary %s \
// RUN:   | FileCheck --check-prefix=CHECK-HIGH %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -S -o - \
// RUN:   -ffp-builtin-accuracy=low -faltmathlib=SVMLAltMathLibrary %s \
// RUN:   | FileCheck --check-prefix=CHECK-LOW %s

// CHECK-HIGH-LABEL: call_sinf:
// CHECK-HIGH: __svml_sinf1_ha
// CHECK-LOW-LABEL: call_sinf:
// CHECK-LOW: __svml_sinf1_ep
float call_sinf(float x) { return __builtin_sinf(x); }

// CHECK-HIGH-LABEL: call_sin:
// CHECK-HIGH: __svml_sin1_ha
// CHECK-LOW-LABEL: call_sin:
// CHECK-LOW: __svml_sin1_ep
double call_sin(double x) { return __builtin_sin(x); }
