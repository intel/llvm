// Check that DebugLoc attached to a builtin call is preserved after translation.

// RUN: %clang_cc1 -triple spir -fdeclare-opencl-builtins -finclude-default-header %s -disable-llvm-passes -emit-llvm-bc -debug-info-kind=line-tables-only -o %t.bc
// Added -opaque-pointers.
// FIXME: Align with the community code when project is ready to enable opaque
// pointers by default
// RUN: llvm-spirv -opaque-pointers %t.bc -spirv-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
// Added -opaque-pointers.
// FIXME: Align with the community code when project is ready to enable opaque
// pointers by default
// RUN: llvm-spirv -opaque-pointers %t.bc -o %t.spv
// Added -opaque-pointers.
// FIXME: Align with the community code when project is ready to enable opaque
// pointers by default
// Added -opaque-pointers.
// FIXME: Align with the community code when project is ready to enable opaque
// pointers by default
// RUN: llvm-spirv -opaque-pointers -r -emit-opaque-pointers %t.spv -o - | llvm-dis -opaque-pointers -o - | FileCheck %s --check-prefix=CHECK-LLVM

// CHECK-SPIRV: Label
// CHECK-SPIRV: ExtInst {{.*}} DebugScope
// CHECK-SPIRV: ExtInst {{.*}} sin
// CHECK-LLVM: call spir_func float @_Z3sinf(float %{{.*}}) {{.*}} !dbg ![[loc:[0-9]+]]
// FIXME: Due to shift in lines, DILocation line moved 14 -> 27
// CHECK-LLVM: ![[loc]] = !DILocation(line: 27, column: 10, scope: !{{.*}})
float f(float x) {
  return sin(x);
}
