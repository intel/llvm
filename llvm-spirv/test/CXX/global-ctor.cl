// RUN: %clang_cc1 -cl-std=clc++ -emit-llvm-bc -triple spir -O0 %s -o %t.bc
// Added -opaque-pointers.
// FIXME: Align with the community code when project is ready to enable opaque
// pointers by default
// RUN: llvm-spirv -opaque-pointers %t.bc -o %t.spv
// RUN: spirv-val %t.spv
// Added -opaque-pointers.
// FIXME: Align with the community code when project is ready to enable opaque
// pointers by default
// RUN: llvm-spirv -opaque-pointers %t.spv -to-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
// Added -opaque-pointers.
// FIXME: Align with the community code when project is ready to enable opaque
// pointers by default
// Added -opaque-pointers.
// FIXME: Align with the community code when project is ready to enable opaque
// pointers by default
// RUN: llvm-spirv -opaque-pointers -r -emit-opaque-pointers %t.spv -o - | llvm-dis -opaque-pointers -o - | FileCheck %s --check-prefix=CHECK-LLVM
// Added -opaque-pointers.
// FIXME: Align with the community code when project is ready to enable opaque
// pointers by default
// RUN: not llvm-spirv -opaque-pointers %t.bc --spirv-max-version=1.0 2>&1 | FileCheck %s --check-prefix=CHECK-SPV10

class Something {
  public:
    Something(int a) : v(a) {}
    int v;
};

Something g(33);

void kernel work(global int *out) {
  *out = g.v;
}

// CHECK-SPIRV: EntryPoint 6 [[work:[0-9]+]] "work"
// CHECK-SPIRV-NOT: ExecutionMode [[work]] 33
// CHECK-SPIRV: EntryPoint 6 [[ctor:[0-9]+]] "_GLOBAL__sub_I_global_ctor.cl"
// CHECK-SPIRV: ExecutionMode [[ctor]] 33

// CHECK-LLVM: llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @[[CTORNAME:_GLOBAL__sub_I[^ ]+]], ptr null }
// CHECK-LLVM: define spir_kernel void @[[CTORNAME]]

// CHECK-SPV10: Feature requires SPIR-V 1.1 or greater: Initializer/Finalizer Execution Mode
