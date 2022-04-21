// RUN: %clang_cc1 -cl-std=clc++ -emit-llvm-bc -triple spir -O0 %s -o %t.bc
// RUN: llvm-spirv %t.bc -o %t.spv
// RUN: spirv-val %t.spv
// RUN: llvm-spirv %t.spv -to-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
// RUN: llvm-spirv -r %t.spv -o - | llvm-dis -o - | FileCheck %s --check-prefix=CHECK-LLVM
// RUN: not llvm-spirv %t.bc --spirv-max-version=1.0 2>&1 | FileCheck %s --check-prefix=CHECK-SPV10

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

// CHECK-LLVM: llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @[[CTORNAME:_GLOBAL__sub_I[^ ]+]], i8* null }
// CHECK-LLVM: define spir_kernel void @[[CTORNAME]]

// CHECK-SPV10: Feature requires SPIR-V 1.1 or greater: Initializer/Finalizer Execution Mode
