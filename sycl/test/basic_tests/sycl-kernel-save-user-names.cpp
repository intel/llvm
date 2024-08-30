// RUN: %clangxx -fsycl -fsycl-device-only -fno-discard-value-names -fno-sycl-early-optimizations -o %t.bc %s
// RUN: sycl-post-link -properties %t.bc -spec-const=emulation -o %t.table
// RUN: llvm-spirv -o %t.spv -spirv-max-version=1.3 -spirv-ext=+all %t.bc
// RUN: llvm-spirv -o %t.rev.bc -r %t.spv
// RUN: llvm-dis %t.rev.bc -o=- | FileCheck %s

// Test to verify that user specified names are retained in SPIR kernel argument
// names. (It is a copy of clang/test/CodeGenSYCL/save-user-names.cpp with just
// additional compilation steps).

#include <sycl/sycl.hpp>

struct NestedSimple {
  int NestedSimpleField;
};

struct NestedComplex {
  int NestedComplexField;
  sycl::accessor<char, 1, sycl::access::mode::read> NestedAccField;
};

struct KernelFunctor {
  int IntField;
  sycl::accessor<char, 1, sycl::access::mode::read> AccField1;
  sycl::accessor<char, 1, sycl::access::mode::read> AccField2;
  NestedSimple NestedSimpleObj;
  NestedComplex NestedComplexObj;
  void operator()() const {}
};

int main() {
  sycl::queue q;

  q.submit([&](sycl::handler &cgh) {
    KernelFunctor FunctorObj;
    cgh.single_task<class Kernel1>(FunctorObj);
  });

  q.submit([&](sycl::handler &cgh) {
    int Data;
    NestedSimple NestedSimpleObj;
    NestedComplex NestedComplexObj;
    sycl::accessor<char, 1, sycl::access::mode::read> CapturedAcc1,
        CapturedAcc2;
    cgh.single_task<class Kernel2>([=]() {
      Data;
      CapturedAcc1;
      CapturedAcc2;
      NestedSimpleObj;
      NestedComplexObj;
    });
  });

  return 0;
}

// Check kernel parameters generated when kernel is defined as Functor

// NOTE: Accessor fields have 4 corresponding openCL kernel arguments. When
// the compiler generates the openCL kernel arguments, they are generated
// with the same name (i.e the user-specified name for accessor). Since LLVM
// IR cannot have same names, a number is appended in IR.
//
// CHECK: define {{.*}}spir_kernel void @{{.*}}Kernel1
// CHECK-SAME: %_arg_IntField
// CHECK-SAME: %_arg_AccField1{{.*}}%_arg_AccField11{{.*}}%_arg_AccField12{{.*}}%_arg_AccField13
// CHECK-SAME: %_arg_AccField2{{.*}}%_arg_AccField24{{.*}}%_arg_AccField25{{.*}}%_arg_AccField26
// CHECK-SAME: %_arg_NestedSimpleObj
// CHECK-SAME: %_arg_NestedComplexField
// CHECK-SAME: %_arg_NestedAccField{{.*}}%_arg_NestedAccField7{{.*}}%_arg_NestedAccField8{{.*}}%_arg_NestedAccField9

// Check kernel parameters generated when kernel is defined as Lambda
//
// CHECK: define {{.*}}spir_kernel void @{{.*}}Kernel2
// CHECK-SAME: %_arg_Data
// CHECK-SAME: %_arg_CapturedAcc1{{.*}}%_arg_CapturedAcc11{{.*}}%_arg_CapturedAcc12{{.*}}%_arg_CapturedAcc13
// CHECK-SAME: %_arg_CapturedAcc2{{.*}}%_arg_CapturedAcc24{{.*}}%_arg_CapturedAcc25{{.*}}%_arg_CapturedAcc26
// CHECK-SAME: %_arg_NestedSimpleObj
// CHECK-SAME: %_arg_NestedComplexField
// CHECK-SAME: %_arg_NestedAccField{{.*}}%_arg_NestedAccField7{{.*}}%_arg_NestedAccField8{{.*}}%_arg_NestedAccField9
