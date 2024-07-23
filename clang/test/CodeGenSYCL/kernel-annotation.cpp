// RUN: %clang_cc1 -triple spir64-unknown-unknown -fsycl-is-device \
// RUN: -emit-llvm %s -o %temp.ll
// RUN: FileCheck -check-prefix=CHECK-SPIR --input-file %temp.ll %s

// RUN: %clang_cc1 -triple nvptx64-unknown-unknown -fsycl-is-device \
// RUN: -emit-llvm %s -o %temp.ll
// RUN: FileCheck -check-prefix=CHECK-NVPTX --input-file %temp.ll %s

// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -fsycl-is-device \
// RUN: -emit-llvm %s -o %temp.ll
// RUN: FileCheck -check-prefix=CHECK-AMDGCN --input-file %temp.ll %s

// The test makes sure that `[nnvm|amdgcn].annotations are correctly generated
// only for their respective targets.

#include "Inputs/sycl.hpp"

sycl::handler H;

class Functor {
public:
  void operator()() const {}
};

// CHECK-SPIR-NOT: annotations =
// CHECK-AMDGCN-NOT: annotations =

// CHECK-NVPTX: nvvm.annotations = !{[[FIRST:![0-9]]], [[SECOND:![0-9]]]}
// CHECK-NVPTX: [[FIRST]] = !{ptr @_ZTS7Functor, !"kernel", i32 1}
// CHECK-NVPTX: [[SECOND]] = !{ptr @_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E5foo_2, !"kernel", i32 1}

int main() {
  sycl::queue q;
  q.submit([&](sycl::handler &cgh) {
    Functor foo{};
    cgh.single_task(foo);
  });

  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<class foo_2>(sycl::range<1>(1),
                                  [=](sycl::item<1> item) {
                                  });
  });
  return 0;
}
