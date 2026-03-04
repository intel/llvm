// RUN: %clang_cc1 -triple spir64-unknown-unknown -fsycl-is-device \
// RUN: -emit-llvm %s -o %temp.ll
// RUN: FileCheck -check-prefix=CHECK-SPIR --input-file %temp.ll %s

// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -fsycl-is-device \
// RUN: -emit-llvm %s -o %temp.ll
// RUN: FileCheck -check-prefix=CHECK-AMDGCN --input-file %temp.ll %s

// The test makes sure that amdgcn annotation is correctly generated
// only for their respective targets.

#include "Inputs/sycl.hpp"

sycl::handler H;

class Functor {
public:
  void operator()() const {}
};

// CHECK-SPIR-NOT: annotations =
// CHECK-AMDGCN-NOT: annotations =

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
