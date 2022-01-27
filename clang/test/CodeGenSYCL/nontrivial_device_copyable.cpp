// RUN: %clang_cc1 -triple spir64 -fsycl-is-device -internal-isystem %S/Inputs -disable-llvm-passes -sycl-std=2020 -emit-llvm %s -o - | FileCheck %s

// Tests that SYCL kernel arguments with non-trivially copyable types are
// passed by-valued.

#include "Inputs/sycl.hpp"
using namespace cl::sycl;

struct NontriviallyCopyable {
  int I;
  NontriviallyCopyable(int I) : I(I) {}
  NontriviallyCopyable(const NontriviallyCopyable &X) : I(X.I) {}
};

void device_func(NontriviallyCopyable X) {
  (void)X;
}

int main() {
  NontriviallyCopyable NontriviallyCopyableObject{10};

  queue Q;
  Q.submit([&](handler &CGH) {
    CGH.single_task<class kernel_name>([=]() {
      device_func(NontriviallyCopyableObject);
    });
  });
}

// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name(%struct.NontriviallyCopyable* noundef byval(%struct.NontriviallyCopyable)
// CHECK-NOT: define {{.*}}spir_func void @{{.*}}device_func{{.*}}({{.*}}byval(%struct.NontriviallyCopyable)
