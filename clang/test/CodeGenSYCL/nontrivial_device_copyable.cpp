// RUN: %clang_cc1 -triple spir64 -fsycl-is-device -internal-isystem %S/Inputs -disable-llvm-passes -sycl-std=2020 -emit-llvm %s -o - | FileCheck %s

// Tests that SYCL kernel arguments with non-trivially copyable types are
// passed by-valued.

#include "Inputs/sycl.hpp"
using namespace sycl;

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

// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name(ptr noundef byval(%struct.NontriviallyCopyable)
// CHECK-NOT: define {{.*}}spir_func void @{{.*}}device_func{{.*}}({{.*}}byval(%struct.NontriviallyCopyable)
// CHECK: define {{.*}}spir_func void @_Z11device_func20NontriviallyCopyable(ptr noundef %X)
// CHECK: %X.indirect_addr = alloca ptr addrspace(4)
// CHECK: %X.indirect_addr.ascast = addrspacecast ptr %X.indirect_addr to ptr addrspace(4)
// CHECK: %X.ascast = addrspacecast ptr %X to ptr addrspace(4)
// CHECK: store ptr addrspace(4) %X.ascast, ptr addrspace(4) %X.indirect_addr.ascast
