// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple amdgcn-amd-amdhsa -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s

// The product of the reqd_work_group_size dimensions below is 2^32, which
// overflows a 32-bit unsigned integer. Check that the resulting
// amdgpu-flat-work-group-size function attribute reflects the full 64-bit
// product instead of a wrapped-around 32-bit value.

#include "Inputs/sycl.hpp"

using namespace sycl;
queue q;

class FunctorOverflow {
public:
  [[sycl::reqd_work_group_size(65536, 65536)]] void operator()() const {}
};

int main() {
  q.submit([&](handler &h) {
    FunctorOverflow f;
    h.single_task<class kernel_overflow>(f);
  });
  return 0;
}

// CHECK: define {{.*}} void @{{.*}}kernel_overflow() #[[ATTR:[0-9]+]]
// CHECK: attributes #[[ATTR]] = { {{.*}}"amdgpu-flat-work-group-size"="4294967296,4294967296"{{.*}} }
