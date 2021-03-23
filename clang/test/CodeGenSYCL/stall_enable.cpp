// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown-sycldevice -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s

#include "sycl.hpp"

using namespace cl::sycl;
queue q;

class Foo {
public:
  [[intel::use_stall_enable_clusters]] void operator()() const {}
};

int main() {
  q.submit([&](handler &h) {
    Foo f;
    h.single_task<class test_kernel1>(f);

    h.single_task<class test_kernel2>(
        []() [[intel::use_stall_enable_clusters]]{});
  });
  return 0;
}

// CHECK: define {{.*}}spir_kernel void @"{{.*}}test_kernel1"() #0 {{.*}} !stall_enable ![[NUM5:[0-9]+]]
// CHECK: define {{.*}}spir_kernel void @"{{.*}}test_kernel2"() #0 {{.*}} !stall_enable ![[NUM5]]
// CHECK: ![[NUM5]] = !{i32 1}
