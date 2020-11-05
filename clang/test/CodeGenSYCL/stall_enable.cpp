// RUN: %clang_cc1 -fsycl -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown-sycldevice -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s

#include "sycl.hpp"

using namespace cl::sycl;
queue q;

[[intel::stall_enable]] void test() {}

class Foo {
public:
  [[intel::stall_enable]] void operator()() const {}
};

int main() {
  q.submit([&](handler &h) {
    Foo boo;
    h.single_task<class test_kernel1>(boo);

    h.single_task<class test_kernel2>(
        []() [[intel::stall_enable]]{});
  });
  return 0;
}

// CHECK: define spir_kernel void @"{{.*}}test_kernel1"() #0 {{.*}} !stall_enable [[FOO:![0-9]+]]
// CHECK: define spir_kernel void @"{{.*}}test_kernel2"() #0 {{.*}} !stall_enable [[FOO:![0-9]+]]
// CHECK: [[FOO]] = !{i32 1}
