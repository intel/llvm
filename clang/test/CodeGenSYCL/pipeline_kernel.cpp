// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown -disable-llvm-passes -sycl-std=2020 -emit-llvm -o - %s | FileCheck %s

#include "sycl.hpp"

using namespace sycl;
queue q;

class Foo {
public:
  [[intel::disable_loop_pipelining]] void operator()() const {}
};

[[intel::disable_loop_pipelining]] void foo() {}

int main() {
  q.submit([&](handler &h) {
    // Test attribute is presented on function metadata.
    Foo f;
    h.single_task<class test_kernel1>(f);

    // Test attribute is not propagated.
    h.single_task<class test_kernel2>(
        []() { foo(); });

    // Test attribute is applied on lambda.
    h.single_task<class test_kernel3>(
        []() [[intel::disable_loop_pipelining]]{});
  });
  return 0;
}

// CHECK: define dso_local spir_kernel void @{{.*}}test_kernel1() #0 {{.*}} !pipeline_kernel ![[NUM5:[0-9]+]]
// CHECK: define dso_local spir_kernel void @{{.*}}test_kernel2() #0 {{.*}} ![[NUM4:[0-9]+]]
// CHECK: define dso_local spir_kernel void @{{.*}}test_kernel3() #0 {{.*}} !pipeline_kernel ![[NUM5]]
// CHECK: ![[NUM4]] = !{}
// CHECK: ![[NUM5]] = !{i32 0}
