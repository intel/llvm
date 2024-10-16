// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s

#include "sycl.hpp"

using namespace sycl;
queue q;

class Foo {
public:
  [[intel::scheduler_target_fmax_mhz(5)]] void operator()() const {}
};

template <int N>
class Functor {
public:
  [[intel::scheduler_target_fmax_mhz(N)]] void operator()() const {}
};

int main() {
  q.submit([&](handler &h) {
    // Test attribute argument size.
    Foo boo;
    h.single_task<class kernel_name1>(boo);

    // Test attribute is applied on lambda.
    h.single_task<class kernel_name2>(
        []() [[intel::scheduler_target_fmax_mhz(42)]]{});

    // Test class template argument.
    Functor<7> f;
    h.single_task<class kernel_name3>(f);

  });
  return 0;
}

// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name1() #0 {{.*}} !scheduler_target_fmax_mhz ![[NUM5:[0-9]+]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name2() #0 {{.*}} !scheduler_target_fmax_mhz ![[NUM42:[0-9]+]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name3() #0 {{.*}} !scheduler_target_fmax_mhz ![[NUM7:[0-9]+]]
// CHECK: ![[NUM5]] = !{i32 5}
// CHECK: ![[NUM42]] = !{i32 42}
// CHECK: ![[NUM7]] = !{i32 7}
