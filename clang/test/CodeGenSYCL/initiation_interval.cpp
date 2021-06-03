// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown-sycldevice -disable-llvm-passes -sycl-std=2020 -emit-llvm -o - %s | FileCheck %s

#include "sycl.hpp"

using namespace cl::sycl;
queue q;

class Foo {
public:
  [[intel::initiation_interval(1)]] void operator()() const {}
};

template <int SIZE>
class Functor {
public:
  [[intel::initiation_interval(SIZE)]] void operator()() const {}
};

[[intel::initiation_interval(5)]] void foo() {}

int main() {
  q.submit([&](handler &h) {
    // Test attribute argument size.
    Foo boo;
    h.single_task<class kernel_name1>(boo);

    // Test attribute is applied on lambda.
    h.single_task<class kernel_name2>(
        []() [[intel::initiation_interval(42)]]{});

    // Test template argument.
    Functor<2> f;
    h.single_task<class kernel_name3>(f);

    // Test attribute is not propagated.
    h.single_task<class kernel_name4>(
        []() { foo(); });
  });
  return 0;
}

// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name1() #0 {{.*}} !initiation_interval ![[NUM1:[0-9]+]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name2() #0 {{.*}} !initiation_interval ![[NUM42:[0-9]+]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name3() #0 {{.*}} !initiation_interval ![[NUM2:[0-9]+]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name4() #0 {{.*}} ![[NUM0:[0-9]+]]
// CHECK: ![[NUM0]] = !{}
// CHECK: ![[NUM1]] = !{i32 1}
// CHECK: ![[NUM42]] = !{i32 42}
// CHECK: ![[NUM2]] = !{i32 2}
