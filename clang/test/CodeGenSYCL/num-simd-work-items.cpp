// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s

#include "sycl.hpp"

using namespace sycl;
queue q;

class Foo {
public:
  [[intel::num_simd_work_items(1)]] void operator()() const {}
};

template <int SIZE>
class Functor {
public:
  [[intel::num_simd_work_items(SIZE)]] void operator()() const {}
};


int main() {
  q.submit([&](handler &h) {
    Foo boo;
    h.single_task<class kernel_name1>(boo);

    h.single_task<class kernel_name2>(
        []() [[intel::num_simd_work_items(42)]]{});

    Functor<2> f;
    h.single_task<class kernel_name3>(f);

  });
  return 0;
}

// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name1() #0 {{.*}} !num_simd_work_items ![[NUM1:[0-9]+]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name2() #0 {{.*}} !num_simd_work_items ![[NUM42:[0-9]+]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name3() #0 {{.*}} !num_simd_work_items ![[NUM2:[0-9]+]]
// CHECK: ![[NUM1]] = !{i32 1}
// CHECK: ![[NUM42]] = !{i32 42}
// CHECK: ![[NUM2]] = !{i32 2}
