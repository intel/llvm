// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown-sycldevice -disable-llvm-passes -emit-llvm -o - -sycl-std=2017 -DSYCL2017 %s
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown-sycldevice -disable-llvm-passes -emit-llvm -o - -sycl-std=2020 -DSYCL2020 %s

#include "sycl.hpp"

using namespace cl::sycl;
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

template <int N>
[[intel::num_simd_work_items(N)]] void func() {}

[[intel::num_simd_work_items(10)]] void func1() {}

int main() {
  q.submit([&](handler &h) {
    Foo boo;
    h.single_task<class kernel_name1>(boo);

    h.single_task<class kernel_name2>(
        []() [[intel::num_simd_work_items(42)]]{});

    Functor<2> f;
    h.single_task<class kernel_name3>(f);

#if defined(SYCL2017)
    // Test template argument with propagated function attribute.
    h.single_task<class kernel_name4>([]() {
      func<4>();
    });

    // Test attribute is propagated.
    h.single_task<class kernel_name5>(
        []() { func1(); });
#endif // SYCL2017

#if defined(SYCL2020)
    // Test attribute is not propagated.
    h.single_task<class kernel_name6>(
        []() { func1(); });
#endif // SYCL2020
  });
  return 0;
}

// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name1"() #0 {{.*}} !num_simd_work_items ![[NUM1:[0-9]+]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name2"() #0 {{.*}} !num_simd_work_items ![[NUM42:[0-9]+]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name3"() #0 {{.*}} !num_simd_work_items ![[NUM2:[0-9]+]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name4"() #0 {{.*}} !num_simd_work_items ![[NUM4:[0-9]+]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name5"() #0 {{.*}} !num_simd_work_items ![[NUM10:[0-9]+]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name6"() #0 {{.*}} ![[NUM0:[0-9]+]]
// CHECK: ![[NUM1]] = !{i32 1}
// CHECK: ![[NUM42]] = !{i32 42}
// CHECK: ![[NUM2]] = !{i32 2}
// CHECK: ![[NUM4]] = !{i32 4}
// CHECK: ![[NUM10]] = !{i32 10}
// CHECK: ![[NUM0]] = !{}
