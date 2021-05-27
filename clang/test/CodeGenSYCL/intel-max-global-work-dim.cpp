// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown-sycldevice -disable-llvm-passes -emit-llvm -o - -sycl-std=2017 -DSYCL2017 %s
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown-sycldevice -disable-llvm-passes -emit-llvm -o - -sycl-std=2020 -DSYCL2020 %s

#include "sycl.hpp"

using namespace cl::sycl;
queue q;

class Foo {
public:
  [[intel::max_global_work_dim(1)]] void operator()() const {}
};

template <int SIZE>
class Functor {
public:
  [[intel::max_global_work_dim(SIZE)]] void operator()() const {}
};

template <int N>
[[intel::max_global_work_dim(N)]] void func() {}

[[intel::max_global_work_dim(2)]] void func1() {}

int main() {
  q.submit([&](handler &h) {
    Foo boo;
    h.single_task<class kernel_name1>(boo);

    h.single_task<class kernel_name2>(
        []() [[intel::max_global_work_dim(2)]]{});

    // Test class template argument.
    Functor<2> f;
    h.single_task<class kernel_name3>(f);

#if defined(SYCL2017)
    // Test template argument with propagated function attribute.
    h.single_task<class kernel_name4>([]() {
      func<2>();
    });

    // Test attribute is propagated.
    h.single_task<class kernel_name5>(
        []() { func1(); });
#endif // SYCL2017

#if defined(SYCL2020)
    // Test attribute is not propagated.
    h.single_task<class kernel_name6>(
        []() { func1(); });
#endif //SYCL2020
  });
  return 0;
}

// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name1"() #0 {{.*}} !max_global_work_dim ![[NUM1:[0-9]+]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name2"() #0 {{.*}} !max_global_work_dim ![[NUM2:[0-9]+]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name3"() #0 {{.*}} !max_global_work_dim ![[NUM2]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name4"() #0 {{.*}} !max_global_work_dim ![[NUM2]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name5"() #0 {{.*}} !max_global_work_dim ![[NUM2]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name6"() #0 {{.*}} ![[NUM0:[0-9]+]]
// CHECK: ![[NUM1]] = !{i32 1}
// CHECK: ![[NUM2]] = !{i32 2}
// CHECK: ![[NUM0]] = !{}
