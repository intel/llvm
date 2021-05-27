// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown-sycldevice -disable-llvm-passes -emit-llvm -o - -sycl-std=2017 -DSYCL2017 %s
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown-sycldevice -disable-llvm-passes -emit-llvm -o - -sycl-std=2020 -DSYCL2020 %s

#include "sycl.hpp"

using namespace cl::sycl;
queue q;

class Functor16 {
public:
  [[intel::reqd_sub_group_size(16)]] void operator()() const {}
};

[[intel::reqd_sub_group_size(8)]] void foo() {}

class Functor8 {
public:
  void operator()() const {
    foo();
  }
};

template <int SIZE>
class Functor2 {
public:
  [[intel::reqd_sub_group_size(SIZE)]] void operator()() const {}
};

template <int N>
[[intel::reqd_sub_group_size(N)]] void func() {}

[[intel::reqd_sub_group_size(10)]] void func1() {}

int main() {
  q.submit([&](handler &h) {
    Functor16 f16;
    h.single_task<class kernel_name1>(f16);

    Functor8 f8;
    h.single_task<class kernel_name2>(f8);

    h.single_task<class kernel_name3>(
        []() [[intel::reqd_sub_group_size(4)]]{});

#if defined(SYCL2020)
    // Test attribute is not propagated.
    h.single_task<class kernel_name4>(
        []() { func1(); });
#endif // SYCL2020

    // Test class template argument.
    Functor2<2> f2;
    h.single_task<class kernel_name5>(f2);

#if defined(SYCL2017)
    // Test template argument with propagated function attribute.
    h.single_task<class kernel_name6>([]() {
      func<2>();
    });

    // Test attribute is propagated.
    h.single_task<class kernel_name7>(
        []() { func1(); });
#endif // SYCL2017
  });
  return 0;
}

// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name1"() #0 {{.*}} !intel_reqd_sub_group_size ![[SGSIZE16:[0-9]+]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name2"() #0 {{.*}} !intel_reqd_sub_group_size ![[SGSIZE8:[0-9]+]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name3"() #0 {{.*}} !intel_reqd_sub_group_size ![[SGSIZE4:[0-9]+]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name4"() #0 {{.*}} ![[SGSIZE0:[0-9]+]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name5"() #0 {{.*}} !intel_reqd_sub_group_size ![[SGSIZE2:[0-9]+]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name6"() #0 {{.*}} !intel_reqd_sub_group_size ![[SGSIZE2]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name7"() #0 {{.*}} !intel_reqd_sub_group_size ![[SGSIZE10:[0-9]+]]
// CHECK: ![[SGSIZE16]] = !{i32 16}
// CHECK: ![[SGSIZE8]] = !{i32 8}
// CHECK: ![[SGSIZE4]] = !{i32 4}
// CHECK: ![[SGSIZE0]] = !{}
// CHECK: ![[SGSIZE2]] = !{i32 2}
// CHECK: ![[SGSIZE10]] = !{i32 10}
