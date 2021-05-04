// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown-sycldevice -disable-llvm-passes -emit-llvm -o - -sycl-std=2017 -DSYCL2017 %s
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown-sycldevice -disable-llvm-passes -emit-llvm -o - -sycl-std=2020 -DSYCL2020 %s

#include "sycl.hpp"

using namespace cl::sycl;
queue q;

class Functor32x16x16 {
public:
  [[cl::reqd_work_group_size(32, 16, 16)]] void operator()() const {}
};

[[cl::reqd_work_group_size(8, 1, 1)]] void f8x1x1() {}

class Functor {
public:
  void operator()() const {
    f8x1x1();
  }
};

template <int SIZE, int SIZE1, int SIZE2>
class FunctorTemp {
public:
  [[cl::reqd_work_group_size(SIZE, SIZE1, SIZE2)]] void operator()() const {}
};

template <int N, int N1, int N2>
[[cl::reqd_work_group_size(N, N1, N2)]] void func() {}

[[cl::reqd_work_group_size(10, 10, 10)]] void func1() {}

int main() {
  q.submit([&](handler &h) {
    Functor32x16x16 f32x16x16;
    h.single_task<class kernel_name1>(f32x16x16);

    Functor f;
    h.single_task<class kernel_name2>(f);

    h.single_task<class kernel_name3>(
        []() [[cl::reqd_work_group_size(8, 8, 8)]]{});

    // Test class template argument.
    FunctorTemp<2, 2, 2> ft;
    h.single_task<class kernel_name4>(ft);

#if defined(SYCL2017)
    // Test template argument with propagated function attribute.
    h.single_task<class kernel_name5>([]() {
      func<8, 4, 4>();
    });

    // Test attribute is propagated.
    h.single_task<class kernel_name6>(
        []() { func1(); });
#endif // SYCL2017

    // Test attribute is applied on lambda.
    h.single_task<class kernel_name7>(
        []() [[cl::reqd_work_group_size(1, 8, 2)]]{});

#if defined(SYCL2020)
    // Test attribute is not propagated.
    h.single_task<class kernel_name8>(
        []() { func1(); });
#endif // SYCL2020

  });
  return 0;
}

// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name1"() #0 {{.*}} !reqd_work_group_size ![[WGSIZE32:[0-9]+]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name2"() #0 {{.*}} !reqd_work_group_size ![[WGSIZE8:[0-9]+]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name3"() #0 {{.*}} !reqd_work_group_size ![[WGSIZE88:[0-9]+]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name4"() #0 {{.*}} !reqd_work_group_size ![[WGSIZE22:[0-9]+]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name5"() #0 {{.*}} !reqd_work_group_size ![[WGSIZE44:[0-9]+]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name6"() #0 {{.*}} !reqd_work_group_size ![[WGSIZE10:[0-9]+]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name7"() #0 {{.*}} !reqd_work_group_size ![[WGSIZE2:[0-9]+]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name8"() #0 {{.*}} ![[NUM0:[0-9]+]]
// CHECK: ![[WGSIZE32]] = !{i32 16, i32 16, i32 32}
// CHECK: ![[WGSIZE8]] = !{i32 1, i32 1, i32 8}
// CHECK: ![[WGSIZE88]] = !{i32 8, i32 8, i32 8}
// CHECK: ![[WGSIZE22]] = !{i32 2, i32 2, i32 2}
// CHECK: ![[WGSIZE44]] = !{i32 4, i32 4, i32 8}
// CHECK: ![[WGSIZE10]] = !{i32 10, i32 10, i32 10}
// CHECK: ![[WGSIZE2]] = !{i32 2, i32 8, i32 1}
// CHECK: ![[NUM0]] = !{}
