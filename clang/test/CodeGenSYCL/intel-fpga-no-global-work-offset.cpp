// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown-sycldevice -disable-llvm-passes -emit-llvm -o - -sycl-std=2017 -DSYCL2017 %s
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown-sycldevice -disable-llvm-passes -emit-llvm -o - -sycl-std=2020 -DSYCL2020 %s

#include "sycl.hpp"

using namespace cl::sycl;
queue q;

class Foo {
public:
  [[intel::no_global_work_offset(1)]] void operator()() const {}
};

template <int SIZE>
class Functor {
public:
  [[intel::no_global_work_offset(SIZE)]] void operator()() const {}
};

template <int N>
[[intel::no_global_work_offset(N)]] void func() {}

[[intel::no_global_work_offset(1)]] void func1() {}

int main() {
  q.submit([&](handler &h) {
    Foo boo;
    h.single_task<class kernel_name1>(boo);

    h.single_task<class kernel_name2>(
        []() [[intel::no_global_work_offset]]{});

    h.single_task<class kernel_name3>(
        []() [[intel::no_global_work_offset(0)]]{});

    // Test class template argument.
    Functor<1> f;
    h.single_task<class kernel_name4>(f);

#if defined(SYCL2017)
    // Test template argument with propagated function attribute.
    h.single_task<class kernel_name5>([]() {
      func<1>();
    });

    // Test attribute is propagated.
    h.single_task<class kernel_name6>(
        []() { func1(); });
#endif // SYCL2017

#if defined(SYCL2020)
    // Test attribute is not propagated.
    h.single_task<class kernel_name7>(
        []() { func1(); });
#endif //SYCL2020
  });
  return 0;
}

// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name1"() #0 {{.*}} !no_global_work_offset ![[NUM5:[0-9]+]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name2"() #0 {{.*}} !no_global_work_offset ![[NUM5]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name3"() #0 {{.*}} ![[NUM4:[0-9]+]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name4"() #0 {{.*}} !no_global_work_offset ![[NUM5]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name5"() #0 {{.*}} !no_global_work_offset ![[NUM5]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name6"() #0 {{.*}} !no_global_work_offset ![[NUM5]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name7"() #0 {{.*}} ![[NUM5]]
// CHECK-NOT: ![[NUM4]]  = !{i32 0}
// CHECK: ![[NUM5]] = !{}
