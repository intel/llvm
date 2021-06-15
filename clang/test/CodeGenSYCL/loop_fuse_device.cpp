// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown-sycldevice -emit-llvm -o - %s | FileCheck %s

#include "sycl.hpp"

using namespace cl::sycl;
queue q;

[[intel::loop_fuse(5)]] void foo() {}

template <int SIZE>
class KernelFunctor5 {
public:
  [[intel::loop_fuse(SIZE)]] void operator()() const {}
};

void bar() {

  q.submit([&](handler &h) {
    // Test template argument.
    KernelFunctor5<5> f5;
    h.single_task<class kernel_name_1>(f5);

    // Test different argument sizes.
    // Emit 1 if there is no argument.
    h.single_task<class kernel_name_2>(
        []() [[intel::loop_fuse]]{});
    h.single_task<class kernel_name_3>(
        []() [[intel::loop_fuse(0)]]{});
    h.single_task<class kernel_name_4>(
        []() [[intel::loop_fuse(1)]]{});
    h.single_task<class kernel_name_5>(
        []() [[intel::loop_fuse(10)]]{});

    // Test attribute is not propagated.
    h.single_task<class kernel_name_6>(
        []() { foo(); });
  });
}

// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name_1() {{.*}} !loop_fuse ![[LF5:[0-9]+]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name_2() {{.*}} !loop_fuse ![[LF1:[0-9]+]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name_3() {{.*}} !loop_fuse ![[LF0:[0-9]+]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name_4() {{.*}} !loop_fuse ![[LF1]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name_5() {{.*}} !loop_fuse ![[LF10:[0-9]+]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name_6()
// CHECK-NOT: !loop_fuse
// CHECK-SAME: {
// CHECK: define {{.*}}spir_func void @{{.*}}foo{{.*}} !loop_fuse ![[LF5]]
// CHECK: ![[LF5]] = !{i32 5, i32 0}
// CHECK: ![[LF1]] = !{i32 1, i32 0}
// CHECK: ![[LF0]] = !{i32 0, i32 0}
// CHECK: ![[LF10]] = !{i32 10, i32 0}
