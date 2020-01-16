// RUN: %clang_cc1 -triple spir64-unknown-unknown-sycldevice -disable-llvm-passes -fsycl-is-device -emit-llvm -o - %s | FileCheck %s

class Foo {
public:
  [[intelfpga::uses_global_work_offset(0)]] void operator()() {}
};

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(Func kernelFunc) {
  kernelFunc();
}

void bar() {
  Foo boo;
  kernel<class kernel_name1>(boo);

  kernel<class kernel_name2>(
      []() [[intelfpga::uses_global_work_offset(0)]]{});

  kernel<class kernel_name3>(
      []() [[intelfpga::uses_global_work_offset(1)]]{});
}

// CHECK: define spir_kernel void @{{.*}}kernel_name1() {{.*}} !uses_global_work_offset ![[NUM5:[0-9]+]]
// CHECK: define spir_kernel void @{{.*}}kernel_name2() {{.*}} !uses_global_work_offset ![[NUM5]]
// CHECK: define spir_kernel void @{{.*}}kernel_name3() {{.*}} ![[NUM4:[0-9]+]]
// CHECK-NOT: ![[NUM4]]  = !{i32 1}
// CHECK: ![[NUM5]] = !{i32 0}
