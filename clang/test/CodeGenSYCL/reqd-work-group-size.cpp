// RUN: %clang_cc1 -std=c++11 -triple spir64-unknown-linux-sycldevice -disable-llvm-passes -fsycl-is-device -emit-llvm -o - %s | FileCheck %s

class Functor32x16x16 {
public:
  [[cl::reqd_work_group_size(32, 16, 16)]] void operator()() {}
};

[[cl::reqd_work_group_size(8, 1, 1)]] void f8x1x1() {}

class Functor {
public:
  void operator()() {
    f8x1x1();
  }
};

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(Func kernelFunc) {
  kernelFunc();
}

void bar() {
  Functor32x16x16 f32x16x16;
  kernel<class kernel_name1>(f32x16x16);

  Functor f;
  kernel<class kernel_name2>(f);
}

// CHECK: define spir_kernel void @{{.*}}kernel_name1() {{.*}} !reqd_work_group_size ![[WGSIZE32:[0-9]+]]
// CHECK: define spir_kernel void @{{.*}}kernel_name2() {{.*}} !reqd_work_group_size ![[WGSIZE8:[0-9]+]]
// CHECK: ![[WGSIZE32]] = !{i32 32, i32 16, i32 16}
// CHECK: ![[WGSIZE8]] = !{i32 8, i32 1, i32 1}

