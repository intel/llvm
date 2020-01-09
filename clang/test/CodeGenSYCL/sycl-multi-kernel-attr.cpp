// RUN: %clang_cc1 -triple spir64-unknown-unknown-sycldevice -disable-llvm-passes -fsycl-is-device -emit-llvm -o - %s | FileCheck %s

class Functor {
public:
  [[cl::intel_reqd_sub_group_size(4), cl::reqd_work_group_size(32, 16, 16)]] void operator()() {}
};


template <typename Name, typename Func>
__attribute__((sycl_kernel)) void kernel(Func kernelFunc) {
  kernelFunc();
}

void bar() {
  Functor foo;
  kernel<class kernel_name>(foo);
}

// CHECK: define spir_kernel void @{{.*}}kernel_name() {{.*}} !reqd_work_group_size ![[WGSIZE:[0-9]+]] !intel_reqd_sub_group_size ![[SGSIZE:[0-9]+]]
// CHECK: ![[WGSIZE]] = !{i32 32, i32 16, i32 16}
// CHECK: ![[SGSIZE]] = !{i32 4}
