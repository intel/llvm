// RUN: %clang_cc1 -triple spir64-unknown-unknown-sycldevice -fsycl-is-device -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(Func kernelFunc) {
  kernelFunc();
}

int main() {
  kernel_single_task<class kernel>([]() {});
  return 0;
}
// CHECK: define spir_kernel void @{{.*}}kernel{{.*}}() #[[KERN_ATTR:[0-9]+]]

// CHECK: #[[KERN_ATTR]] = { {{.*}}"sycl-module-id"="{{.*}}module-id.cpp"{{.*}} }
