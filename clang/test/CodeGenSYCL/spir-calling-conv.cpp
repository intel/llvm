// RUN: DISABLE_INFER_AS=1 %clang_cc1 -triple spir64-unknown-linux-sycldevice -std=c++11 -fsycl-is-device -disable-llvm-passes -S -emit-llvm -x c++ %s -o - | FileCheck %s --check-prefixes CHECK,CHECK-OLD
// RUN: %clang_cc1 -triple spir64-unknown-linux-sycldevice -std=c++11 -fsycl-is-device -disable-llvm-passes -S -emit-llvm -x c++ %s -o - | FileCheck %s --check-prefixes CHECK,CHECK-NEW

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(Func kernelFunc) {
  kernelFunc();
}

int main() {

  // CHECK: define spir_kernel void @_ZTSZ4mainE15kernel_function()

  // CHECK-OLD: call spir_func void @"_ZZ4mainENK3$_0clEv"(%"class.{{.*}}.anon"* %0)
  // CHECK-NEW: call spir_func void @"_ZZ4mainENK3$_0clEv"(%"class.{{.*}}.anon" addrspace(4)* %2)

  // CHECK-OLD: define internal spir_func void @"_ZZ4mainENK3$_0clEv"(%"class.{{.*}}anon"* %this)
  // CHECK-NEW: define internal spir_func void @"_ZZ4mainENK3$_0clEv"(%"class.{{.*}}anon" addrspace(4)* %this)

  kernel_single_task<class kernel_function>([]() {});
  return 0;
}

// TODO: SYCL specific fail - analyze and enable
// XFAIL: windows-msvc
