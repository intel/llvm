// RUN: %clang_cc1 -sycl-std=2020 -fsycl-is-device -fsycl-allow-func-ptr -internal-isystem %S/Inputs -disable-llvm-passes -triple spir64-unknown-unknown -opaque-pointers -emit-llvm -o - %s | FileCheck %s

// Test that the type of function object invoked from the kernel has
// the right address space.

#include "sycl.hpp"

using namespace sycl;
queue q;

// CHECK: define dso_local spir_func noundef i32 @{{.*}}bar10{{.*}}()
[[intel::device_indirectly_callable]] int bar10() { return 10; }

// CHECK: define linkonce_odr spir_func noundef i32 @{{.*}}invoke_function{{.*}}(ptr noundef nonnull %f)
template <typename Callable>
auto invoke_function(Callable &&f) {
  // CHECK: %f.addr = alloca ptr, align 8
  // CHECK: %f.addr.ascast = addrspacecast ptr %f.addr to ptr addrspace(4)
  // CHECK: store ptr %f, ptr addrspace(4) %f.addr.ascast, align 8
  // CHECK: %0 = load ptr, ptr addrspace(4) %f.addr.ascast, align 8
  // CHECK: %call = call spir_func noundef i32 %0()
  return f();
}

int main() {
  kernel_single_task<class KernelName>(
      [=]() {
        invoke_function(bar10);
      });
  return 0;
}
