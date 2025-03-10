// RUN: %clang_cc1 -sycl-std=2020 -fsycl-is-device -fsycl-allow-func-ptr=defined -internal-isystem %S/Inputs -disable-llvm-passes -triple spir64-unknown-unknown -emit-llvm -o - %s | FileCheck %s

// Test that the type of function object invoked from the kernel has
// the right address space.

#include "sycl.hpp"

using namespace sycl;
queue q;

int bar10() { return 10; }
[[intel::device_indirectly_callable]] int bar20() { return 20; }

template <typename Callable>
auto invoke_function(Callable &&f) {
  return f();
}

int main() {
  kernel_single_task<class KernelName>(
      [=]() {
        invoke_function(bar10);
        invoke_function(bar20);
      });
  return 0;
}

// CHECK: define dso_local spir_func noundef i32 @{{.*}}bar20{{.*}}()

// CHECK: @_ZZ4mainENKUlvE_clEv
// CHECK: call {{.*}}invoke_function{{.*}}(ptr noundef nonnull @_Z5bar10v)
// CHECK: call {{.*}}invoke_function{{.*}}(ptr noundef nonnull @_Z5bar20v)

// CHECK: define linkonce_odr spir_func noundef i32 @{{.*}}invoke_function{{.*}}(ptr noundef nonnull %f)
// CHECK: %f.addr = alloca ptr, align 8
// CHECK: %f.addr.ascast = addrspacecast ptr %f.addr to ptr addrspace(4)
// CHECK: store ptr %f, ptr addrspace(4) %f.addr.ascast, align 8
// CHECK: %0 = load ptr, ptr addrspace(4) %f.addr.ascast, align 8
// CHECK: %call = call spir_func noundef i32 %0()


// CHECK: define linkonce_odr spir_func noundef i32 @{{.*}}bar10{{.*}}()
