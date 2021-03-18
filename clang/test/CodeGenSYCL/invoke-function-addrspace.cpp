// RUN: %clang_cc1 -sycl-std=2020 -fsycl-is-device -fsycl-allow-func-ptr -internal-isystem %S/Inputs -disable-llvm-passes -triple spir64-unknown-unknown-sycldevice -emit-llvm -o - %s | FileCheck %s

// Test that the type of function object invoked from the kernel has
// the right address space.

#include "sycl.hpp"

using namespace cl::sycl;
queue q;

// CHECK: define linkonce_odr spir_func i32 @{{.*}}invoke_function{{.*}}(i32 () addrspace(4)* %f)
template <typename Callable>
auto invoke_function(Callable &&f) {
  // CHECK: %f.addr = alloca i32 () addrspace(4)*, align 8
  // CHECK: %f.addr.ascast = addrspacecast i32 () addrspace(4)** %f.addr to i32 () addrspace(4)* addrspace(4)*
  // CHECK: store i32 () addrspace(4)* %f, i32 () addrspace(4)* addrspace(4)* %f.addr.ascast, align 8
  // CHECK: %0 = load i32 () addrspace(4)*, i32 () addrspace(4)* addrspace(4)* %f.addr.ascast, align 8
  // CHECK: %call = call spir_func addrspace(4) i32 %0()
  return f();
}

// CHECK: define dso_local spir_func i32 @{{.*}}bar10{{.*}}()
int bar10() { return 10; }

int main() {
  kernel_single_task<class KernelName>(
      [=]() {
        invoke_function(bar10);
      });
  return 0;
}
