// RUN: %clangxx -fsycl-device-only -S -emit-llvm %s -o - | FileCheck %s  --check-prefix CHECK-IR

#include "sycl/sycl.hpp"

namespace syclexp = sycl::ext::oneapi::experimental;

int main() {
  sycl::queue Q;

  auto Ptr = sycl::malloc_shared<int>(1, Q);
  syclexp::annotated_ptr<int, decltype(syclexp::properties(syclexp::restrict))>
      AnnotPtr{Ptr};
  Q.submit([&](sycl::handler &CGH) {
     CGH.single_task([=]() { *AnnotPtr = 42; });
   }).wait();
  free(Ptr, Q);

  return 0;
}

// CHECK-IR: spir_kernel void @_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_EUlvE_(ptr addrspace(1) noalias noundef align 4 "sycl-restrict" %_arg_AnnotPtr)
