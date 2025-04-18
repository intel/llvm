// RUN: %clangxx -fsycl-device-only -Xclang -fsycl-is-device -emit-llvm %s -S -o %t.ll
// RUN: FileCheck %s --input-file %t.ll

// Check SYCL FE metadata is updated when dead argument elimination removes an
// argument

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>
using namespace sycl;

template <typename name, typename Func>
__attribute__((sycl_kernel)) void my_kernel(Func kernelFunc) {
  kernelFunc();
}

SYCL_EXTERNAL SYCL_ESIMD_FUNCTION ESIMD_NOINLINE void callee(int x) {}

// CHECK: define dso_local spir_kernel {{.*}} !sycl_kernel_omit_args ![[#MD:]]
SYCL_EXTERNAL void __attribute__((noinline)) caller(int x) {
  my_kernel<class kernel_abc>([=]() SYCL_ESIMD_KERNEL { callee(x); });
}

//CHECK: [[#MD]] = !{i1 true}
