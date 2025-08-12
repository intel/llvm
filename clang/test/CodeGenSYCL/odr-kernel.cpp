// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown -sycl-std=2020 -emit-llvm -o - %s | FileCheck %s
//
// Kernel definition may be shared by multiple translation unit if a kernel is
// defined as a functor in a header file. Therefore, we need to make sure that
// the linkage for emitted kernel is correct, i.e. it allows to merge the same
// symbols without triggering multiple definitions error.

#include "sycl.hpp"

// CHECK-DAG: define weak_odr spir_kernel void @_ZTS13FunctorInline
// CHECK-DAG: define dso_local spir_kernel void @_ZTS15FunctorNoInline
// CHECK-DAG: define dso_local spir_kernel void @_ZTSZ4mainE10KernelName
// CHECK-DAG: define dso_local spir_kernel void @_Z32__sycl_kernel_FreeFunctionKernelv
// CHECK-DAG: define weak_odr spir_kernel void @_Z38__sycl_kernel_FreeFunctionKernelInlinev

class FunctorInline {
public:
  void operator()(sycl::id<1>) const {}
};

class FunctorNoInline {
public:
  void operator()(sycl::id<1>) const;
};
void FunctorNoInline::operator()(sycl::id<1>) const {}

class FunctorNoInline2 {
public:
  void operator()() const;
};
void FunctorNoInline2::operator()() const {}


[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 2)]]
void FreeFunctionKernel() {}

[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 2)]]
inline void FreeFunctionKernelInline() {}


struct KernelLaunchWrapper {
  template <typename KernelName, typename KernelType>
  __attribute__((sycl_kernel))
  static void kernel_single_task(const KernelType &kernelFunc) {
    kernelFunc();
  }
};

int main() {
  sycl::queue q;

  q.submit([&](sycl::handler &cgh) {
    FunctorInline f;
    cgh.parallel_for(sycl::range<1>(1024), f);
  });

  q.submit([&](sycl::handler &cgh) {
    FunctorNoInline f;
    cgh.parallel_for(sycl::range<1>(1024), f);
  });

  {
    FunctorNoInline2 f;
    KernelLaunchWrapper::kernel_single_task<class KernelName>(f);
  }
}
