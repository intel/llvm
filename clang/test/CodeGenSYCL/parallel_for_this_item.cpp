// RUN: %clang_cc1 -fsycl -fsycl-is-device -triple spir64-unknown-unknown-sycldevice -fsycl-int-header=%t.h %s -fsyntax-only
// RUN: FileCheck -input-file=%t.h %s

// This test checks that compiler generates correct kernel description
// for parallel_for kernels that use the this_item API.

// CHECK: __SYCL_INLINE_NAMESPACE(cl) {
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: namespace detail {

// CHECK: static constexpr
// CHECK-NEXT: const char* const kernel_names[] = {
// CHECK-NEXT:   "_ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE3GNU",
// CHECK-NEXT:   "_ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE3EMU",
// CHECK-NEXT:   "_ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE3OWL",
// CHECK-NEXT:   "_ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE3RAT"
// CHECK-NEXT: };

// CHECK: template <> struct KernelInfo<class GNU> {
// CHECK: __SYCL_DLL_LOCAL
// CHECK_NEXT: static constexpr bool callsThisItem() { return 0; }

// CHECK: template <> struct KernelInfo<class EMU> {
// CHECK: __SYCL_DLL_LOCAL
// CHECK_NEXT: static constexpr bool callsThisItem() { return 1; }

// CHECK: template <> struct KernelInfo<class OWL> {
// CHECK: __SYCL_DLL_LOCAL
// CHECK_NEXT: static constexpr bool callsThisItem() { return 0; }

// CHECK: template <> struct KernelInfo<class RAT> {
// CHECK: __SYCL_DLL_LOCAL
// CHECK_NEXT: static constexpr bool callsThisItem() { return 1; }

#include "Inputs/sycl.hpp"

int main() {
  cl::sycl::queue myQueue;
  myQueue.submit([&](cl::sycl::handler &cgh) {
    cgh.parallel_for<class GNU>(cl::sycl::range<1>(1),
                                [=](cl::sycl::item<1> I) {});
    cgh.parallel_for<class EMU>(
        cl::sycl::range<1>(1),
        [=](cl::sycl::item<1> I) { cl::sycl::this_item<1>(); });
    cgh.parallel_for<class OWL>(cl::sycl::range<1>(1),
                                [=](cl::sycl::id<1> I) {});
    cgh.parallel_for<class RAT>(cl::sycl::range<1>(1), [=](cl::sycl::id<1> I) {
      cl::sycl::this_item<1>();
    });
  });

  return 0;
}
