// RUN: %clang_cc1 -fsycl -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown-sycldevice -fsycl-int-header=%t.h %s -fsyntax-only
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

// CHECK:template <> struct KernelInfo<class GNU> {
// CHECK-NEXT:  __SYCL_DLL_LOCAL
// CHECK-NEXT:    static constexpr const char* getName() { return "_ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE3GNU"; }
// CHECK-NEXT:  __SYCL_DLL_LOCAL
// CHECK-NEXT:    static constexpr unsigned getNumParams() { return 0; }
// CHECK-NEXT:  __SYCL_DLL_LOCAL
// CHECK-NEXT:    static constexpr const kernel_param_desc_t& getParamDesc(unsigned i) {
// CHECK-NEXT:    return kernel_signatures[i+0];
// CHECK-NEXT:  }
// CHECK-NEXT:  __SYCL_DLL_LOCAL
// CHECK-NEXT:  static constexpr bool isESIMD() { return 0; }
// CHECK-NEXT:  __SYCL_DLL_LOCAL
// CHECK-NEXT:  static constexpr bool callsThisItem() { return 0; }
// CHECK-NEXT:};
// CHECK-NEXT:template <> struct KernelInfo<class EMU> {
// CHECK-NEXT:  __SYCL_DLL_LOCAL
// CHECK-NEXT:    static constexpr const char* getName() { return "_ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE3EMU"; }
// CHECK-NEXT:  __SYCL_DLL_LOCAL
// CHECK-NEXT:    static constexpr unsigned getNumParams() { return 0; }
// CHECK-NEXT:  __SYCL_DLL_LOCAL
// CHECK-NEXT:    static constexpr const kernel_param_desc_t& getParamDesc(unsigned i) {
// CHECK-NEXT:    return kernel_signatures[i+0];
// CHECK-NEXT:  }
// CHECK-NEXT:  __SYCL_DLL_LOCAL
// CHECK-NEXT:  static constexpr bool isESIMD() { return 0; }
// CHECK-NEXT:  __SYCL_DLL_LOCAL
// CHECK-NEXT:    static constexpr bool callsThisItem() { return 1; }
// CHECK-NEXT:};
// CHECK-NEXT:template <> struct KernelInfo<class OWL> {
// CHECK-NEXT:  __SYCL_DLL_LOCAL
// CHECK-NEXT:    static constexpr const char* getName() { return "_ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE3OWL"; }
// CHECK-NEXT:  __SYCL_DLL_LOCAL
// CHECK-NEXT:    static constexpr unsigned getNumParams() { return 0; }
// CHECK-NEXT:  __SYCL_DLL_LOCAL
// CHECK-NEXT:    static constexpr const kernel_param_desc_t& getParamDesc(unsigned i) {
// CHECK-NEXT:    return kernel_signatures[i+0];
// CHECK-NEXT:  }
// CHECK-NEXT:  __SYCL_DLL_LOCAL
// CHECK-NEXT:  static constexpr bool isESIMD() { return 0; }
// CHECK-NEXT:  __SYCL_DLL_LOCAL
// CHECK-NEXT:    static constexpr bool callsThisItem() { return 0; }
// CHECK-NEXT:};
// CHECK-NEXT:template <> struct KernelInfo<class RAT> {
// CHECK-NEXT:  __SYCL_DLL_LOCAL
// CHECK-NEXT:    static constexpr const char* getName() { return "_ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE3RAT"; }
// CHECK-NEXT:  __SYCL_DLL_LOCAL
// CHECK-NEXT:    static constexpr unsigned getNumParams() { return 0; }
// CHECK-NEXT:  __SYCL_DLL_LOCAL
// CHECK-NEXT:    static constexpr const kernel_param_desc_t& getParamDesc(unsigned i) {
// CHECK-NEXT:    return kernel_signatures[i+0];
// CHECK-NEXT: }
// CHECK-NEXT:  __SYCL_DLL_LOCAL
// CHECK-NEXT:  static constexpr bool isESIMD() { return 0; }
// CHECK-NEXT: __SYCL_DLL_LOCAL
// CHECK-NEXT:    static constexpr bool callsThisItem() { return 1; }
// CHECK-NEXT:};

#include "sycl.hpp"

using namespace cl::sycl;

int main() {
  ::queue myQueue;
  myQueue.submit([&](::handler &cgh) {
    cgh.parallel_for<class GNU>(::range<1>(1),
                                [=](::item<1> I) {});
    cgh.parallel_for<class EMU>(
        ::range<1>(1),
        [=](::item<1> I) { ::this_item<1>(); });
    cgh.parallel_for<class OWL>(::range<1>(1),
                                [=](::id<1> I) {});
    cgh.parallel_for<class RAT>(::range<1>(1), [=](::id<1> I) {
      ::this_item<1>();
    });
  });

  return 0;
}
