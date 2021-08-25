// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown-sycldevice -fsycl-int-header=%t.h %s -fsyntax-only
// RUN: FileCheck -input-file=%t.h %s

// This test checks that compiler generates correct kernel description
// for parallel_for kernels that use the this_item API.

// CHECK: __SYCL_INLINE_NAMESPACE(cl) {
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: namespace detail {

// CHECK: static constexpr
// CHECK-NEXT: const char* const kernel_names[] = {
// CHECK-NEXT:   "_ZTSZZ4mainENKUlRN2cl4sycl7handlerEE_clES2_E3GNU",
// CHECK-NEXT:   "_ZTSZZ4mainENKUlRN2cl4sycl7handlerEE_clES2_E3EMU",
// CHECK-NEXT:   "_ZTSZZ4mainENKUlRN2cl4sycl7handlerEE_clES2_E3COW",
// CHECK-NEXT:   "_ZTSZZ4mainENKUlRN2cl4sycl7handlerEE_clES2_E3OWL",
// CHECK-NEXT:   "_ZTSZZ4mainENKUlRN2cl4sycl7handlerEE_clES2_E3RAT",
// CHECK-NEXT:   "_ZTSZZ4mainENKUlRN2cl4sycl7handlerEE_clES2_E3CAT",
// CHECK-NEXT:   "_ZTSZZ4mainENKUlRN2cl4sycl7handlerEE_clES2_E3FOX",
// CHECK-NEXT:   "_ZTSZZ4mainENKUlRN2cl4sycl7handlerEE_clES2_E3PIG",
// CHECK-NEXT:   "_ZTSZZ4mainENKUlRN2cl4sycl7handlerEE_clES2_E3BEE",
// CHECK-NEXT:   "_ZTSZZ4mainENKUlRN2cl4sycl7handlerEE_clES2_E3DOG"
// CHECK-NEXT: };

// CHECK:template <> struct KernelInfo<GNU> {
// CHECK-NEXT:  __SYCL_DLL_LOCAL
// CHECK-NEXT:    static constexpr const char* getName() { return "_ZTSZZ4mainENKUlRN2cl4sycl7handlerEE_clES2_E3GNU"; }
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
// CHECK-NEXT:  __SYCL_DLL_LOCAL
// CHECK-NEXT:  static constexpr bool callsAnyThisFreeFunction() { return 0; }
// CHECK-NEXT:};
// CHECK-NEXT:template <> struct KernelInfo<EMU> {
// CHECK-NEXT:  __SYCL_DLL_LOCAL
// CHECK-NEXT:    static constexpr const char* getName() { return "_ZTSZZ4mainENKUlRN2cl4sycl7handlerEE_clES2_E3EMU"; }
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
// CHECK-NEXT:  __SYCL_DLL_LOCAL
// CHECK-NEXT:    static constexpr bool callsAnyThisFreeFunction() { return 1; }
// CHECK-NEXT:};
// CHECK-NEXT:template <> struct KernelInfo<COW> {
// CHECK-NEXT:  __SYCL_DLL_LOCAL
// CHECK-NEXT:    static constexpr const char* getName() { return "_ZTSZZ4mainENKUlRN2cl4sycl7handlerEE_clES2_E3COW"; }
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
// CHECK-NEXT:  __SYCL_DLL_LOCAL
// CHECK-NEXT:    static constexpr bool callsAnyThisFreeFunction() { return 1; }
// CHECK-NEXT:};
// CHECK-NEXT:template <> struct KernelInfo<OWL> {
// CHECK-NEXT:  __SYCL_DLL_LOCAL
// CHECK-NEXT:    static constexpr const char* getName() { return "_ZTSZZ4mainENKUlRN2cl4sycl7handlerEE_clES2_E3OWL"; }
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
// CHECK-NEXT:  __SYCL_DLL_LOCAL
// CHECK-NEXT:    static constexpr bool callsAnyThisFreeFunction() { return 0; }
// CHECK-NEXT:};
// CHECK-NEXT:template <> struct KernelInfo<RAT> {
// CHECK-NEXT:  __SYCL_DLL_LOCAL
// CHECK-NEXT:    static constexpr const char* getName() { return "_ZTSZZ4mainENKUlRN2cl4sycl7handlerEE_clES2_E3RAT"; }
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
// CHECK-NEXT: __SYCL_DLL_LOCAL
// CHECK-NEXT:    static constexpr bool callsAnyThisFreeFunction() { return 1; }
// CHECK-NEXT:};
// CHECK-NEXT:template <> struct KernelInfo<CAT> {
// CHECK-NEXT:  __SYCL_DLL_LOCAL
// CHECK-NEXT:    static constexpr const char* getName() { return "_ZTSZZ4mainENKUlRN2cl4sycl7handlerEE_clES2_E3CAT"; }
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
// CHECK-NEXT: __SYCL_DLL_LOCAL
// CHECK-NEXT:    static constexpr bool callsAnyThisFreeFunction() { return 1; }
// CHECK-NEXT:};
// CHECK-NEXT:template <> struct KernelInfo<FOX> {
// CHECK-NEXT:  __SYCL_DLL_LOCAL
// CHECK-NEXT:    static constexpr const char* getName() { return "_ZTSZZ4mainENKUlRN2cl4sycl7handlerEE_clES2_E3FOX"; }
// CHECK-NEXT:  __SYCL_DLL_LOCAL
// CHECK-NEXT:    static constexpr unsigned getNumParams() { return 0; }
// CHECK-NEXT:  __SYCL_DLL_LOCAL
// CHECK-NEXT:    static constexpr const kernel_param_desc_t& getParamDesc(unsigned i) {
// CHECK-NEXT:    return kernel_signatures[i+0];
// CHECK-NEXT: }
// CHECK-NEXT:  __SYCL_DLL_LOCAL
// CHECK-NEXT:  static constexpr bool isESIMD() { return 0; }
// CHECK-NEXT: __SYCL_DLL_LOCAL
// CHECK-NEXT:    static constexpr bool callsThisItem() { return 0; }
// CHECK-NEXT: __SYCL_DLL_LOCAL
// CHECK-NEXT:    static constexpr bool callsAnyThisFreeFunction() { return 1; }
// CHECK-NEXT:};
// CHECK-NEXT:template <> struct KernelInfo<PIG> {
// CHECK-NEXT:  __SYCL_DLL_LOCAL
// CHECK-NEXT:    static constexpr const char* getName() { return "_ZTSZZ4mainENKUlRN2cl4sycl7handlerEE_clES2_E3PIG"; }
// CHECK-NEXT:  __SYCL_DLL_LOCAL
// CHECK-NEXT:    static constexpr unsigned getNumParams() { return 0; }
// CHECK-NEXT:  __SYCL_DLL_LOCAL
// CHECK-NEXT:    static constexpr const kernel_param_desc_t& getParamDesc(unsigned i) {
// CHECK-NEXT:    return kernel_signatures[i+0];
// CHECK-NEXT: }
// CHECK-NEXT:  __SYCL_DLL_LOCAL
// CHECK-NEXT:  static constexpr bool isESIMD() { return 0; }
// CHECK-NEXT: __SYCL_DLL_LOCAL
// CHECK-NEXT:    static constexpr bool callsThisItem() { return 0; }
// CHECK-NEXT: __SYCL_DLL_LOCAL
// CHECK-NEXT:    static constexpr bool callsAnyThisFreeFunction() { return 1; }
// CHECK-NEXT:};
// CHECK-NEXT:template <> struct KernelInfo<BEE> {
// CHECK-NEXT:  __SYCL_DLL_LOCAL
// CHECK-NEXT:    static constexpr const char* getName() { return "_ZTSZZ4mainENKUlRN2cl4sycl7handlerEE_clES2_E3BEE"; }
// CHECK-NEXT:  __SYCL_DLL_LOCAL
// CHECK-NEXT:    static constexpr unsigned getNumParams() { return 0; }
// CHECK-NEXT:  __SYCL_DLL_LOCAL
// CHECK-NEXT:    static constexpr const kernel_param_desc_t& getParamDesc(unsigned i) {
// CHECK-NEXT:    return kernel_signatures[i+0];
// CHECK-NEXT:  }
// CHECK-NEXT:  __SYCL_DLL_LOCAL
// CHECK-NEXT:    static constexpr bool isESIMD() { return 0; }
// CHECK-NEXT:  __SYCL_DLL_LOCAL
// CHECK-NEXT:    static constexpr bool callsThisItem() { return 1; }
// CHECK-NEXT: __SYCL_DLL_LOCAL
// CHECK-NEXT:    static constexpr bool callsAnyThisFreeFunction() { return 1; }
// CHECK-NEXT:};
// CHECK-NEXT:template <> struct KernelInfo<DOG> {
// CHECK-NEXT:  __SYCL_DLL_LOCAL
// CHECK-NEXT:    static constexpr const char* getName() { return "_ZTSZZ4mainENKUlRN2cl4sycl7handlerEE_clES2_E3DOG"; }
// CHECK-NEXT:  __SYCL_DLL_LOCAL
// CHECK-NEXT:    static constexpr unsigned getNumParams() { return 0; }
// CHECK-NEXT:  __SYCL_DLL_LOCAL
// CHECK-NEXT:    static constexpr const kernel_param_desc_t& getParamDesc(unsigned i) {
// CHECK-NEXT:    return kernel_signatures[i+0];
// CHECK-NEXT:  }
// CHECK-NEXT:  __SYCL_DLL_LOCAL
// CHECK-NEXT:    static constexpr bool isESIMD() { return 0; }
// CHECK-NEXT:  __SYCL_DLL_LOCAL
// CHECK-NEXT:    static constexpr bool callsThisItem() { return 1; }
// CHECK-NEXT: __SYCL_DLL_LOCAL
// CHECK-NEXT:    static constexpr bool callsAnyThisFreeFunction() { return 1; }
// CHECK-NEXT:};

#include "sycl.hpp"

using namespace cl::sycl;

SYCL_EXTERNAL item<1> g() { return this_item<1>(); }
SYCL_EXTERNAL item<1> f() { return g(); }
SYCL_EXTERNAL item<1> s() { return ext::oneapi::experimental::this_item<1>(); }
SYCL_EXTERNAL item<1> h() { return s(); }

// This is a similar-looking this_item function but not the real one.
template <int Dims> item<Dims> this_item(int i) { return item<1>{i}; }

// This is a method named this_item but not the real one.
class C {
public:
  template <int Dims> item<Dims> this_item() { return item<1>{66}; };
};

int main() {
  queue myQueue;
  myQueue.submit([&](::handler &cgh) {
    // This kernel does not call sycl::this_item
    cgh.parallel_for<class GNU>(range<1>(1),
                                [=](item<1> I) { this_item<1>(55); });

    // This kernel calls sycl::this_item
    cgh.parallel_for<class EMU>(range<1>(1),
                                [=](::item<1> I) { this_item<1>(); });

    // This kernel calls sycl::ext::oneapi::experimental::this_item
    cgh.parallel_for<class COW>(range<1>(1), [=](::item<1> I) {
      ext::oneapi::experimental::this_item<1>();
    });

    // This kernel does not call sycl::this_item
    cgh.parallel_for<class OWL>(range<1>(1), [=](id<1> I) {
      class C c;
      c.this_item<1>();
    });

    // This kernel calls sycl::this_item
    cgh.parallel_for<class RAT>(range<1>(1), [=](id<1> I) { f(); });

    // This kernel calls sycl::ext::oneapi::experimental::this_item
    cgh.parallel_for<class CAT>(range<1>(1), [=](id<1> I) { h(); });

    // This kernel does not call sycl::this_item, but does call this_id
    cgh.parallel_for<class FOX>(range<1>(1), [=](id<1> I) { this_id<1>(); });

    // This kernel calls sycl::ext::oneapi::experimental::this_id
    cgh.parallel_for<class PIG>(range<1>(1), [=](id<1> I) {
      ext::oneapi::experimental::this_id<1>();
    });

    // This kernel calls sycl::this_item
    cgh.parallel_for<class BEE>(range<1>(1), [=](auto I) { this_item<1>(); });

    // This kernel calls sycl::ext::oneapi::experimental::this_item
    cgh.parallel_for<class DOG>(range<1>(1), [=](auto I) {
      ext::oneapi::experimental::this_item<1>();
    });
  });

  return 0;
}
