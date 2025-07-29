// This is auto-generated SYCL integration header.

#include <sycl/detail/defines_elementary.hpp>
#include <sycl/detail/kernel_desc.hpp>
#include <sycl/ext/oneapi/experimental/free_function_traits.hpp>

#ifndef SYCL_LANGUAGE_VERSION
#define SYCL_LANGUAGE_VERSION 202012L
#endif //SYCL_LANGUAGE_VERSION

// Forward declarations of templated kernel function types:
class MyKernel;
namespace sycl { inline namespace _V1 { namespace detail { 
template <typename Name> class __pf_kernel_wrapper;
}}}

namespace sycl {
inline namespace _V1 {
namespace detail {
// names of all kernels defined in the corresponding source
static constexpr
const char* const kernel_names[] = {
  "_ZTSN4sycl3_V16detail19__pf_kernel_wrapperIZZ4mainENKUlRNS0_7handlerEE_clES4_E8MyKernelEE",
  "_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E8MyKernel",
  "",
};

// array representing signatures of all kernels defined in the
// corresponding source
static constexpr
const kernel_param_desc_t kernel_signatures[] = {
  //--- _ZTSN4sycl3_V16detail19__pf_kernel_wrapperIZZ4mainENKUlRNS0_7handlerEE_clES4_E8MyKernelEE
  { kernel_param_kind_t::kind_std_layout, 8, 0 },
  { kernel_param_kind_t::kind_accessor, 4062, 8 },

  //--- _ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E8MyKernel
  { kernel_param_kind_t::kind_accessor, 4062, 0 },

  { kernel_param_kind_t::kind_invalid, -987654321, -987654321 }, 
};

// Specializations of KernelInfo for kernel function types:
template <> struct KernelInfo<::sycl::detail::__pf_kernel_wrapper<MyKernel>> {
  __SYCL_DLL_LOCAL
  static constexpr const char* getName() { return "_ZTSN4sycl3_V16detail19__pf_kernel_wrapperIZZ4mainENKUlRNS0_7handlerEE_clES4_E8MyKernelEE"; }
  __SYCL_DLL_LOCAL
  static constexpr unsigned getNumParams() { return 2; }
  __SYCL_DLL_LOCAL
  static constexpr const kernel_param_desc_t& getParamDesc(unsigned i) {
    return kernel_signatures[i+0];
  }
  __SYCL_DLL_LOCAL
  static constexpr bool isESIMD() { return 0; }
  __SYCL_DLL_LOCAL
  static constexpr const char* getFileName() {
#ifndef NDEBUG
    return "handler.hpp";
#else
    return "";
#endif
  }
  __SYCL_DLL_LOCAL
  static constexpr const char* getFunctionName() {
#ifndef NDEBUG
    return "::sycl::detail::__pf_kernel_wrapper<MyKernel>";
#else
    return "";
#endif
  }
  __SYCL_DLL_LOCAL
  static constexpr unsigned getLineNumber() {
#ifndef NDEBUG
    return 330;
#else
    return 0;
#endif
  }
  __SYCL_DLL_LOCAL
  static constexpr unsigned getColumnNumber() {
#ifndef NDEBUG
    return 7;
#else
    return 0;
#endif
  }
  // Returns the size of the kernel object in bytes.
  __SYCL_DLL_LOCAL
  static constexpr long getKernelSize() { return 40; }
};
template <> struct KernelInfo<MyKernel> {
  __SYCL_DLL_LOCAL
  static constexpr const char* getName() { return "_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E8MyKernel"; }
  __SYCL_DLL_LOCAL
  static constexpr unsigned getNumParams() { return 1; }
  __SYCL_DLL_LOCAL
  static constexpr const kernel_param_desc_t& getParamDesc(unsigned i) {
    return kernel_signatures[i+2];
  }
  __SYCL_DLL_LOCAL
  static constexpr bool isESIMD() { return 0; }
  __SYCL_DLL_LOCAL
  static constexpr const char* getFileName() {
#ifndef NDEBUG
    return "check_large_access.cpp";
#else
    return "";
#endif
  }
  __SYCL_DLL_LOCAL
  static constexpr const char* getFunctionName() {
#ifndef NDEBUG
    return "MyKernel";
#else
    return "";
#endif
  }
  __SYCL_DLL_LOCAL
  static constexpr unsigned getLineNumber() {
#ifndef NDEBUG
    return 20;
#else
    return 0;
#endif
  }
  __SYCL_DLL_LOCAL
  static constexpr unsigned getColumnNumber() {
#ifndef NDEBUG
    return 32;
#else
    return 0;
#endif
  }
  // Returns the size of the kernel object in bytes.
  __SYCL_DLL_LOCAL
  static constexpr long getKernelSize() { return 32; }
};

} // namespace detail
} // namespace _V1
} // namespace sycl
