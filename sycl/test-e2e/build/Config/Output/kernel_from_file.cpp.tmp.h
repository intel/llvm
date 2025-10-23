// This is auto-generated SYCL integration header.

#include <sycl/detail/defines_elementary.hpp>
#include <sycl/detail/kernel_desc.hpp>
#include <sycl/ext/oneapi/experimental/free_function_traits.hpp>
#include <sycl/access/access.hpp>

#ifndef SYCL_LANGUAGE_VERSION
#define SYCL_LANGUAGE_VERSION 202012L
#endif //SYCL_LANGUAGE_VERSION

// Forward declarations of templated kernel function types:
class my_kernel;

namespace sycl {
inline namespace _V1 {
namespace detail {
// names of all kernels defined in the corresponding source
static constexpr
const char* const kernel_names[] = {
  "_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9my_kernel",
  "",
};

// array representing signatures of all kernels defined in the
// corresponding source
static constexpr
const kernel_param_desc_t kernel_signatures[] = {
  //--- _ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9my_kernel
  { kernel_param_kind_t::kind_accessor, 4062, 0 },

  { kernel_param_kind_t::kind_invalid, -987654321, -987654321 }, 
};

// Specializations of KernelInfo for kernel function types:
template <> struct KernelInfo<my_kernel> {
  __SYCL_DLL_LOCAL
  static constexpr const char* getName() { return "_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9my_kernel"; }
  __SYCL_DLL_LOCAL
  static constexpr unsigned getNumParams() { return 1; }
  __SYCL_DLL_LOCAL
  static constexpr const kernel_param_desc_t& getParamDesc(unsigned i) {
    return kernel_signatures[i+0];
  }
  __SYCL_DLL_LOCAL
  static constexpr bool isESIMD() { return 0; }
  __SYCL_DLL_LOCAL
  static constexpr const char* getFileName() {
#ifndef NDEBUG
    return "kernel_from_file.cpp";
#else
    return "";
#endif
  }
  __SYCL_DLL_LOCAL
  static constexpr const char* getFunctionName() {
#ifndef NDEBUG
    return "my_kernel";
#else
    return "";
#endif
  }
  __SYCL_DLL_LOCAL
  static constexpr unsigned getLineNumber() {
#ifndef NDEBUG
    return 34;
#else
    return 0;
#endif
  }
  __SYCL_DLL_LOCAL
  static constexpr unsigned getColumnNumber() {
#ifndef NDEBUG
    return 40;
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
