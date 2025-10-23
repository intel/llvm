// This is auto-generated SYCL integration header.

#include <sycl/detail/defines_elementary.hpp>
#include <sycl/detail/kernel_desc.hpp>
#include <sycl/ext/oneapi/experimental/free_function_traits.hpp>
#include <sycl/access/access.hpp>

#ifndef SYCL_LANGUAGE_VERSION
#define SYCL_LANGUAGE_VERSION 202012L
#endif //SYCL_LANGUAGE_VERSION

// Forward declarations of templated kernel function types:

namespace sycl {
inline namespace _V1 {
namespace detail {
// names of all kernels defined in the corresponding source
static constexpr
const char* const kernel_names[] = {
  "__sycl_kernel_ff_0",
  "_Z18__sycl_kernel_ff_1Piii",
  "_Z18__sycl_kernel_ff_1Pii",
  "_Z18__sycl_kernel_ff_3IiEvPT_S0_",
  "_ZTSZZ6test_0N4sycl3_V15queueEENKUlRNS0_7handlerEE_clES3_EUlvE_",
  "_ZTSZZ6test_1N4sycl3_V15queueEENKUlRNS0_7handlerEE_clES3_EUlNS0_7nd_itemILi1EEEE_",
  "_ZTSZZ6test_2N4sycl3_V15queueEENKUlRNS0_7handlerEE_clES3_EUlNS0_7nd_itemILi2EEEE_",
  "_ZTSZZ6test_3N4sycl3_V15queueEENKUlRNS0_7handlerEE_clES3_EUlNS0_7nd_itemILi2EEEE_",
  "",
};

// array representing signatures of all kernels defined in the
// corresponding source
static constexpr
const kernel_param_desc_t kernel_signatures[] = {
  //--- __sycl_kernel_ff_0
  { kernel_param_kind_t::kind_pointer, 8, 0 },
  { kernel_param_kind_t::kind_std_layout, 4, 8 },
  { kernel_param_kind_t::kind_std_layout, 4, 12 },

  //--- _Z18__sycl_kernel_ff_1Piii
  { kernel_param_kind_t::kind_pointer, 8, 0 },
  { kernel_param_kind_t::kind_std_layout, 4, 8 },
  { kernel_param_kind_t::kind_std_layout, 4, 12 },

  //--- _Z18__sycl_kernel_ff_1Pii
  { kernel_param_kind_t::kind_pointer, 8, 0 },
  { kernel_param_kind_t::kind_std_layout, 4, 8 },

  //--- _Z18__sycl_kernel_ff_3IiEvPT_S0_
  { kernel_param_kind_t::kind_pointer, 8, 0 },
  { kernel_param_kind_t::kind_std_layout, 4, 8 },

  //--- _ZTSZZ6test_0N4sycl3_V15queueEENKUlRNS0_7handlerEE_clES3_EUlvE_
  { kernel_param_kind_t::kind_std_layout, 4, 0 },
  { kernel_param_kind_t::kind_std_layout, 4, 4 },
  { kernel_param_kind_t::kind_pointer, 8, 8 },

  //--- _ZTSZZ6test_1N4sycl3_V15queueEENKUlRNS0_7handlerEE_clES3_EUlNS0_7nd_itemILi1EEEE_
  { kernel_param_kind_t::kind_pointer, 8, 0 },
  { kernel_param_kind_t::kind_std_layout, 4, 8 },

  //--- _ZTSZZ6test_2N4sycl3_V15queueEENKUlRNS0_7handlerEE_clES3_EUlNS0_7nd_itemILi2EEEE_
  { kernel_param_kind_t::kind_pointer, 8, 0 },
  { kernel_param_kind_t::kind_std_layout, 4, 8 },

  //--- _ZTSZZ6test_3N4sycl3_V15queueEENKUlRNS0_7handlerEE_clES3_EUlNS0_7nd_itemILi2EEEE_
  { kernel_param_kind_t::kind_pointer, 8, 0 },
  { kernel_param_kind_t::kind_std_layout, 4, 8 },

  { kernel_param_kind_t::kind_invalid, -987654321, -987654321 }, 
};

// Specializations of KernelInfo for kernel function types:
template <> struct KernelInfoData<'_', '_', 's', 'y', 'c', 'l', '_', 'k', 'e', 'r', 'n', 'e', 'l', '_', 'f', 'f', '_', '0'> {
  __SYCL_DLL_LOCAL
  static constexpr const char* getName() { return "__sycl_kernel_ff_0"; }
  __SYCL_DLL_LOCAL
  static constexpr unsigned getNumParams() { return 3; }
  __SYCL_DLL_LOCAL
  static constexpr const kernel_param_desc_t& getParamDesc(unsigned i) {
    return kernel_signatures[i+0];
  }
  __SYCL_DLL_LOCAL
  static constexpr bool isESIMD() { return 0; }
  __SYCL_DLL_LOCAL
  static constexpr const char* getFileName() {
#ifndef NDEBUG
    return "free_function_kernels.cpp";
#else
    return "";
#endif
  }
  __SYCL_DLL_LOCAL
  static constexpr const char* getFunctionName() {
#ifndef NDEBUG
    return "void (int *, int, int)";
#else
    return "";
#endif
  }
  __SYCL_DLL_LOCAL
  static constexpr unsigned getLineNumber() {
#ifndef NDEBUG
    return 47;
#else
    return 0;
#endif
  }
  __SYCL_DLL_LOCAL
  static constexpr unsigned getColumnNumber() {
#ifndef NDEBUG
    return 403;
#else
    return 0;
#endif
  }
  // Returns the size of the kernel object in bytes.
  __SYCL_DLL_LOCAL
  static constexpr long getKernelSize() { return 0; }
};
template <> struct KernelInfoData<'_', 'Z', '1', '8', '_', '_', 's', 'y', 'c', 'l', '_', 'k', 'e', 'r', 'n', 'e', 'l', '_', 'f', 'f', '_', '1', 'P', 'i', 'i', 'i'> {
  __SYCL_DLL_LOCAL
  static constexpr const char* getName() { return "_Z18__sycl_kernel_ff_1Piii"; }
  __SYCL_DLL_LOCAL
  static constexpr unsigned getNumParams() { return 3; }
  __SYCL_DLL_LOCAL
  static constexpr const kernel_param_desc_t& getParamDesc(unsigned i) {
    return kernel_signatures[i+3];
  }
  __SYCL_DLL_LOCAL
  static constexpr bool isESIMD() { return 0; }
  __SYCL_DLL_LOCAL
  static constexpr const char* getFileName() {
#ifndef NDEBUG
    return "free_function_kernels.cpp";
#else
    return "";
#endif
  }
  __SYCL_DLL_LOCAL
  static constexpr const char* getFunctionName() {
#ifndef NDEBUG
    return "void (int *, int, int)";
#else
    return "";
#endif
  }
  __SYCL_DLL_LOCAL
  static constexpr unsigned getLineNumber() {
#ifndef NDEBUG
    return 99;
#else
    return 0;
#endif
  }
  __SYCL_DLL_LOCAL
  static constexpr unsigned getColumnNumber() {
#ifndef NDEBUG
    return 6;
#else
    return 0;
#endif
  }
  // Returns the size of the kernel object in bytes.
  __SYCL_DLL_LOCAL
  static constexpr long getKernelSize() { return 0; }
};
template <> struct KernelInfoData<'_', 'Z', '1', '8', '_', '_', 's', 'y', 'c', 'l', '_', 'k', 'e', 'r', 'n', 'e', 'l', '_', 'f', 'f', '_', '1', 'P', 'i', 'i'> {
  __SYCL_DLL_LOCAL
  static constexpr const char* getName() { return "_Z18__sycl_kernel_ff_1Pii"; }
  __SYCL_DLL_LOCAL
  static constexpr unsigned getNumParams() { return 2; }
  __SYCL_DLL_LOCAL
  static constexpr const kernel_param_desc_t& getParamDesc(unsigned i) {
    return kernel_signatures[i+6];
  }
  __SYCL_DLL_LOCAL
  static constexpr bool isESIMD() { return 0; }
  __SYCL_DLL_LOCAL
  static constexpr const char* getFileName() {
#ifndef NDEBUG
    return "free_function_kernels.cpp";
#else
    return "";
#endif
  }
  __SYCL_DLL_LOCAL
  static constexpr const char* getFunctionName() {
#ifndef NDEBUG
    return "void (int *, int)";
#else
    return "";
#endif
  }
  __SYCL_DLL_LOCAL
  static constexpr unsigned getLineNumber() {
#ifndef NDEBUG
    return 149;
#else
    return 0;
#endif
  }
  __SYCL_DLL_LOCAL
  static constexpr unsigned getColumnNumber() {
#ifndef NDEBUG
    return 6;
#else
    return 0;
#endif
  }
  // Returns the size of the kernel object in bytes.
  __SYCL_DLL_LOCAL
  static constexpr long getKernelSize() { return 0; }
};
template <> struct KernelInfoData<'_', 'Z', '1', '8', '_', '_', 's', 'y', 'c', 'l', '_', 'k', 'e', 'r', 'n', 'e', 'l', '_', 'f', 'f', '_', '3', 'I', 'i', 'E', 'v', 'P', 'T', '_', 'S', '0', '_'> {
  __SYCL_DLL_LOCAL
  static constexpr const char* getName() { return "_Z18__sycl_kernel_ff_3IiEvPT_S0_"; }
  __SYCL_DLL_LOCAL
  static constexpr unsigned getNumParams() { return 2; }
  __SYCL_DLL_LOCAL
  static constexpr const kernel_param_desc_t& getParamDesc(unsigned i) {
    return kernel_signatures[i+8];
  }
  __SYCL_DLL_LOCAL
  static constexpr bool isESIMD() { return 0; }
  __SYCL_DLL_LOCAL
  static constexpr const char* getFileName() {
#ifndef NDEBUG
    return "free_function_kernels.cpp";
#else
    return "";
#endif
  }
  __SYCL_DLL_LOCAL
  static constexpr const char* getFunctionName() {
#ifndef NDEBUG
    return "void (int *, int)";
#else
    return "";
#endif
  }
  __SYCL_DLL_LOCAL
  static constexpr unsigned getLineNumber() {
#ifndef NDEBUG
    return 204;
#else
    return 0;
#endif
  }
  __SYCL_DLL_LOCAL
  static constexpr unsigned getColumnNumber() {
#ifndef NDEBUG
    return 6;
#else
    return 0;
#endif
  }
  // Returns the size of the kernel object in bytes.
  __SYCL_DLL_LOCAL
  static constexpr long getKernelSize() { return 0; }
};
template <> struct KernelInfoData<'_', 'Z', 'T', 'S', 'Z', 'Z', '6', 't', 'e', 's', 't', '_', '0', 'N', '4', 's', 'y', 'c', 'l', '3', '_', 'V', '1', '5', 'q', 'u', 'e', 'u', 'e', 'E', 'E', 'N', 'K', 'U', 'l', 'R', 'N', 'S', '0', '_', '7', 'h', 'a', 'n', 'd', 'l', 'e', 'r', 'E', 'E', '_', 'c', 'l', 'E', 'S', '3', '_', 'E', 'U', 'l', 'v', 'E', '_'> {
  __SYCL_DLL_LOCAL
  static constexpr const char* getName() { return "_ZTSZZ6test_0N4sycl3_V15queueEENKUlRNS0_7handlerEE_clES3_EUlvE_"; }
  __SYCL_DLL_LOCAL
  static constexpr unsigned getNumParams() { return 3; }
  __SYCL_DLL_LOCAL
  static constexpr const kernel_param_desc_t& getParamDesc(unsigned i) {
    return kernel_signatures[i+10];
  }
  __SYCL_DLL_LOCAL
  static constexpr bool isESIMD() { return 0; }
  __SYCL_DLL_LOCAL
  static constexpr const char* getFileName() {
#ifndef NDEBUG
    return "free_function_kernels.cpp";
#else
    return "";
#endif
  }
  __SYCL_DLL_LOCAL
  static constexpr const char* getFunctionName() {
#ifndef NDEBUG
    return "class (lambda)";
#else
    return "";
#endif
  }
  __SYCL_DLL_LOCAL
  static constexpr unsigned getLineNumber() {
#ifndef NDEBUG
    return 65;
#else
    return 0;
#endif
  }
  __SYCL_DLL_LOCAL
  static constexpr unsigned getColumnNumber() {
#ifndef NDEBUG
    return 25;
#else
    return 0;
#endif
  }
  // Returns the size of the kernel object in bytes.
  __SYCL_DLL_LOCAL
  static constexpr long getKernelSize() { return 16; }
};
template <> struct KernelInfoData<'_', 'Z', 'T', 'S', 'Z', 'Z', '6', 't', 'e', 's', 't', '_', '1', 'N', '4', 's', 'y', 'c', 'l', '3', '_', 'V', '1', '5', 'q', 'u', 'e', 'u', 'e', 'E', 'E', 'N', 'K', 'U', 'l', 'R', 'N', 'S', '0', '_', '7', 'h', 'a', 'n', 'd', 'l', 'e', 'r', 'E', 'E', '_', 'c', 'l', 'E', 'S', '3', '_', 'E', 'U', 'l', 'N', 'S', '0', '_', '7', 'n', 'd', '_', 'i', 't', 'e', 'm', 'I', 'L', 'i', '1', 'E', 'E', 'E', 'E', '_'> {
  __SYCL_DLL_LOCAL
  static constexpr const char* getName() { return "_ZTSZZ6test_1N4sycl3_V15queueEENKUlRNS0_7handlerEE_clES3_EUlNS0_7nd_itemILi1EEEE_"; }
  __SYCL_DLL_LOCAL
  static constexpr unsigned getNumParams() { return 2; }
  __SYCL_DLL_LOCAL
  static constexpr const kernel_param_desc_t& getParamDesc(unsigned i) {
    return kernel_signatures[i+13];
  }
  __SYCL_DLL_LOCAL
  static constexpr bool isESIMD() { return 0; }
  __SYCL_DLL_LOCAL
  static constexpr const char* getFileName() {
#ifndef NDEBUG
    return "free_function_kernels.cpp";
#else
    return "";
#endif
  }
  __SYCL_DLL_LOCAL
  static constexpr const char* getFunctionName() {
#ifndef NDEBUG
    return "class (lambda)";
#else
    return "";
#endif
  }
  __SYCL_DLL_LOCAL
  static constexpr unsigned getLineNumber() {
#ifndef NDEBUG
    return 114;
#else
    return 0;
#endif
  }
  __SYCL_DLL_LOCAL
  static constexpr unsigned getColumnNumber() {
#ifndef NDEBUG
    return 30;
#else
    return 0;
#endif
  }
  // Returns the size of the kernel object in bytes.
  __SYCL_DLL_LOCAL
  static constexpr long getKernelSize() { return 16; }
};
template <> struct KernelInfoData<'_', 'Z', 'T', 'S', 'Z', 'Z', '6', 't', 'e', 's', 't', '_', '2', 'N', '4', 's', 'y', 'c', 'l', '3', '_', 'V', '1', '5', 'q', 'u', 'e', 'u', 'e', 'E', 'E', 'N', 'K', 'U', 'l', 'R', 'N', 'S', '0', '_', '7', 'h', 'a', 'n', 'd', 'l', 'e', 'r', 'E', 'E', '_', 'c', 'l', 'E', 'S', '3', '_', 'E', 'U', 'l', 'N', 'S', '0', '_', '7', 'n', 'd', '_', 'i', 't', 'e', 'm', 'I', 'L', 'i', '2', 'E', 'E', 'E', 'E', '_'> {
  __SYCL_DLL_LOCAL
  static constexpr const char* getName() { return "_ZTSZZ6test_2N4sycl3_V15queueEENKUlRNS0_7handlerEE_clES3_EUlNS0_7nd_itemILi2EEEE_"; }
  __SYCL_DLL_LOCAL
  static constexpr unsigned getNumParams() { return 2; }
  __SYCL_DLL_LOCAL
  static constexpr const kernel_param_desc_t& getParamDesc(unsigned i) {
    return kernel_signatures[i+15];
  }
  __SYCL_DLL_LOCAL
  static constexpr bool isESIMD() { return 0; }
  __SYCL_DLL_LOCAL
  static constexpr const char* getFileName() {
#ifndef NDEBUG
    return "free_function_kernels.cpp";
#else
    return "";
#endif
  }
  __SYCL_DLL_LOCAL
  static constexpr const char* getFunctionName() {
#ifndef NDEBUG
    return "class (lambda)";
#else
    return "";
#endif
  }
  __SYCL_DLL_LOCAL
  static constexpr unsigned getLineNumber() {
#ifndef NDEBUG
    return 167;
#else
    return 0;
#endif
  }
  __SYCL_DLL_LOCAL
  static constexpr unsigned getColumnNumber() {
#ifndef NDEBUG
    return 30;
#else
    return 0;
#endif
  }
  // Returns the size of the kernel object in bytes.
  __SYCL_DLL_LOCAL
  static constexpr long getKernelSize() { return 16; }
};
template <> struct KernelInfoData<'_', 'Z', 'T', 'S', 'Z', 'Z', '6', 't', 'e', 's', 't', '_', '3', 'N', '4', 's', 'y', 'c', 'l', '3', '_', 'V', '1', '5', 'q', 'u', 'e', 'u', 'e', 'E', 'E', 'N', 'K', 'U', 'l', 'R', 'N', 'S', '0', '_', '7', 'h', 'a', 'n', 'd', 'l', 'e', 'r', 'E', 'E', '_', 'c', 'l', 'E', 'S', '3', '_', 'E', 'U', 'l', 'N', 'S', '0', '_', '7', 'n', 'd', '_', 'i', 't', 'e', 'm', 'I', 'L', 'i', '2', 'E', 'E', 'E', 'E', '_'> {
  __SYCL_DLL_LOCAL
  static constexpr const char* getName() { return "_ZTSZZ6test_3N4sycl3_V15queueEENKUlRNS0_7handlerEE_clES3_EUlNS0_7nd_itemILi2EEEE_"; }
  __SYCL_DLL_LOCAL
  static constexpr unsigned getNumParams() { return 2; }
  __SYCL_DLL_LOCAL
  static constexpr const kernel_param_desc_t& getParamDesc(unsigned i) {
    return kernel_signatures[i+17];
  }
  __SYCL_DLL_LOCAL
  static constexpr bool isESIMD() { return 0; }
  __SYCL_DLL_LOCAL
  static constexpr const char* getFileName() {
#ifndef NDEBUG
    return "free_function_kernels.cpp";
#else
    return "";
#endif
  }
  __SYCL_DLL_LOCAL
  static constexpr const char* getFunctionName() {
#ifndef NDEBUG
    return "class (lambda)";
#else
    return "";
#endif
  }
  __SYCL_DLL_LOCAL
  static constexpr unsigned getLineNumber() {
#ifndef NDEBUG
    return 225;
#else
    return 0;
#endif
  }
  __SYCL_DLL_LOCAL
  static constexpr unsigned getColumnNumber() {
#ifndef NDEBUG
    return 30;
#else
    return 0;
#endif
  }
  // Returns the size of the kernel object in bytes.
  __SYCL_DLL_LOCAL
  static constexpr long getKernelSize() { return 16; }
};

} // namespace detail
} // namespace _V1
} // namespace sycl

// Definition of __sycl_kernel_ff_0 as a free function kernel

// Forward declarations of kernel and its argument types:

extern "C" void ff_0(int * ptr, int start, int end);
static constexpr auto __sycl_shim1() {
  return (void (*)(int *, int, int))ff_0;
}
namespace sycl {
template <>
struct ext::oneapi::experimental::is_kernel<__sycl_shim1()> {
  static constexpr bool value = true;
};
template <>
struct ext::oneapi::experimental::is_single_task_kernel<__sycl_shim1()> {
  static constexpr bool value = true;
};
}

// Definition of _Z18__sycl_kernel_ff_1Piii as a free function kernel

// Forward declarations of kernel and its argument types:

void ff_1(int * ptr, int start, int end);
static constexpr auto __sycl_shim2() {
  return (void (*)(int *, int, int))ff_1;
}
namespace sycl {
template <>
struct ext::oneapi::experimental::is_kernel<__sycl_shim2()> {
  static constexpr bool value = true;
};
template <>
struct ext::oneapi::experimental::is_nd_range_kernel<__sycl_shim2(), 1> {
  static constexpr bool value = true;
};
}

// Definition of _Z18__sycl_kernel_ff_1Pii as a free function kernel

// Forward declarations of kernel and its argument types:

void ff_1(int * ptr, int start);
static constexpr auto __sycl_shim3() {
  return (void (*)(int *, int))ff_1;
}
namespace sycl {
template <>
struct ext::oneapi::experimental::is_kernel<__sycl_shim3()> {
  static constexpr bool value = true;
};
template <>
struct ext::oneapi::experimental::is_nd_range_kernel<__sycl_shim3(), 2> {
  static constexpr bool value = true;
};
}

// Definition of _Z18__sycl_kernel_ff_3IiEvPT_S0_ as a free function kernel

// Forward declarations of kernel and its argument types:

template <typename T> void ff_3(T * ptr, T start);
static constexpr auto __sycl_shim4() {
  return (void (*)(int *, int))ff_3<int>;
}
namespace sycl {
template <>
struct ext::oneapi::experimental::is_kernel<__sycl_shim4()> {
  static constexpr bool value = true;
};
template <>
struct ext::oneapi::experimental::is_nd_range_kernel<__sycl_shim4(), 2> {
  static constexpr bool value = true;
};
}

#include <sycl/kernel_bundle.hpp>

// Definition of kernel_id of __sycl_kernel_ff_0
namespace sycl {
template <>
kernel_id ext::oneapi::experimental::get_kernel_id<__sycl_shim1()>() {
  return sycl::detail::get_kernel_id_impl(std::string_view{"__sycl_kernel_ff_0"});
}
}

// Definition of kernel_id of _Z18__sycl_kernel_ff_1Piii
namespace sycl {
template <>
kernel_id ext::oneapi::experimental::get_kernel_id<__sycl_shim2()>() {
  return sycl::detail::get_kernel_id_impl(std::string_view{"_Z18__sycl_kernel_ff_1Piii"});
}
}

// Definition of kernel_id of _Z18__sycl_kernel_ff_1Pii
namespace sycl {
template <>
kernel_id ext::oneapi::experimental::get_kernel_id<__sycl_shim3()>() {
  return sycl::detail::get_kernel_id_impl(std::string_view{"_Z18__sycl_kernel_ff_1Pii"});
}
}

// Definition of kernel_id of _Z18__sycl_kernel_ff_3IiEvPT_S0_
namespace sycl {
template <>
kernel_id ext::oneapi::experimental::get_kernel_id<__sycl_shim4()>() {
  return sycl::detail::get_kernel_id_impl(std::string_view{"_Z18__sycl_kernel_ff_3IiEvPT_S0_"});
}
}
