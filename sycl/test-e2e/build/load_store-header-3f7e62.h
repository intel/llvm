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
  "_ZTSZZ7RunTestIN4sycl3_V13vecIiLi4EEEEiRNS1_5queueEENKUlRNS1_7handlerEE_clES7_EUlNS1_7nd_itemILi1EEEE_",
  "_ZTSZZ7RunTestIN4sycl3_V13vecIiLi4EEEEiRNS1_5queueEENKUlRNS1_7handlerEE0_clES7_EUlNS1_7nd_itemILi1EEEE_",
  "_ZTSZZ7RunTestIN4sycl3_V13vecIfLi4EEEEiRNS1_5queueEENKUlRNS1_7handlerEE_clES7_EUlNS1_7nd_itemILi1EEEE_",
  "_ZTSZZ7RunTestIN4sycl3_V13vecIfLi4EEEEiRNS1_5queueEENKUlRNS1_7handlerEE0_clES7_EUlNS1_7nd_itemILi1EEEE_",
  "_ZTSZZ7RunTestIN4sycl3_V13vecINS1_3ext6oneapi8bfloat16ELi4EEEEiRNS1_5queueEENKUlRNS1_7handlerEE_clESA_EUlNS1_7nd_itemILi1EEEE_",
  "_ZTSZZ7RunTestIN4sycl3_V13vecINS1_3ext6oneapi8bfloat16ELi4EEEEiRNS1_5queueEENKUlRNS1_7handlerEE0_clESA_EUlNS1_7nd_itemILi1EEEE_",
  "_ZTSZZ7RunTestIN4sycl3_V13vecINS1_6detail9half_impl4halfELi4EEEEiRNS1_5queueEENKUlRNS1_7handlerEE_clESA_EUlNS1_7nd_itemILi1EEEE_",
  "_ZTSZZ7RunTestIN4sycl3_V13vecINS1_6detail9half_impl4halfELi4EEEEiRNS1_5queueEENKUlRNS1_7handlerEE0_clESA_EUlNS1_7nd_itemILi1EEEE_",
  "_ZTSZZ7RunTestIN4sycl3_V13vecIdLi4EEEEiRNS1_5queueEENKUlRNS1_7handlerEE_clES7_EUlNS1_7nd_itemILi1EEEE_",
  "_ZTSZZ7RunTestIN4sycl3_V13vecIdLi4EEEEiRNS1_5queueEENKUlRNS1_7handlerEE0_clES7_EUlNS1_7nd_itemILi1EEEE_",
  "",
};

// array representing signatures of all kernels defined in the
// corresponding source
static constexpr
const kernel_param_desc_t kernel_signatures[] = {
  //--- _ZTSZZ7RunTestIN4sycl3_V13vecIiLi4EEEEiRNS1_5queueEENKUlRNS1_7handlerEE_clES7_EUlNS1_7nd_itemILi1EEEE_
  { kernel_param_kind_t::kind_std_layout, 96, 0 },
  { kernel_param_kind_t::kind_accessor, 4062, 96 },
  { kernel_param_kind_t::kind_accessor, 4064, 128 },
  { kernel_param_kind_t::kind_accessor, 4062, 160 },

  //--- _ZTSZZ7RunTestIN4sycl3_V13vecIiLi4EEEEiRNS1_5queueEENKUlRNS1_7handlerEE0_clES7_EUlNS1_7nd_itemILi1EEEE_
  { kernel_param_kind_t::kind_std_layout, 96, 0 },
  { kernel_param_kind_t::kind_accessor, 4062, 96 },
  { kernel_param_kind_t::kind_std_layout, 96, 128 },
  { kernel_param_kind_t::kind_accessor, 4064, 224 },

  //--- _ZTSZZ7RunTestIN4sycl3_V13vecIfLi4EEEEiRNS1_5queueEENKUlRNS1_7handlerEE_clES7_EUlNS1_7nd_itemILi1EEEE_
  { kernel_param_kind_t::kind_std_layout, 96, 0 },
  { kernel_param_kind_t::kind_accessor, 4062, 96 },
  { kernel_param_kind_t::kind_accessor, 4064, 128 },
  { kernel_param_kind_t::kind_accessor, 4062, 160 },

  //--- _ZTSZZ7RunTestIN4sycl3_V13vecIfLi4EEEEiRNS1_5queueEENKUlRNS1_7handlerEE0_clES7_EUlNS1_7nd_itemILi1EEEE_
  { kernel_param_kind_t::kind_std_layout, 96, 0 },
  { kernel_param_kind_t::kind_accessor, 4062, 96 },
  { kernel_param_kind_t::kind_std_layout, 96, 128 },
  { kernel_param_kind_t::kind_accessor, 4064, 224 },

  //--- _ZTSZZ7RunTestIN4sycl3_V13vecINS1_3ext6oneapi8bfloat16ELi4EEEEiRNS1_5queueEENKUlRNS1_7handlerEE_clESA_EUlNS1_7nd_itemILi1EEEE_
  { kernel_param_kind_t::kind_std_layout, 48, 0 },
  { kernel_param_kind_t::kind_accessor, 4062, 48 },
  { kernel_param_kind_t::kind_accessor, 4064, 80 },
  { kernel_param_kind_t::kind_accessor, 4062, 112 },

  //--- _ZTSZZ7RunTestIN4sycl3_V13vecINS1_3ext6oneapi8bfloat16ELi4EEEEiRNS1_5queueEENKUlRNS1_7handlerEE0_clESA_EUlNS1_7nd_itemILi1EEEE_
  { kernel_param_kind_t::kind_std_layout, 48, 0 },
  { kernel_param_kind_t::kind_accessor, 4062, 48 },
  { kernel_param_kind_t::kind_std_layout, 48, 80 },
  { kernel_param_kind_t::kind_accessor, 4064, 128 },

  //--- _ZTSZZ7RunTestIN4sycl3_V13vecINS1_6detail9half_impl4halfELi4EEEEiRNS1_5queueEENKUlRNS1_7handlerEE_clESA_EUlNS1_7nd_itemILi1EEEE_
  { kernel_param_kind_t::kind_std_layout, 48, 0 },
  { kernel_param_kind_t::kind_accessor, 4062, 48 },
  { kernel_param_kind_t::kind_accessor, 4064, 80 },
  { kernel_param_kind_t::kind_accessor, 4062, 112 },

  //--- _ZTSZZ7RunTestIN4sycl3_V13vecINS1_6detail9half_impl4halfELi4EEEEiRNS1_5queueEENKUlRNS1_7handlerEE0_clESA_EUlNS1_7nd_itemILi1EEEE_
  { kernel_param_kind_t::kind_std_layout, 48, 0 },
  { kernel_param_kind_t::kind_accessor, 4062, 48 },
  { kernel_param_kind_t::kind_std_layout, 48, 80 },
  { kernel_param_kind_t::kind_accessor, 4064, 128 },

  //--- _ZTSZZ7RunTestIN4sycl3_V13vecIdLi4EEEEiRNS1_5queueEENKUlRNS1_7handlerEE_clES7_EUlNS1_7nd_itemILi1EEEE_
  { kernel_param_kind_t::kind_std_layout, 192, 0 },
  { kernel_param_kind_t::kind_accessor, 4062, 192 },
  { kernel_param_kind_t::kind_accessor, 4064, 224 },
  { kernel_param_kind_t::kind_accessor, 4062, 256 },

  //--- _ZTSZZ7RunTestIN4sycl3_V13vecIdLi4EEEEiRNS1_5queueEENKUlRNS1_7handlerEE0_clES7_EUlNS1_7nd_itemILi1EEEE_
  { kernel_param_kind_t::kind_std_layout, 192, 0 },
  { kernel_param_kind_t::kind_accessor, 4062, 192 },
  { kernel_param_kind_t::kind_std_layout, 192, 224 },
  { kernel_param_kind_t::kind_accessor, 4064, 416 },

  { kernel_param_kind_t::kind_invalid, -987654321, -987654321 }, 
};

// Specializations of KernelInfo for kernel function types:
template <> struct KernelInfoData<'_', 'Z', 'T', 'S', 'Z', 'Z', '7', 'R', 'u', 'n', 'T', 'e', 's', 't', 'I', 'N', '4', 's', 'y', 'c', 'l', '3', '_', 'V', '1', '3', 'v', 'e', 'c', 'I', 'i', 'L', 'i', '4', 'E', 'E', 'E', 'E', 'i', 'R', 'N', 'S', '1', '_', '5', 'q', 'u', 'e', 'u', 'e', 'E', 'E', 'N', 'K', 'U', 'l', 'R', 'N', 'S', '1', '_', '7', 'h', 'a', 'n', 'd', 'l', 'e', 'r', 'E', 'E', '_', 'c', 'l', 'E', 'S', '7', '_', 'E', 'U', 'l', 'N', 'S', '1', '_', '7', 'n', 'd', '_', 'i', 't', 'e', 'm', 'I', 'L', 'i', '1', 'E', 'E', 'E', 'E', '_'> {
  __SYCL_DLL_LOCAL
  static constexpr const char* getName() { return "_ZTSZZ7RunTestIN4sycl3_V13vecIiLi4EEEEiRNS1_5queueEENKUlRNS1_7handlerEE_clES7_EUlNS1_7nd_itemILi1EEEE_"; }
  __SYCL_DLL_LOCAL
  static constexpr unsigned getNumParams() { return 4; }
  __SYCL_DLL_LOCAL
  static constexpr const kernel_param_desc_t& getParamDesc(unsigned i) {
    return kernel_signatures[i+0];
  }
  __SYCL_DLL_LOCAL
  static constexpr bool isESIMD() { return 0; }
  __SYCL_DLL_LOCAL
  static constexpr const char* getFileName() {
#ifndef NDEBUG
    return "load_store.cpp";
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
    return 68;
#else
    return 0;
#endif
  }
  __SYCL_DLL_LOCAL
  static constexpr unsigned getColumnNumber() {
#ifndef NDEBUG
    return 51;
#else
    return 0;
#endif
  }
  // Returns the size of the kernel object in bytes.
  __SYCL_DLL_LOCAL
  static constexpr long getKernelSize() { return 192; }
};
template <> struct KernelInfoData<'_', 'Z', 'T', 'S', 'Z', 'Z', '7', 'R', 'u', 'n', 'T', 'e', 's', 't', 'I', 'N', '4', 's', 'y', 'c', 'l', '3', '_', 'V', '1', '3', 'v', 'e', 'c', 'I', 'i', 'L', 'i', '4', 'E', 'E', 'E', 'E', 'i', 'R', 'N', 'S', '1', '_', '5', 'q', 'u', 'e', 'u', 'e', 'E', 'E', 'N', 'K', 'U', 'l', 'R', 'N', 'S', '1', '_', '7', 'h', 'a', 'n', 'd', 'l', 'e', 'r', 'E', 'E', '0', '_', 'c', 'l', 'E', 'S', '7', '_', 'E', 'U', 'l', 'N', 'S', '1', '_', '7', 'n', 'd', '_', 'i', 't', 'e', 'm', 'I', 'L', 'i', '1', 'E', 'E', 'E', 'E', '_'> {
  __SYCL_DLL_LOCAL
  static constexpr const char* getName() { return "_ZTSZZ7RunTestIN4sycl3_V13vecIiLi4EEEEiRNS1_5queueEENKUlRNS1_7handlerEE0_clES7_EUlNS1_7nd_itemILi1EEEE_"; }
  __SYCL_DLL_LOCAL
  static constexpr unsigned getNumParams() { return 4; }
  __SYCL_DLL_LOCAL
  static constexpr const kernel_param_desc_t& getParamDesc(unsigned i) {
    return kernel_signatures[i+4];
  }
  __SYCL_DLL_LOCAL
  static constexpr bool isESIMD() { return 0; }
  __SYCL_DLL_LOCAL
  static constexpr const char* getFileName() {
#ifndef NDEBUG
    return "load_store.cpp";
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
    return 133;
#else
    return 0;
#endif
  }
  __SYCL_DLL_LOCAL
  static constexpr unsigned getColumnNumber() {
#ifndef NDEBUG
    return 51;
#else
    return 0;
#endif
  }
  // Returns the size of the kernel object in bytes.
  __SYCL_DLL_LOCAL
  static constexpr long getKernelSize() { return 256; }
};
template <> struct KernelInfoData<'_', 'Z', 'T', 'S', 'Z', 'Z', '7', 'R', 'u', 'n', 'T', 'e', 's', 't', 'I', 'N', '4', 's', 'y', 'c', 'l', '3', '_', 'V', '1', '3', 'v', 'e', 'c', 'I', 'f', 'L', 'i', '4', 'E', 'E', 'E', 'E', 'i', 'R', 'N', 'S', '1', '_', '5', 'q', 'u', 'e', 'u', 'e', 'E', 'E', 'N', 'K', 'U', 'l', 'R', 'N', 'S', '1', '_', '7', 'h', 'a', 'n', 'd', 'l', 'e', 'r', 'E', 'E', '_', 'c', 'l', 'E', 'S', '7', '_', 'E', 'U', 'l', 'N', 'S', '1', '_', '7', 'n', 'd', '_', 'i', 't', 'e', 'm', 'I', 'L', 'i', '1', 'E', 'E', 'E', 'E', '_'> {
  __SYCL_DLL_LOCAL
  static constexpr const char* getName() { return "_ZTSZZ7RunTestIN4sycl3_V13vecIfLi4EEEEiRNS1_5queueEENKUlRNS1_7handlerEE_clES7_EUlNS1_7nd_itemILi1EEEE_"; }
  __SYCL_DLL_LOCAL
  static constexpr unsigned getNumParams() { return 4; }
  __SYCL_DLL_LOCAL
  static constexpr const kernel_param_desc_t& getParamDesc(unsigned i) {
    return kernel_signatures[i+8];
  }
  __SYCL_DLL_LOCAL
  static constexpr bool isESIMD() { return 0; }
  __SYCL_DLL_LOCAL
  static constexpr const char* getFileName() {
#ifndef NDEBUG
    return "load_store.cpp";
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
    return 68;
#else
    return 0;
#endif
  }
  __SYCL_DLL_LOCAL
  static constexpr unsigned getColumnNumber() {
#ifndef NDEBUG
    return 51;
#else
    return 0;
#endif
  }
  // Returns the size of the kernel object in bytes.
  __SYCL_DLL_LOCAL
  static constexpr long getKernelSize() { return 192; }
};
template <> struct KernelInfoData<'_', 'Z', 'T', 'S', 'Z', 'Z', '7', 'R', 'u', 'n', 'T', 'e', 's', 't', 'I', 'N', '4', 's', 'y', 'c', 'l', '3', '_', 'V', '1', '3', 'v', 'e', 'c', 'I', 'f', 'L', 'i', '4', 'E', 'E', 'E', 'E', 'i', 'R', 'N', 'S', '1', '_', '5', 'q', 'u', 'e', 'u', 'e', 'E', 'E', 'N', 'K', 'U', 'l', 'R', 'N', 'S', '1', '_', '7', 'h', 'a', 'n', 'd', 'l', 'e', 'r', 'E', 'E', '0', '_', 'c', 'l', 'E', 'S', '7', '_', 'E', 'U', 'l', 'N', 'S', '1', '_', '7', 'n', 'd', '_', 'i', 't', 'e', 'm', 'I', 'L', 'i', '1', 'E', 'E', 'E', 'E', '_'> {
  __SYCL_DLL_LOCAL
  static constexpr const char* getName() { return "_ZTSZZ7RunTestIN4sycl3_V13vecIfLi4EEEEiRNS1_5queueEENKUlRNS1_7handlerEE0_clES7_EUlNS1_7nd_itemILi1EEEE_"; }
  __SYCL_DLL_LOCAL
  static constexpr unsigned getNumParams() { return 4; }
  __SYCL_DLL_LOCAL
  static constexpr const kernel_param_desc_t& getParamDesc(unsigned i) {
    return kernel_signatures[i+12];
  }
  __SYCL_DLL_LOCAL
  static constexpr bool isESIMD() { return 0; }
  __SYCL_DLL_LOCAL
  static constexpr const char* getFileName() {
#ifndef NDEBUG
    return "load_store.cpp";
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
    return 133;
#else
    return 0;
#endif
  }
  __SYCL_DLL_LOCAL
  static constexpr unsigned getColumnNumber() {
#ifndef NDEBUG
    return 51;
#else
    return 0;
#endif
  }
  // Returns the size of the kernel object in bytes.
  __SYCL_DLL_LOCAL
  static constexpr long getKernelSize() { return 256; }
};
template <> struct KernelInfoData<'_', 'Z', 'T', 'S', 'Z', 'Z', '7', 'R', 'u', 'n', 'T', 'e', 's', 't', 'I', 'N', '4', 's', 'y', 'c', 'l', '3', '_', 'V', '1', '3', 'v', 'e', 'c', 'I', 'N', 'S', '1', '_', '3', 'e', 'x', 't', '6', 'o', 'n', 'e', 'a', 'p', 'i', '8', 'b', 'f', 'l', 'o', 'a', 't', '1', '6', 'E', 'L', 'i', '4', 'E', 'E', 'E', 'E', 'i', 'R', 'N', 'S', '1', '_', '5', 'q', 'u', 'e', 'u', 'e', 'E', 'E', 'N', 'K', 'U', 'l', 'R', 'N', 'S', '1', '_', '7', 'h', 'a', 'n', 'd', 'l', 'e', 'r', 'E', 'E', '_', 'c', 'l', 'E', 'S', 'A', '_', 'E', 'U', 'l', 'N', 'S', '1', '_', '7', 'n', 'd', '_', 'i', 't', 'e', 'm', 'I', 'L', 'i', '1', 'E', 'E', 'E', 'E', '_'> {
  __SYCL_DLL_LOCAL
  static constexpr const char* getName() { return "_ZTSZZ7RunTestIN4sycl3_V13vecINS1_3ext6oneapi8bfloat16ELi4EEEEiRNS1_5queueEENKUlRNS1_7handlerEE_clESA_EUlNS1_7nd_itemILi1EEEE_"; }
  __SYCL_DLL_LOCAL
  static constexpr unsigned getNumParams() { return 4; }
  __SYCL_DLL_LOCAL
  static constexpr const kernel_param_desc_t& getParamDesc(unsigned i) {
    return kernel_signatures[i+16];
  }
  __SYCL_DLL_LOCAL
  static constexpr bool isESIMD() { return 0; }
  __SYCL_DLL_LOCAL
  static constexpr const char* getFileName() {
#ifndef NDEBUG
    return "load_store.cpp";
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
    return 68;
#else
    return 0;
#endif
  }
  __SYCL_DLL_LOCAL
  static constexpr unsigned getColumnNumber() {
#ifndef NDEBUG
    return 51;
#else
    return 0;
#endif
  }
  // Returns the size of the kernel object in bytes.
  __SYCL_DLL_LOCAL
  static constexpr long getKernelSize() { return 144; }
};
template <> struct KernelInfoData<'_', 'Z', 'T', 'S', 'Z', 'Z', '7', 'R', 'u', 'n', 'T', 'e', 's', 't', 'I', 'N', '4', 's', 'y', 'c', 'l', '3', '_', 'V', '1', '3', 'v', 'e', 'c', 'I', 'N', 'S', '1', '_', '3', 'e', 'x', 't', '6', 'o', 'n', 'e', 'a', 'p', 'i', '8', 'b', 'f', 'l', 'o', 'a', 't', '1', '6', 'E', 'L', 'i', '4', 'E', 'E', 'E', 'E', 'i', 'R', 'N', 'S', '1', '_', '5', 'q', 'u', 'e', 'u', 'e', 'E', 'E', 'N', 'K', 'U', 'l', 'R', 'N', 'S', '1', '_', '7', 'h', 'a', 'n', 'd', 'l', 'e', 'r', 'E', 'E', '0', '_', 'c', 'l', 'E', 'S', 'A', '_', 'E', 'U', 'l', 'N', 'S', '1', '_', '7', 'n', 'd', '_', 'i', 't', 'e', 'm', 'I', 'L', 'i', '1', 'E', 'E', 'E', 'E', '_'> {
  __SYCL_DLL_LOCAL
  static constexpr const char* getName() { return "_ZTSZZ7RunTestIN4sycl3_V13vecINS1_3ext6oneapi8bfloat16ELi4EEEEiRNS1_5queueEENKUlRNS1_7handlerEE0_clESA_EUlNS1_7nd_itemILi1EEEE_"; }
  __SYCL_DLL_LOCAL
  static constexpr unsigned getNumParams() { return 4; }
  __SYCL_DLL_LOCAL
  static constexpr const kernel_param_desc_t& getParamDesc(unsigned i) {
    return kernel_signatures[i+20];
  }
  __SYCL_DLL_LOCAL
  static constexpr bool isESIMD() { return 0; }
  __SYCL_DLL_LOCAL
  static constexpr const char* getFileName() {
#ifndef NDEBUG
    return "load_store.cpp";
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
    return 133;
#else
    return 0;
#endif
  }
  __SYCL_DLL_LOCAL
  static constexpr unsigned getColumnNumber() {
#ifndef NDEBUG
    return 51;
#else
    return 0;
#endif
  }
  // Returns the size of the kernel object in bytes.
  __SYCL_DLL_LOCAL
  static constexpr long getKernelSize() { return 160; }
};
template <> struct KernelInfoData<'_', 'Z', 'T', 'S', 'Z', 'Z', '7', 'R', 'u', 'n', 'T', 'e', 's', 't', 'I', 'N', '4', 's', 'y', 'c', 'l', '3', '_', 'V', '1', '3', 'v', 'e', 'c', 'I', 'N', 'S', '1', '_', '6', 'd', 'e', 't', 'a', 'i', 'l', '9', 'h', 'a', 'l', 'f', '_', 'i', 'm', 'p', 'l', '4', 'h', 'a', 'l', 'f', 'E', 'L', 'i', '4', 'E', 'E', 'E', 'E', 'i', 'R', 'N', 'S', '1', '_', '5', 'q', 'u', 'e', 'u', 'e', 'E', 'E', 'N', 'K', 'U', 'l', 'R', 'N', 'S', '1', '_', '7', 'h', 'a', 'n', 'd', 'l', 'e', 'r', 'E', 'E', '_', 'c', 'l', 'E', 'S', 'A', '_', 'E', 'U', 'l', 'N', 'S', '1', '_', '7', 'n', 'd', '_', 'i', 't', 'e', 'm', 'I', 'L', 'i', '1', 'E', 'E', 'E', 'E', '_'> {
  __SYCL_DLL_LOCAL
  static constexpr const char* getName() { return "_ZTSZZ7RunTestIN4sycl3_V13vecINS1_6detail9half_impl4halfELi4EEEEiRNS1_5queueEENKUlRNS1_7handlerEE_clESA_EUlNS1_7nd_itemILi1EEEE_"; }
  __SYCL_DLL_LOCAL
  static constexpr unsigned getNumParams() { return 4; }
  __SYCL_DLL_LOCAL
  static constexpr const kernel_param_desc_t& getParamDesc(unsigned i) {
    return kernel_signatures[i+24];
  }
  __SYCL_DLL_LOCAL
  static constexpr bool isESIMD() { return 0; }
  __SYCL_DLL_LOCAL
  static constexpr const char* getFileName() {
#ifndef NDEBUG
    return "load_store.cpp";
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
    return 68;
#else
    return 0;
#endif
  }
  __SYCL_DLL_LOCAL
  static constexpr unsigned getColumnNumber() {
#ifndef NDEBUG
    return 51;
#else
    return 0;
#endif
  }
  // Returns the size of the kernel object in bytes.
  __SYCL_DLL_LOCAL
  static constexpr long getKernelSize() { return 144; }
};
template <> struct KernelInfoData<'_', 'Z', 'T', 'S', 'Z', 'Z', '7', 'R', 'u', 'n', 'T', 'e', 's', 't', 'I', 'N', '4', 's', 'y', 'c', 'l', '3', '_', 'V', '1', '3', 'v', 'e', 'c', 'I', 'N', 'S', '1', '_', '6', 'd', 'e', 't', 'a', 'i', 'l', '9', 'h', 'a', 'l', 'f', '_', 'i', 'm', 'p', 'l', '4', 'h', 'a', 'l', 'f', 'E', 'L', 'i', '4', 'E', 'E', 'E', 'E', 'i', 'R', 'N', 'S', '1', '_', '5', 'q', 'u', 'e', 'u', 'e', 'E', 'E', 'N', 'K', 'U', 'l', 'R', 'N', 'S', '1', '_', '7', 'h', 'a', 'n', 'd', 'l', 'e', 'r', 'E', 'E', '0', '_', 'c', 'l', 'E', 'S', 'A', '_', 'E', 'U', 'l', 'N', 'S', '1', '_', '7', 'n', 'd', '_', 'i', 't', 'e', 'm', 'I', 'L', 'i', '1', 'E', 'E', 'E', 'E', '_'> {
  __SYCL_DLL_LOCAL
  static constexpr const char* getName() { return "_ZTSZZ7RunTestIN4sycl3_V13vecINS1_6detail9half_impl4halfELi4EEEEiRNS1_5queueEENKUlRNS1_7handlerEE0_clESA_EUlNS1_7nd_itemILi1EEEE_"; }
  __SYCL_DLL_LOCAL
  static constexpr unsigned getNumParams() { return 4; }
  __SYCL_DLL_LOCAL
  static constexpr const kernel_param_desc_t& getParamDesc(unsigned i) {
    return kernel_signatures[i+28];
  }
  __SYCL_DLL_LOCAL
  static constexpr bool isESIMD() { return 0; }
  __SYCL_DLL_LOCAL
  static constexpr const char* getFileName() {
#ifndef NDEBUG
    return "load_store.cpp";
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
    return 133;
#else
    return 0;
#endif
  }
  __SYCL_DLL_LOCAL
  static constexpr unsigned getColumnNumber() {
#ifndef NDEBUG
    return 51;
#else
    return 0;
#endif
  }
  // Returns the size of the kernel object in bytes.
  __SYCL_DLL_LOCAL
  static constexpr long getKernelSize() { return 160; }
};
template <> struct KernelInfoData<'_', 'Z', 'T', 'S', 'Z', 'Z', '7', 'R', 'u', 'n', 'T', 'e', 's', 't', 'I', 'N', '4', 's', 'y', 'c', 'l', '3', '_', 'V', '1', '3', 'v', 'e', 'c', 'I', 'd', 'L', 'i', '4', 'E', 'E', 'E', 'E', 'i', 'R', 'N', 'S', '1', '_', '5', 'q', 'u', 'e', 'u', 'e', 'E', 'E', 'N', 'K', 'U', 'l', 'R', 'N', 'S', '1', '_', '7', 'h', 'a', 'n', 'd', 'l', 'e', 'r', 'E', 'E', '_', 'c', 'l', 'E', 'S', '7', '_', 'E', 'U', 'l', 'N', 'S', '1', '_', '7', 'n', 'd', '_', 'i', 't', 'e', 'm', 'I', 'L', 'i', '1', 'E', 'E', 'E', 'E', '_'> {
  __SYCL_DLL_LOCAL
  static constexpr const char* getName() { return "_ZTSZZ7RunTestIN4sycl3_V13vecIdLi4EEEEiRNS1_5queueEENKUlRNS1_7handlerEE_clES7_EUlNS1_7nd_itemILi1EEEE_"; }
  __SYCL_DLL_LOCAL
  static constexpr unsigned getNumParams() { return 4; }
  __SYCL_DLL_LOCAL
  static constexpr const kernel_param_desc_t& getParamDesc(unsigned i) {
    return kernel_signatures[i+32];
  }
  __SYCL_DLL_LOCAL
  static constexpr bool isESIMD() { return 0; }
  __SYCL_DLL_LOCAL
  static constexpr const char* getFileName() {
#ifndef NDEBUG
    return "load_store.cpp";
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
    return 68;
#else
    return 0;
#endif
  }
  __SYCL_DLL_LOCAL
  static constexpr unsigned getColumnNumber() {
#ifndef NDEBUG
    return 51;
#else
    return 0;
#endif
  }
  // Returns the size of the kernel object in bytes.
  __SYCL_DLL_LOCAL
  static constexpr long getKernelSize() { return 288; }
};
template <> struct KernelInfoData<'_', 'Z', 'T', 'S', 'Z', 'Z', '7', 'R', 'u', 'n', 'T', 'e', 's', 't', 'I', 'N', '4', 's', 'y', 'c', 'l', '3', '_', 'V', '1', '3', 'v', 'e', 'c', 'I', 'd', 'L', 'i', '4', 'E', 'E', 'E', 'E', 'i', 'R', 'N', 'S', '1', '_', '5', 'q', 'u', 'e', 'u', 'e', 'E', 'E', 'N', 'K', 'U', 'l', 'R', 'N', 'S', '1', '_', '7', 'h', 'a', 'n', 'd', 'l', 'e', 'r', 'E', 'E', '0', '_', 'c', 'l', 'E', 'S', '7', '_', 'E', 'U', 'l', 'N', 'S', '1', '_', '7', 'n', 'd', '_', 'i', 't', 'e', 'm', 'I', 'L', 'i', '1', 'E', 'E', 'E', 'E', '_'> {
  __SYCL_DLL_LOCAL
  static constexpr const char* getName() { return "_ZTSZZ7RunTestIN4sycl3_V13vecIdLi4EEEEiRNS1_5queueEENKUlRNS1_7handlerEE0_clES7_EUlNS1_7nd_itemILi1EEEE_"; }
  __SYCL_DLL_LOCAL
  static constexpr unsigned getNumParams() { return 4; }
  __SYCL_DLL_LOCAL
  static constexpr const kernel_param_desc_t& getParamDesc(unsigned i) {
    return kernel_signatures[i+36];
  }
  __SYCL_DLL_LOCAL
  static constexpr bool isESIMD() { return 0; }
  __SYCL_DLL_LOCAL
  static constexpr const char* getFileName() {
#ifndef NDEBUG
    return "load_store.cpp";
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
    return 133;
#else
    return 0;
#endif
  }
  __SYCL_DLL_LOCAL
  static constexpr unsigned getColumnNumber() {
#ifndef NDEBUG
    return 51;
#else
    return 0;
#endif
  }
  // Returns the size of the kernel object in bytes.
  __SYCL_DLL_LOCAL
  static constexpr long getKernelSize() { return 448; }
};

} // namespace detail
} // namespace _V1
} // namespace sycl
