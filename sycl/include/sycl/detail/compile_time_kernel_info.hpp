//==------------------- compile_time_kernel_info.hpp -----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <sycl/detail/kernel_desc.hpp>
#include <sycl/detail/string_view.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {

template <typename KernelNameType>
constexpr kernel_param_desc_t getKernelParamDesc(int Idx) {
#ifndef __INTEL_SYCL_USE_INTEGRATION_HEADERS
  kernel_param_desc_t ParamDesc;
  ParamDesc.kind =
      __builtin_sycl_kernel_param_kind(KernelIdentity<KernelNameType>(), Idx);
  ParamDesc.info = ParamDesc.kind == kernel_param_kind_t::kind_accessor
                       ? __builtin_sycl_kernel_param_access_target(
                             KernelIdentity<KernelNameType>(), Idx)
                       : __builtin_sycl_kernel_param_size(
                             KernelIdentity<KernelNameType>(), Idx);
  ParamDesc.offset =
      __builtin_sycl_kernel_param_offset(KernelIdentity<KernelNameType>(), Idx);
  return ParamDesc;
#else
  return KernelInfo<KernelNameType>::getParamDesc(Idx);
#endif
}

inline namespace compile_time_kernel_info_v1 {
// This is being passed across ABI boundary, so we don't use std::string_view,
// at least for as long as we support user apps built with GNU libstdc++'s
// pre-C++11 ABI.
struct CompileTimeKernelInfoTy {
  detail::string_view Name{};
  unsigned NumParams = 0;
  bool IsESIMD = false;
  // TODO: Can we just have code_location here?
  detail::string_view FileName{};
  detail::string_view FunctionName{};
  unsigned LineNumber = 0;
  unsigned ColumnNumber = 0;
  int64_t KernelSize = 0;
  using ParamDescGetterT = kernel_param_desc_t (*)(int);
  ParamDescGetterT ParamDescGetter = nullptr;

  bool HasSpecialCaptures = [this]() constexpr {
    // No-compile time info for the kernel (i.e., kernel_bundle/interop/etc.),
    // be conservative:
    if (NumParams == 0)
      return true;

    for (unsigned I = 0; I < NumParams; ++I) {
      auto ParamDesc = ParamDescGetter(I);
      if (ParamDesc.kind != kernel_param_kind_t::kind_std_layout &&
          ParamDesc.kind != kernel_param_kind_t::kind_pointer)
        return true;
    }

    return false;
  }();
};

template <class Kernel>
inline constexpr CompileTimeKernelInfoTy CompileTimeKernelInfo{
#ifndef __INTEL_SYCL_USE_INTEGRATION_HEADERS
    __builtin_sycl_kernel_name(KernelIdentity<Kernel>()),
    __builtin_sycl_kernel_param_count(KernelIdentity<Kernel>()),
    false /*IsESIMD*/, // TODO needs a builtin counterpart
    __builtin_sycl_kernel_file_name(KernelIdentity<Kernel>()),
    __builtin_sycl_kernel_function_name(KernelIdentity<Kernel>()),
    __builtin_sycl_kernel_line_number(KernelIdentity<Kernel>()),
    __builtin_sycl_kernel_column_number(KernelIdentity<Kernel>()),
    // TODO needs a builtin counterpart, but is currently only used for checking
    // cases with external host compiler, which use integration headers.
    0 /* KernelSize */, &getKernelParamDesc<Kernel>
#else
    detail::string_view{KernelInfo<Kernel>::getName()},
    KernelInfo<Kernel>::getNumParams(), KernelInfo<Kernel>::isESIMD(),
    detail::string_view{KernelInfo<Kernel>::getFileName()},
    detail::string_view{KernelInfo<Kernel>::getFunctionName()},
    KernelInfo<Kernel>::getLineNumber(), KernelInfo<Kernel>::getColumnNumber(),
    KernelInfo<Kernel>::getKernelSize(),
    // Can't use KernelInfo::getParamDesc due to different return type (const
    // ref vs. by val):
    &getKernelParamDesc<Kernel>
#endif
};
} // namespace compile_time_kernel_info_v1
} // namespace detail
} // namespace _V1
} // namespace sycl
