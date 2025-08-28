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
inline namespace compile_time_kernel_info_v1 {

// This is being passed across ABI boundary, so we don't use std::string_view,
// at least for as long as we support user apps built with GNU libstdc++'s
// pre-C++11 ABI.
struct CompileTimeKernelInfoTy {
  detail::string_view Name;
  unsigned NumParams = 0;
  bool IsESIMD = false;
  detail::string_view FileName{};
  detail::string_view FunctionName{};
  unsigned LineNumber = 0;
  unsigned ColumnNumber = 0;
  int64_t KernelSize = 0;
  using ParamDescGetterT = kernel_param_desc_t (*)(int);
  ParamDescGetterT ParamDescGetter = nullptr;
  bool HasSpecialCaptures = true;
};

template <class Kernel>
inline constexpr CompileTimeKernelInfoTy CompileTimeKernelInfo{
    std::string_view(getKernelName<Kernel>()),
    getKernelNumParams<Kernel>(),
    isKernelESIMD<Kernel>(),
    std::string_view(getKernelFileName<Kernel>()),
    std::string_view(getKernelFunctionName<Kernel>()),
    getKernelLineNumber<Kernel>(),
    getKernelColumnNumber<Kernel>(),
    getKernelSize<Kernel>(),
    &getKernelParamDesc<Kernel>,
    hasSpecialCaptures<Kernel>()};

} // namespace compile_time_kernel_info_v1
} // namespace detail
} // namespace _V1
} // namespace sycl
