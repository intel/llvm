//==-------------------- kernel_global_info.cpp ----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/program_manager/program_manager.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail::free_function_info_map {

__SYCL_EXPORT void add(const char *UniqueId, const void *KernelGlobalPtr) {
  detail::ProgramManager::getInstance().registerKernelGlobalInfo(
      UniqueId, KernelGlobalPtr);
}

} // namespace detail::free_function_info_map
} // namespace _V1
} // namespace sycl
