//==------- ipc_memory.hpp -- SYCL inter-process for physical mem ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#if (!defined(_HAS_STD_BYTE) || _HAS_STD_BYTE != 0)

#include <sycl/context.hpp>
#include <sycl/detail/defines_elementary.hpp>
#include <sycl/detail/export.hpp>
#include <sycl/device.hpp>
#include <sycl/ext/oneapi/virtual_mem/physical_mem.hpp>
#include <sycl/platform.hpp>

#include "detail/ipc_common.hpp"

#include <cstddef>

#if __has_include(<span>)
#include <span>
#endif

namespace sycl {
inline namespace _V1 {

namespace detail {
__SYCL_EXPORT sycl::ext::oneapi::experimental::physical_mem
openIPCPhysicalMemHandle(const std::byte *HandleData, size_t HandleDataSize,
                         const sycl::context &Ctx, const sycl::device &Dev);
}

namespace ext::oneapi::experimental::ipc::physical_memory {

__SYCL_EXPORT ipc::handle get(physical_mem &physmem);

__SYCL_EXPORT void put(handle &ipc_handle, const sycl::context &ctx);

inline void put(handle &ipc_handle) {
  sycl::device Dev;
  sycl::context Ctx = Dev.get_platform().khr_get_default_context();
  return put(ipc_handle, Ctx);
}

inline physical_mem open(const ipc::handle_data_t &HandleData,
                         const sycl::context &Ctx, const sycl::device &Dev) {
  return sycl::detail::openIPCPhysicalMemHandle(HandleData.data(),
                                                HandleData.size(), Ctx, Dev);
}

inline physical_mem open(ipc::handle_data_t HandleData,
                         const sycl::device &Dev) {
  sycl::context Ctx = Dev.get_platform().khr_get_default_context();
  return open(HandleData, Ctx, Dev);
}

inline physical_mem open(ipc::handle_data_t HandleData) {
  sycl::device Dev;
  sycl::context Ctx = Dev.get_platform().khr_get_default_context();
  return open(HandleData, Ctx, Dev);
}

#if __cpp_lib_span
inline physical_mem open(const ipc::handle_data_view_t &HandleDataView,
                         const sycl::context &Ctx, const sycl::device &Dev) {
  return sycl::detail::openIPCPhysicalMemHandle(
      HandleDataView.data(), HandleDataView.size(), Ctx, Dev);
}

inline physical_mem open(ipc::handle_data_view_t HandleDataView,
                         const sycl::device &Dev) {
  sycl::context Ctx = Dev.get_platform().khr_get_default_context();
  return open(HandleDataView, Ctx, Dev);
}

inline physical_mem open(ipc::handle_data_view_t HandleDataView) {
  sycl::device Dev;
  sycl::context Ctx = Dev.get_platform().khr_get_default_context();
  return open(HandleDataView, Ctx, Dev);
}
#endif

} // namespace ext::oneapi::experimental::ipc::physical_memory
} // namespace _V1
} // namespace sycl

#endif
