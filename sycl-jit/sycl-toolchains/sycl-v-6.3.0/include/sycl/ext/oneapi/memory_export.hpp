//==------------- memory_export.hpp --- SYCL memory export -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "interop_common.hpp"     // For external_mem_handle_type.
#include <sycl/context.hpp>       // For context.
#include <sycl/detail/export.hpp> // For __SYCL_EXPORT.
#include <sycl/device.hpp>        // For device.
#include <sycl/queue.hpp>         // For queue.

#include <cstddef> // For size_t.

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

namespace detail {

template <external_mem_handle_type ExternalMemHandleType> struct exported_mem;

template <> struct exported_mem<external_mem_handle_type::opaque_fd> {
  using type = int;
};

template <> struct exported_mem<external_mem_handle_type::win32_nt_handle> {
  using type = void *;
};

template <external_mem_handle_type ExternalMemHandleType>
using exported_mem_t = typename exported_mem<ExternalMemHandleType>::type;

__SYCL_EXPORT int export_device_mem_opaque_fd(void *DeviceMemory,
                                              const sycl::device &SyclDevice,
                                              const sycl::context &SyclContext);

__SYCL_EXPORT void *
export_device_mem_win32_nt_handle(void *DeviceMemory,
                                  const sycl::device &SyclDevice,
                                  const sycl::context &SyclContext);
} // namespace detail

__SYCL_EXPORT void *alloc_exportable_device_mem(
    size_t Alignment, size_t Size,
    external_mem_handle_type ExternalMemHandleType,
    const sycl::device &SyclDevice, const sycl::context &SyclContext,
    [[maybe_unused]] const sycl::property_list &PropList = {});

inline void *alloc_exportable_device_mem(
    size_t Alignment, size_t Size,
    external_mem_handle_type ExternalMemHandleType,
    const sycl::queue &SyclQueue,
    [[maybe_unused]] const sycl::property_list &PropList = {}) {
  return alloc_exportable_device_mem(Alignment, Size, ExternalMemHandleType,
                                     SyclQueue.get_device(),
                                     SyclQueue.get_context(), PropList);
}

__SYCL_EXPORT void free_exportable_memory(void *DeviceMemory,
                                          const sycl::device &SyclDevice,
                                          const sycl::context &SyclContext);

inline void free_exportable_memory(void *DeviceMemory,
                                   const sycl::queue &SyclQueue) {
  free_exportable_memory(DeviceMemory, SyclQueue.get_device(),
                         SyclQueue.get_context());
}

// Available only when
// ExternalMemHandleType == external_mem_handle_type::opaque_fd.
template <external_mem_handle_type ExternalMemHandleType,
          std::enable_if_t<ExternalMemHandleType ==
                               external_mem_handle_type::opaque_fd,
                           bool> = true>
inline detail::exported_mem_t<ExternalMemHandleType>
export_device_mem_handle(void *DeviceMemory, const sycl::device &SyclDevice,
                         const sycl::context &SyclContext) {
  return detail::export_device_mem_opaque_fd(DeviceMemory, SyclDevice,
                                             SyclContext);
}

// Available only when
// ExternalMemHandleType == external_mem_handle_type::opaque_fd.
template <external_mem_handle_type ExternalMemHandleType,
          std::enable_if_t<ExternalMemHandleType ==
                               external_mem_handle_type::opaque_fd,
                           bool> = true>
inline detail::exported_mem_t<ExternalMemHandleType>
export_device_mem_handle(void *DeviceMemory, const sycl::queue &SyclQueue) {
  return export_device_mem_handle<ExternalMemHandleType>(
      DeviceMemory, SyclQueue.get_device(), SyclQueue.get_context());
}

// Available only when
// ExternalMemHandleType == external_mem_handle_type::win32_nt_handle.
template <external_mem_handle_type ExternalMemHandleType,
          std::enable_if_t<ExternalMemHandleType ==
                               external_mem_handle_type::win32_nt_handle,
                           bool> = true>
inline detail::exported_mem_t<ExternalMemHandleType>
export_device_mem_handle(void *DeviceMemory, const sycl::device &SyclDevice,
                         const sycl::context &SyclContext) {
  return detail::export_device_mem_win32_nt_handle(DeviceMemory, SyclDevice,
                                                   SyclContext);
}

// Available only when
// ExternalMemHandleType == external_mem_handle_type::win32_nt_handle.
template <external_mem_handle_type ExternalMemHandleType,
          std::enable_if_t<ExternalMemHandleType ==
                               external_mem_handle_type::win32_nt_handle,
                           bool> = true>
inline detail::exported_mem_t<ExternalMemHandleType>
export_device_mem_handle(void *DeviceMemory, const sycl::queue &SyclQueue) {
  return export_device_mem_handle<ExternalMemHandleType>(
      DeviceMemory, SyclQueue.get_device(), SyclQueue.get_context());
}

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
