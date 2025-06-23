//==------------- memory_export.hpp --- SYCL memory export -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cstddef> // for size_t

#include <sycl/context.hpp>       // for context
#include <sycl/detail/export.hpp> // for __SYCL_EXPORT
#include <sycl/device.hpp>        // for device
#include <sycl/queue.hpp>         // for queue

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

// Types of external memory handles
enum class export_external_mem_handle_type {
  opaque_fd = 0,
  win32_nt = 1,
};

namespace detail {

__SYCL_EXPORT int export_device_mem_opaque_fd(void *deviceMemory,
                                              const sycl::device &syclDevice,
                                              const sycl::context &syclContext);

__SYCL_EXPORT void *
export_device_mem_win32_nt(void *deviceMemory, const sycl::device &syclDevice,
                           const sycl::context &syclContext);
} // namespace detail

__SYCL_EXPORT void *alloc_exportable_device_mem(
    size_t alignment, size_t size,
    export_external_mem_handle_type externalMemHandleType,
    const sycl::device &syclDevice, const sycl::context &syclContext);

inline void *alloc_exportable_device_mem(
    size_t alignment, size_t size,
    export_external_mem_handle_type externalMemHandleType,
    const sycl::queue &syclQueue) {
  return alloc_exportable_device_mem(size, alignment, externalMemHandleType,
                                     syclQueue.get_device(),
                                     syclQueue.get_context());
}

__SYCL_EXPORT void free_exportable_memory(void *deviceMemory,
                                          const sycl::device &syclDevice,
                                          const sycl::context &syclContext);

inline void free_exportable_memory(void *deviceMemory,
                                   const sycl::queue &syclQueue) {
  free_exportable_memory(deviceMemory, syclQueue.get_device(),
                         syclQueue.get_context());
}

// Available only when
// ExternalMemType == export_external_mem_handle_type::opaque_fd
template <export_external_mem_handle_type ExternalMemType,
          std::enable_if_t<ExternalMemType ==
                               export_external_mem_handle_type::opaque_fd,
                           bool> = true>
__SYCL_EXPORT int export_device_mem_handle(void *deviceMemory,
                                           const sycl::device &syclDevice,
                                           const sycl::context &syclContext) {
  return detail::export_device_mem_opaque_fd(deviceMemory, syclDevice,
                                             syclContext);
}

// Available only when
// ExternalMemType == export_external_mem_handle_type::opaque_fd
template <export_external_mem_handle_type ExternalMemType,
          std::enable_if_t<ExternalMemType ==
                               export_external_mem_handle_type::opaque_fd,
                           bool> = true>
inline int export_device_mem_handle(void *deviceMemory,
                                    const sycl::queue &syclQueue) {
  return export_device_mem_handle<ExternalMemType>(
      deviceMemory, syclQueue.get_device(), syclQueue.get_context());
}

// Available only when
// ExternalMemType == export_external_mem_handle_type::win32_nt
template <export_external_mem_handle_type ExternalMemType,
          std::enable_if_t<ExternalMemType ==
                               export_external_mem_handle_type::win32_nt,
                           bool> = true>
__SYCL_EXPORT void *export_device_mem_handle(void *deviceMemory,
                                             const sycl::device &syclDevice,
                                             const sycl::context &syclContext) {
  return detail::export_device_mem_win32_nt(deviceMemory, syclDevice,
                                            syclContext);
}

// Available only when
// ExternalMemType == export_external_mem_handle_type::win32_nt
template <export_external_mem_handle_type ExternalMemType,
          std::enable_if_t<ExternalMemType ==
                               export_external_mem_handle_type::win32_nt,
                           bool> = true>
inline void *export_device_mem_handle(void *deviceMemory,
                                      const sycl::queue &syclQueue) {
  return export_device_mem_handle<ExternalMemType>(
      deviceMemory, syclQueue.get_device(), syclQueue.get_context());
}

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
