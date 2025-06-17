//==------------- memory_export.cpp --- SYCL memory export -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/detail/export.hpp> // for __SYCL_EXPORT
#include <sycl/detail/ur.hpp>
#include <sycl/ext/oneapi/memory_export.hpp>

#include <detail/context_impl.hpp>
#include <detail/queue_impl.hpp>

#include <assert.h> // for assert
#include <stddef.h> // for size_t

#ifndef _WIN32
#include <unistd.h> // for close
#else
#include <handleapi.h> // for CloseHandle
#endif                 // _WIN32
namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

__SYCL_EXPORT void *
alloc_exportable_memory(size_t alignment, size_t size,
                        external_mem_handle_type externalMemHandleType,
                        const sycl::device &syclDevice,
                        const sycl::context &syclContext) {
  auto [urDevice, urCtx, Adapter] = get_ur_handles(syclDevice, syclContext);

  void *retDeviceMemory = nullptr;

  ur_exp_external_mem_type_t urExternalMemType;
  switch (externalMemHandleType) {
  case external_mem_handle_type::opaque_fd:
    urExternalMemType = UR_EXP_EXTERNAL_MEM_TYPE_OPAQUE_FD;
    break;
  case external_mem_handle_type::win32_nt_handle:
    urExternalMemType = UR_EXP_EXTERNAL_MEM_TYPE_WIN32_NT;
    break;
  default:
    throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                          "Invalid external memory handle type");
  }

  Adapter
      ->call<sycl::errc::runtime,
             sycl::detail::UrApiKind::urMemoryExportAllocExportableMemoryExp>(
          urCtx, urDevice, alignment, size, urExternalMemType,
          &retDeviceMemory);

  return retDeviceMemory;
}

template <>
__SYCL_EXPORT resource_fd export_memory_handle<resource_fd>(
    void *deviceMemory, const sycl::device &syclDevice,
    const sycl::context &syclContext) {
  auto [urDevice, urCtx, Adapter] = get_ur_handles(syclDevice, syclContext);

  ur_exp_external_mem_type_t urExternalMemType =
      UR_EXP_EXTERNAL_MEM_TYPE_OPAQUE_FD;

  int retFDHandle = 0;

  Adapter->call<sycl::errc::runtime,
                sycl::detail::UrApiKind::urMemoryExportExportMemoryHandleExp>(
      urCtx, urDevice, urExternalMemType, deviceMemory,
      reinterpret_cast<void *>(&retFDHandle));

  return resource_fd{retFDHandle};
}

template <>
__SYCL_EXPORT resource_win32_handle export_memory_handle<resource_win32_handle>(
    void *deviceMemory, const sycl::device &syclDevice,
    const sycl::context &syclContext) {
  auto [urDevice, urCtx, Adapter] = get_ur_handles(syclDevice, syclContext);

  ur_exp_external_mem_type_t urExternalMemType =
      UR_EXP_EXTERNAL_MEM_TYPE_WIN32_NT;

  void *retNTHandle;

  Adapter->call<sycl::errc::runtime,
                sycl::detail::UrApiKind::urMemoryExportExportMemoryHandleExp>(
      urCtx, urDevice, urExternalMemType, deviceMemory,
      static_cast<void *>(&retNTHandle));

  return resource_win32_handle{retNTHandle};
}

__SYCL_EXPORT void free_exportable_memory(void *deviceMemory,
                                          const sycl::device &syclDevice,
                                          const sycl::context &syclContext) {
  auto [urDevice, urCtx, Adapter] = get_ur_handles(syclDevice, syclContext);

  Adapter->call<sycl::errc::runtime,
                sycl::detail::UrApiKind::urMemoryExportFreeExportableMemoryExp>(
      urCtx, urDevice, deviceMemory);

  return;
}

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
