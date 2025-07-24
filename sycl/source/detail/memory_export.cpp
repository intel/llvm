//==------------- memory_export.cpp --- SYCL memory export -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/detail/export.hpp> // For __SYCL_EXPORT.
#include <sycl/detail/ur.hpp>
#include <sycl/ext/oneapi/memory_export.hpp>

#include <detail/context_impl.hpp>
#include <detail/queue_impl.hpp>

#include <cassert> // For assert.
#include <cstddef> // For size_t.

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

__SYCL_EXPORT void *alloc_exportable_device_mem(
    size_t Alignment, size_t Size,
    external_mem_handle_type ExternalMemHandleType,
    const sycl::device &SyclDevice, const sycl::context &SyclContext,
    [[maybe_unused]] const sycl::property_list &PropList) {

  if (!SyclDevice.has(sycl::aspect::ext_oneapi_exportable_device_mem)) {
    throw sycl::exception(
        sycl::make_error_code(sycl::errc::feature_not_supported),
        "Device does not support memory export");
  }

  auto [UrDevice, UrCtx, Adapter] = get_ur_handles(SyclDevice, SyclContext);

  void *RetDeviceMemory = nullptr;

  ur_exp_external_mem_type_t UrExternalMemType;
  switch (ExternalMemHandleType) {
  case external_mem_handle_type::opaque_fd:
    UrExternalMemType = UR_EXP_EXTERNAL_MEM_TYPE_OPAQUE_FD;
    break;
  case external_mem_handle_type::win32_nt_handle:
    UrExternalMemType = UR_EXP_EXTERNAL_MEM_TYPE_WIN32_NT;
    break;
  default:
    throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                          "Invalid external memory handle type");
  }

  Adapter
      ->call<sycl::errc::runtime,
             sycl::detail::UrApiKind::urMemoryExportAllocExportableMemoryExp>(
          UrCtx, UrDevice, Alignment, Size, UrExternalMemType,
          &RetDeviceMemory);

  return RetDeviceMemory;
}

__SYCL_EXPORT void free_exportable_memory(void *DeviceMemory,
                                          const sycl::device &SyclDevice,
                                          const sycl::context &SyclContext) {
  auto [UrDevice, UrCtx, Adapter] = get_ur_handles(SyclDevice, SyclContext);

  Adapter->call<sycl::errc::runtime,
                sycl::detail::UrApiKind::urMemoryExportFreeExportableMemoryExp>(
      UrCtx, UrDevice, DeviceMemory);

  return;
}

namespace detail {

__SYCL_EXPORT int
export_device_mem_opaque_fd(void *DeviceMemory, const sycl::device &SyclDevice,
                            const sycl::context &SyclContext) {
  auto [UrDevice, UrCtx, Adapter] = get_ur_handles(SyclDevice, SyclContext);

  ur_exp_external_mem_type_t UrExternalMemType =
      UR_EXP_EXTERNAL_MEM_TYPE_OPAQUE_FD;

  int RetFDHandle = 0;

  Adapter->call<sycl::errc::runtime,
                sycl::detail::UrApiKind::urMemoryExportExportMemoryHandleExp>(
      UrCtx, UrDevice, UrExternalMemType, DeviceMemory,
      reinterpret_cast<void *>(&RetFDHandle));

  return RetFDHandle;
}

__SYCL_EXPORT void *
export_device_mem_win32_nt_handle(void *DeviceMemory,
                                  const sycl::device &SyclDevice,
                                  const sycl::context &SyclContext) {
  auto [UrDevice, UrCtx, Adapter] = get_ur_handles(SyclDevice, SyclContext);

  ur_exp_external_mem_type_t UrExternalMemType =
      UR_EXP_EXTERNAL_MEM_TYPE_WIN32_NT;

  void *RetNTHandle;

  Adapter->call<sycl::errc::runtime,
                sycl::detail::UrApiKind::urMemoryExportExportMemoryHandleExp>(
      UrCtx, UrDevice, UrExternalMemType, DeviceMemory,
      static_cast<void *>(&RetNTHandle));

  return RetNTHandle;
}

} // namespace detail

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
