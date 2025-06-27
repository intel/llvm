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

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

__SYCL_EXPORT void *alloc_exportable_device_mem(
    size_t alignment, size_t size,
    export_external_mem_handle_type externalMemHandleType,
    const sycl::device &syclDevice, const sycl::context &syclContext,
    [[maybe_unused]] const sycl::property_list &propList) {

  if (!syclDevice.has(sycl::aspect::ext_oneapi_memory_export_linear)) {
    throw sycl::exception(
        sycl::make_error_code(sycl::errc::feature_not_supported),
        "Device does not support memory export");
  }

  auto [urDevice, urCtx, Adapter] = get_ur_handles(syclDevice, syclContext);

  void *retDeviceMemory = nullptr;

  ur_exp_external_mem_type_t urExternalMemType;
  switch (externalMemHandleType) {
  case export_external_mem_handle_type::opaque_fd:
    urExternalMemType = UR_EXP_EXTERNAL_MEM_TYPE_OPAQUE_FD;
    break;
  case export_external_mem_handle_type::win32_nt:
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

__SYCL_EXPORT void free_exportable_memory(void *deviceMemory,
                                          const sycl::device &syclDevice,
                                          const sycl::context &syclContext) {
  if (!syclDevice.has(sycl::aspect::ext_oneapi_memory_export_linear)) {
    throw sycl::exception(
        sycl::make_error_code(sycl::errc::feature_not_supported),
        "Device does not support memory export");
  }

  auto [urDevice, urCtx, Adapter] = get_ur_handles(syclDevice, syclContext);

  Adapter->call<sycl::errc::runtime,
                sycl::detail::UrApiKind::urMemoryExportFreeExportableMemoryExp>(
      urCtx, urDevice, deviceMemory);

  return;
}

namespace detail {

__SYCL_EXPORT int
export_device_mem_opaque_fd(void *deviceMemory, const sycl::device &syclDevice,
                            const sycl::context &syclContext) {
  if (!syclDevice.has(sycl::aspect::ext_oneapi_memory_export_linear)) {
    throw sycl::exception(
        sycl::make_error_code(sycl::errc::feature_not_supported),
        "Device does not support memory export");
  }

  auto [urDevice, urCtx, Adapter] = get_ur_handles(syclDevice, syclContext);

  ur_exp_external_mem_type_t urExternalMemType =
      UR_EXP_EXTERNAL_MEM_TYPE_OPAQUE_FD;

  int retFDHandle = 0;

  Adapter->call<sycl::errc::runtime,
                sycl::detail::UrApiKind::urMemoryExportExportMemoryHandleExp>(
      urCtx, urDevice, urExternalMemType, deviceMemory,
      reinterpret_cast<void *>(&retFDHandle));

  return retFDHandle;
}

__SYCL_EXPORT void *
export_device_mem_win32_nt_handle(void *deviceMemory,
                                  const sycl::device &syclDevice,
                                  const sycl::context &syclContext) {
  if (!syclDevice.has(sycl::aspect::ext_oneapi_memory_export_linear)) {
    throw sycl::exception(
        sycl::make_error_code(sycl::errc::feature_not_supported),
        "Device does not support memory export");
  }

  auto [urDevice, urCtx, Adapter] = get_ur_handles(syclDevice, syclContext);

  ur_exp_external_mem_type_t urExternalMemType =
      UR_EXP_EXTERNAL_MEM_TYPE_WIN32_NT;

  void *retNTHandle;

  Adapter->call<sycl::errc::runtime,
                sycl::detail::UrApiKind::urMemoryExportExportMemoryHandleExp>(
      urCtx, urDevice, urExternalMemType, deviceMemory,
      static_cast<void *>(&retNTHandle));

  return retNTHandle;
}

} // namespace detail

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
