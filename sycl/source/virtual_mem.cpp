//==- virtual_mem.cpp - sycl_ext_oneapi_virtual_mem virtual mem free funcs -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/context_impl.hpp>
#include <detail/device_impl.hpp>
#include <detail/physical_mem_impl.hpp>
#include <detail/virtual_mem.hpp>

// System headers for querying page-size.
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

size_t
get_mem_granularity_for_allocation_size(const detail::device_impl &SyclDevice,
                                        const detail::context_impl &SyclContext,
                                        granularity_mode Mode,
                                        const size_t AllocationSize) {
  if (!SyclDevice.has(aspect::ext_oneapi_virtual_mem))
    throw sycl::exception(
        sycl::make_error_code(sycl::errc::feature_not_supported),
        "Device does not support aspect::ext_oneapi_virtual_mem.");

  ur_virtual_mem_granularity_info_t GranularityQuery = [=]() {
    switch (Mode) {
    case granularity_mode::minimum:
      return UR_VIRTUAL_MEM_GRANULARITY_INFO_MINIMUM;
    case granularity_mode::recommended:
      return UR_VIRTUAL_MEM_GRANULARITY_INFO_RECOMMENDED;
    }
    throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                          "Unrecognized granularity mode.");
  }();

  auto [urDevice, urCtx, Adapter] = get_ur_handles(SyclDevice, SyclContext);
#ifndef NDEBUG
  size_t InfoOutputSize = 0;
  Adapter->call<sycl::detail::UrApiKind::urVirtualMemGranularityGetInfo>(
      urCtx, urDevice, AllocationSize, GranularityQuery, 0u, nullptr,
      &InfoOutputSize);
  assert(InfoOutputSize == sizeof(size_t) &&
         "Unexpected output size of granularity info query.");
#endif // NDEBUG
  size_t Granularity = 0;
  Adapter->call<sycl::detail::UrApiKind::urVirtualMemGranularityGetInfo>(
      urCtx, urDevice, AllocationSize, GranularityQuery, sizeof(size_t),
      &Granularity, nullptr);
  if (Granularity == 0)
    throw sycl::exception(
        sycl::make_error_code(sycl::errc::invalid),
        "Unexpected granularity result: memory granularity shouldn't be 0.");
  return Granularity;
}

__SYCL_EXPORT size_t get_mem_granularity(const device &SyclDevice,
                                         const context &SyclContext,
                                         granularity_mode Mode) {
  return get_mem_granularity_for_allocation_size(
      *detail::getSyclObjImpl(SyclDevice), *detail::getSyclObjImpl(SyclContext),
      Mode, 1);
}

__SYCL_EXPORT size_t get_mem_granularity(const context &SyclContext,
                                         granularity_mode Mode) {
  const std::vector<device> Devices = SyclContext.get_devices();
  if (!std::all_of(Devices.cbegin(), Devices.cend(), [](const device &Dev) {
        return Dev.has(aspect::ext_oneapi_virtual_mem);
      })) {
    throw sycl::exception(
        sycl::make_error_code(sycl::errc::feature_not_supported),
        "One or more devices in the context does not support "
        "aspect::ext_oneapi_virtual_mem.");
  }

  // CUDA only needs page-size granularity.
  if (SyclContext.get_backend() == backend::ext_oneapi_cuda) {
#ifdef _WIN32
    SYSTEM_INFO SystemInfo;
    GetSystemInfo(&SystemInfo);
    return static_cast<size_t>(SystemInfo.dwPageSize);
#else
    return static_cast<size_t>(sysconf(_SC_PAGESIZE));
#endif
  }

  // Otherwise, we find the least common multiple of granularity of the devices
  // in the context.
  size_t LCMGranularity = get_mem_granularity(Devices[0], SyclContext, Mode);
  for (size_t I = 1; I < Devices.size(); ++I) {
    size_t DevGranularity = get_mem_granularity(Devices[I], SyclContext, Mode);
    size_t GCD = LCMGranularity;
    size_t Rem = DevGranularity % GCD;
    while (Rem != 0) {
      std::swap(GCD, Rem);
      Rem %= GCD;
    }
    LCMGranularity *= DevGranularity / GCD;
  }
  return LCMGranularity;
}

__SYCL_EXPORT uintptr_t reserve_virtual_mem(uintptr_t Start, size_t NumBytes,
                                            const context &SyclContext) {
  std::vector<device> Devs = SyclContext.get_devices();
  if (std::any_of(Devs.cbegin(), Devs.cend(), [](const device &Dev) {
        return !Dev.has(aspect::ext_oneapi_virtual_mem);
      }))
    throw sycl::exception(
        sycl::make_error_code(sycl::errc::feature_not_supported),
        "One or more devices in the supplied context does not support "
        "aspect::ext_oneapi_virtual_mem.");

  auto [urCtx, Adapter] = get_ur_handles(SyclContext);
  void *OutPtr = nullptr;
  Adapter->call<sycl::detail::UrApiKind::urVirtualMemReserve>(
      urCtx, reinterpret_cast<void *>(Start), NumBytes, &OutPtr);
  return reinterpret_cast<uintptr_t>(OutPtr);
}

__SYCL_EXPORT void free_virtual_mem(uintptr_t Ptr, size_t NumBytes,
                                    const context &SyclContext) {
  auto [urCtx, Adapter] = get_ur_handles(SyclContext);
  Adapter->call<sycl::detail::UrApiKind::urVirtualMemFree>(
      urCtx, reinterpret_cast<void *>(Ptr), NumBytes);
}

__SYCL_EXPORT void set_access_mode(const void *Ptr, size_t NumBytes,
                                   address_access_mode Mode,
                                   const context &SyclContext) {
  auto AccessFlags = sycl::detail::AccessModeToVirtualAccessFlags(Mode);
  auto [urCtx, Adapter] = get_ur_handles(SyclContext);
  Adapter->call<sycl::detail::UrApiKind::urVirtualMemSetAccess>(
      urCtx, Ptr, NumBytes, AccessFlags);
}

__SYCL_EXPORT address_access_mode get_access_mode(const void *Ptr,
                                                  size_t NumBytes,
                                                  const context &SyclContext) {
  auto [urCtx, Adapter] = get_ur_handles(SyclContext);
#ifndef NDEBUG
  size_t InfoOutputSize = 0;
  Adapter->call<sycl::detail::UrApiKind::urVirtualMemGetInfo>(
      urCtx, Ptr, NumBytes, UR_VIRTUAL_MEM_INFO_ACCESS_MODE, 0u, nullptr,
      &InfoOutputSize);
  assert(InfoOutputSize == sizeof(ur_virtual_mem_access_flags_t) &&
         "Unexpected output size of access mode info query.");
#endif // NDEBUG
  ur_virtual_mem_access_flags_t AccessFlags;
  Adapter->call<sycl::detail::UrApiKind::urVirtualMemGetInfo>(
      urCtx, Ptr, NumBytes, UR_VIRTUAL_MEM_INFO_ACCESS_MODE,
      sizeof(ur_virtual_mem_access_flags_t), &AccessFlags, nullptr);

  if (AccessFlags & UR_VIRTUAL_MEM_ACCESS_FLAG_READ_WRITE)
    return address_access_mode::read_write;
  if (AccessFlags & UR_VIRTUAL_MEM_ACCESS_FLAG_READ_ONLY)
    return address_access_mode::read;
  return address_access_mode::none;
}

__SYCL_EXPORT void unmap(const void *Ptr, size_t NumBytes,
                         const context &SyclContext) {
  auto [urCtx, Adapter] = get_ur_handles(SyclContext);
  Adapter->call<sycl::detail::UrApiKind::urVirtualMemUnmap>(urCtx, Ptr,
                                                            NumBytes);
}

} // Namespace ext::oneapi::experimental
} // namespace _V1
} // Namespace sycl
