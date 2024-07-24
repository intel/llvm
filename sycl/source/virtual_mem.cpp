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
#include <sycl/ext/oneapi/virtual_mem/virtual_mem.hpp>

// System headers for querying page-size.
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

__SYCL_EXPORT size_t get_mem_granularity(const device &SyclDevice,
                                         const context &SyclContext,
                                         granularity_mode Mode) {
  if (!SyclDevice.has(aspect::ext_oneapi_virtual_mem))
    throw sycl::exception(
        sycl::make_error_code(sycl::errc::feature_not_supported),
        "Device does not support aspect::ext_oneapi_virtual_mem.");

  pi_virtual_mem_granularity_info GranularityQuery = [=]() {
    switch (Mode) {
    case granularity_mode::minimum:
      return PI_EXT_ONEAPI_VIRTUAL_MEM_GRANULARITY_INFO_MINIMUM;
    case granularity_mode::recommended:
      return PI_EXT_ONEAPI_VIRTUAL_MEM_GRANULARITY_INFO_RECOMMENDED;
    }
    throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                          "Unrecognized granularity mode.");
  }();

  std::shared_ptr<sycl::detail::device_impl> DeviceImpl =
      sycl::detail::getSyclObjImpl(SyclDevice);
  std::shared_ptr<sycl::detail::context_impl> ContextImpl =
      sycl::detail::getSyclObjImpl(SyclContext);
  const sycl::detail::PluginPtr &Plugin = ContextImpl->getPlugin();
#ifndef NDEBUG
  size_t InfoOutputSize;
  Plugin->call<sycl::detail::PiApiKind::piextVirtualMemGranularityGetInfo>(
      ContextImpl->getHandleRef(), DeviceImpl->getHandleRef(), GranularityQuery,
      0, nullptr, &InfoOutputSize);
  assert(InfoOutputSize == sizeof(size_t) &&
         "Unexpected output size of granularity info query.");
#endif // NDEBUG
  size_t Granularity = 0;
  Plugin->call<sycl::detail::PiApiKind::piextVirtualMemGranularityGetInfo>(
      ContextImpl->getHandleRef(), DeviceImpl->getHandleRef(), GranularityQuery,
      sizeof(size_t), &Granularity, nullptr);
  return Granularity;
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

  std::shared_ptr<sycl::detail::context_impl> ContextImpl =
      sycl::detail::getSyclObjImpl(SyclContext);
  const sycl::detail::PluginPtr &Plugin = ContextImpl->getPlugin();
  void *OutPtr = nullptr;
  Plugin->call<sycl::detail::PiApiKind::piextVirtualMemReserve>(
      ContextImpl->getHandleRef(), reinterpret_cast<void *>(Start), NumBytes,
      &OutPtr);
  return reinterpret_cast<uintptr_t>(OutPtr);
}

__SYCL_EXPORT void free_virtual_mem(uintptr_t Ptr, size_t NumBytes,
                                    const context &SyclContext) {
  std::shared_ptr<sycl::detail::context_impl> ContextImpl =
      sycl::detail::getSyclObjImpl(SyclContext);
  const sycl::detail::PluginPtr &Plugin = ContextImpl->getPlugin();
  Plugin->call<sycl::detail::PiApiKind::piextVirtualMemFree>(
      ContextImpl->getHandleRef(), reinterpret_cast<void *>(Ptr), NumBytes);
}

__SYCL_EXPORT void set_access_mode(const void *Ptr, size_t NumBytes,
                                   address_access_mode Mode,
                                   const context &SyclContext) {
  sycl::detail::pi::PiVirtualAccessFlags AccessFlags =
      sycl::detail::AccessModeToVirtualAccessFlags(Mode);
  std::shared_ptr<sycl::detail::context_impl> ContextImpl =
      sycl::detail::getSyclObjImpl(SyclContext);
  const sycl::detail::PluginPtr &Plugin = ContextImpl->getPlugin();
  Plugin->call<sycl::detail::PiApiKind::piextVirtualMemSetAccess>(
      ContextImpl->getHandleRef(), Ptr, NumBytes, AccessFlags);
}

__SYCL_EXPORT address_access_mode get_access_mode(const void *Ptr,
                                                  size_t NumBytes,
                                                  const context &SyclContext) {
  std::shared_ptr<sycl::detail::context_impl> ContextImpl =
      sycl::detail::getSyclObjImpl(SyclContext);
  const sycl::detail::PluginPtr &Plugin = ContextImpl->getPlugin();
#ifndef NDEBUG
  size_t InfoOutputSize;
  Plugin->call<sycl::detail::PiApiKind::piextVirtualMemGetInfo>(
      ContextImpl->getHandleRef(), Ptr, NumBytes,
      PI_EXT_ONEAPI_VIRTUAL_MEM_INFO_ACCESS_MODE, 0, nullptr, &InfoOutputSize);
  assert(InfoOutputSize == sizeof(sycl::detail::pi::PiVirtualAccessFlags) &&
         "Unexpected output size of access mode info query.");
#endif // NDEBUG
  sycl::detail::pi::PiVirtualAccessFlags AccessFlags;
  Plugin->call<sycl::detail::PiApiKind::piextVirtualMemGetInfo>(
      ContextImpl->getHandleRef(), Ptr, NumBytes,
      PI_EXT_ONEAPI_VIRTUAL_MEM_INFO_ACCESS_MODE,
      sizeof(sycl::detail::pi::PiVirtualAccessFlags), &AccessFlags, nullptr);

  if (AccessFlags & PI_VIRTUAL_ACCESS_FLAG_RW)
    return address_access_mode::read_write;
  if (AccessFlags & PI_VIRTUAL_ACCESS_FLAG_READ_ONLY)
    return address_access_mode::read;
  return address_access_mode::none;
}

__SYCL_EXPORT void unmap(const void *Ptr, size_t NumBytes,
                         const context &SyclContext) {
  std::shared_ptr<sycl::detail::context_impl> ContextImpl =
      sycl::detail::getSyclObjImpl(SyclContext);
  const sycl::detail::PluginPtr &Plugin = ContextImpl->getPlugin();
  Plugin->call<sycl::detail::PiApiKind::piextVirtualMemUnmap>(
      ContextImpl->getHandleRef(), Ptr, NumBytes);
}

} // Namespace ext::oneapi::experimental
} // namespace _V1
} // Namespace sycl
