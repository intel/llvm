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

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext::oneapi::experimental {

__SYCL_EXPORT size_t get_minimum_mem_granularity(const device &SyclDevice,
                                                 const context &SyclContext) {
  if (!SyclDevice.has(aspect::ext_oneapi_virtual_mem))
    throw sycl::exception(
        sycl::make_error_code(sycl::errc::feature_not_supported),
        "Device does not support aspect::ext_oneapi_virtual_mem.");

  std::shared_ptr<sycl::detail::device_impl> DeviceImpl =
      sycl::detail::getSyclObjImpl(SyclDevice);
  std::shared_ptr<sycl::detail::context_impl> ContextImpl =
      sycl::detail::getSyclObjImpl(SyclContext);
  const sycl::detail::plugin &Plugin = ContextImpl->getPlugin();
#ifndef NDEBUG
  size_t InfoOutputSize;
  Plugin.call<sycl::detail::PiApiKind::piextVirtualMemGranularityGetInfo>(
      ContextImpl->getHandleRef(), DeviceImpl->getHandleRef(),
      PI_EXT_ONEAPI_VIRTUAL_MEM_GRANULARITY_INFO_MINIMUM, 0, nullptr,
      &InfoOutputSize);
  assert(InfoOutputSize == sizeof(size_t) &&
         "Unexpected output size of granularity info query.");
#endif // NDEBUG
  size_t Granularity;
  Plugin.call<sycl::detail::PiApiKind::piextVirtualMemGranularityGetInfo>(
      ContextImpl->getHandleRef(), DeviceImpl->getHandleRef(),
      PI_EXT_ONEAPI_VIRTUAL_MEM_GRANULARITY_INFO_MINIMUM, sizeof(size_t),
      &Granularity, nullptr);
  return Granularity;
}

__SYCL_EXPORT size_t get_recommended_mem_granularity(
    const device &SyclDevice, const context &SyclContext) {
  if (!SyclDevice.has(aspect::ext_oneapi_virtual_mem))
    throw sycl::exception(
        sycl::make_error_code(sycl::errc::feature_not_supported),
        "Device does not support aspect::ext_oneapi_virtual_mem.");

  std::shared_ptr<sycl::detail::device_impl> DeviceImpl =
      sycl::detail::getSyclObjImpl(SyclDevice);
  std::shared_ptr<sycl::detail::context_impl> ContextImpl =
      sycl::detail::getSyclObjImpl(SyclContext);
  const sycl::detail::plugin &Plugin = ContextImpl->getPlugin();
#ifndef NDEBUG
  size_t InfoOutputSize;
  Plugin.call<sycl::detail::PiApiKind::piextVirtualMemGranularityGetInfo>(
      ContextImpl->getHandleRef(), DeviceImpl->getHandleRef(),
      PI_EXT_ONEAPI_VIRTUAL_MEM_GRANULARITY_INFO_RECOMMENDED, 0, nullptr,
      &InfoOutputSize);
  assert(InfoOutputSize == sizeof(size_t) &&
         "Unexpected output size of granularity info query.");
#endif // NDEBUG
  size_t Granularity;
  Plugin.call<sycl::detail::PiApiKind::piextVirtualMemGranularityGetInfo>(
      ContextImpl->getHandleRef(), DeviceImpl->getHandleRef(),
      PI_EXT_ONEAPI_VIRTUAL_MEM_GRANULARITY_INFO_RECOMMENDED, sizeof(size_t),
      &Granularity, nullptr);
  return Granularity;
}

__SYCL_EXPORT void *reserve_virtual_mem(const void *Start, size_t NumBytes,
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
  const sycl::detail::plugin &Plugin = ContextImpl->getPlugin();
  void *OutPtr = nullptr;
  Plugin.call<sycl::detail::PiApiKind::piextVirtualMemReserve>(
      ContextImpl->getHandleRef(), Start, NumBytes, &OutPtr);
  return OutPtr;
}

__SYCL_EXPORT void free_virtual_mem(const void *Ptr, size_t NumBytes,
                                    const context &SyclContext) {
  std::shared_ptr<sycl::detail::context_impl> ContextImpl =
      sycl::detail::getSyclObjImpl(SyclContext);
  const sycl::detail::plugin &Plugin = ContextImpl->getPlugin();
  Plugin.call<sycl::detail::PiApiKind::piextVirtualMemFree>(
      ContextImpl->getHandleRef(), Ptr, NumBytes);
}

__SYCL_EXPORT void set_access_mode(const void *Ptr, size_t NumBytes,
                                   access_mode Mode,
                                   const context &SyclContext) {
  sycl::detail::RT::PiVirtualAccessFlags AccessFlags =
      sycl::detail::AccessModeToVirtualAccessFlags(Mode);
  std::shared_ptr<sycl::detail::context_impl> ContextImpl =
      sycl::detail::getSyclObjImpl(SyclContext);
  const sycl::detail::plugin &Plugin = ContextImpl->getPlugin();
  Plugin.call<sycl::detail::PiApiKind::piextVirtualMemSetAccess>(
      ContextImpl->getHandleRef(), Ptr, NumBytes, AccessFlags);
}

__SYCL_EXPORT void set_inaccessible(const void *Ptr, size_t NumBytes,
                                    const context &SyclContext) {
  std::shared_ptr<sycl::detail::context_impl> ContextImpl =
      sycl::detail::getSyclObjImpl(SyclContext);
  const sycl::detail::plugin &Plugin = ContextImpl->getPlugin();
  Plugin.call<sycl::detail::PiApiKind::piextVirtualMemSetAccess>(
      ContextImpl->getHandleRef(), Ptr, NumBytes, 0);
}

__SYCL_EXPORT std::optional<access_mode>
get_access_mode(const void *Ptr, size_t NumBytes, const context &SyclContext) {
  std::shared_ptr<sycl::detail::context_impl> ContextImpl =
      sycl::detail::getSyclObjImpl(SyclContext);
  const sycl::detail::plugin &Plugin = ContextImpl->getPlugin();
#ifndef NDEBUG
  size_t InfoOutputSize;
  Plugin.call<sycl::detail::PiApiKind::piextVirtualMemGetInfo>(
      ContextImpl->getHandleRef(), Ptr, NumBytes,
      PI_EXT_ONEAPI_VIRTUAL_MEM_INFO_ACCESS_MODE, 0, nullptr, &InfoOutputSize);
  assert(InfoOutputSize == sizeof(RT::PiVirtualAccessFlags) &&
         "Unexpected output size of access mode info query.");
#endif // NDEBUG
  RT::PiVirtualAccessFlags AccessFlags;
  Plugin.call<sycl::detail::PiApiKind::piextVirtualMemGetInfo>(
      ContextImpl->getHandleRef(), Ptr, NumBytes,
      PI_EXT_ONEAPI_VIRTUAL_MEM_INFO_ACCESS_MODE,
      sizeof(RT::PiVirtualAccessFlags), &AccessFlags, nullptr);

  if (AccessFlags & PI_VIRTUAL_ACCESS_FLAG_RW)
    return access_mode::read_write;
  if (AccessFlags & PI_VIRTUAL_ACCESS_FLAG_READ_ONLY)
    return access_mode::read;
  return std::nullopt;
}

__SYCL_EXPORT void unmap(const void *Ptr, size_t NumBytes,
                         const context &SyclContext) {
  std::shared_ptr<sycl::detail::context_impl> ContextImpl =
      sycl::detail::getSyclObjImpl(SyclContext);
  const sycl::detail::plugin &Plugin = ContextImpl->getPlugin();
  Plugin.call<sycl::detail::PiApiKind::piextVirtualMemUnmap>(
      ContextImpl->getHandleRef(), Ptr, NumBytes);
}

} // Namespace ext::oneapi::experimental
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // Namespace sycl
