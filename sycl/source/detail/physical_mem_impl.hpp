//==- physical_mem_impl.hpp - sycl_ext_oneapi_virtual_mem physical_mem impl ==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <detail/context_impl.hpp>
#include <detail/device_impl.hpp>
#include <sycl/access/access.hpp>
#include <sycl/context.hpp>
#include <sycl/detail/common.hpp>
#include <sycl/device.hpp>
#include <sycl/ext/oneapi/virtual_mem/physical_mem.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {

inline RT::PiVirtualAccessFlags AccessModeToVirtualAccessFlags(
    ext::oneapi::experimental::address_access_mode Mode) {
  switch (Mode) {
  case ext::oneapi::experimental::address_access_mode::read:
    return PI_VIRTUAL_ACCESS_FLAG_READ_ONLY;
  case ext::oneapi::experimental::address_access_mode::read_write:
    return PI_VIRTUAL_ACCESS_FLAG_RW;
  case ext::oneapi::experimental::address_access_mode::none:
    return 0;
  default:
    throw sycl::exception(make_error_code(errc::invalid),
                          "Invalid address_access_mode.");
  }
}

class physical_mem_impl {
public:
  physical_mem_impl(const device &SyclDevice, const context &SyclContext,
                    size_t NumBytes)
      : MDevice(getSyclObjImpl(SyclDevice)),
        MContext(getSyclObjImpl(SyclContext)), MNumBytes(NumBytes) {
    const plugin &Plugin = MContext->getPlugin();
    Plugin.call<PiApiKind::piextPhysicalMemCreate>(MContext->getHandleRef(),
                                                   MDevice->getHandleRef(),
                                                   MNumBytes, &MPhysicalMem);
  }

  ~physical_mem_impl() {
    const plugin &Plugin = MContext->getPlugin();
    Plugin.call<PiApiKind::piextPhysicalMemRelease>(MPhysicalMem);
  }

  void *map(uintptr_t Ptr, size_t NumBytes,
            ext::oneapi::experimental::address_access_mode Mode,
            size_t Offset) const {
    RT::PiVirtualAccessFlags AccessFlags = AccessModeToVirtualAccessFlags(Mode);
    const plugin &Plugin = MContext->getPlugin();
    void *ResultPtr = reinterpret_cast<void *>(Ptr);
    Plugin.call<PiApiKind::piextVirtualMemMap>(
        MContext->getHandleRef(), ResultPtr, NumBytes, MPhysicalMem, Offset,
        AccessFlags);
    return ResultPtr;
  }

  context get_context() const {
    return createSyclObjFromImpl<context>(MContext);
  }
  device get_device() const { return createSyclObjFromImpl<device>(MDevice); }
  size_t size() const noexcept { return MNumBytes; }

  RT::PiPhysicalMem &getHandleRef() { return MPhysicalMem; }
  const RT::PiPhysicalMem &getHandleRef() const { return MPhysicalMem; }

private:
  RT::PiPhysicalMem MPhysicalMem = nullptr;
  const std::shared_ptr<device_impl> MDevice;
  const std::shared_ptr<context_impl> MContext;
  const size_t MNumBytes;
};

} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
