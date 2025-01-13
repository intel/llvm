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
#include <sycl/exception.hpp>
#include <sycl/ext/oneapi/virtual_mem/physical_mem.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {

inline ur_virtual_mem_access_flag_t AccessModeToVirtualAccessFlags(
    ext::oneapi::experimental::address_access_mode Mode) {
  switch (Mode) {
  case ext::oneapi::experimental::address_access_mode::read:
    return UR_VIRTUAL_MEM_ACCESS_FLAG_READ_ONLY;
  case ext::oneapi::experimental::address_access_mode::read_write:
    return UR_VIRTUAL_MEM_ACCESS_FLAG_READ_WRITE;
  case ext::oneapi::experimental::address_access_mode::none:
    return UR_VIRTUAL_MEM_ACCESS_FLAG_NONE;
  }
  throw sycl::exception(make_error_code(errc::invalid),
                        "Invalid address_access_mode.");
}

class physical_mem_impl {
public:
  physical_mem_impl(const device &SyclDevice, const context &SyclContext,
                    size_t NumBytes)
      : MDevice(getSyclObjImpl(SyclDevice)),
        MContext(getSyclObjImpl(SyclContext)), MNumBytes(NumBytes) {
    const AdapterPtr &Adapter = MContext->getAdapter();

    auto Err = Adapter->call_nocheck<UrApiKind::urPhysicalMemCreate>(
        MContext->getHandleRef(), MDevice->getHandleRef(), MNumBytes, nullptr,
        &MPhysicalMem);

    if (Err == UR_RESULT_ERROR_OUT_OF_RESOURCES ||
        Err == UR_RESULT_ERROR_OUT_OF_HOST_MEMORY)
      throw sycl::exception(make_error_code(errc::memory_allocation),
                            "Failed to allocate physical memory.");
    Adapter->checkUrResult(Err);
  }

  ~physical_mem_impl() noexcept(false) {
    const AdapterPtr &Adapter = MContext->getAdapter();
    Adapter->call<UrApiKind::urPhysicalMemRelease>(MPhysicalMem);
  }

  void *map(uintptr_t Ptr, size_t NumBytes,
            ext::oneapi::experimental::address_access_mode Mode,
            size_t Offset) const {
    auto AccessFlags = AccessModeToVirtualAccessFlags(Mode);
    const AdapterPtr &Adapter = MContext->getAdapter();
    void *ResultPtr = reinterpret_cast<void *>(Ptr);
    Adapter->call<UrApiKind::urVirtualMemMap>(MContext->getHandleRef(),
                                              ResultPtr, NumBytes, MPhysicalMem,
                                              Offset, AccessFlags);
    return ResultPtr;
  }

  context get_context() const {
    return createSyclObjFromImpl<context>(MContext);
  }
  device get_device() const { return createSyclObjFromImpl<device>(MDevice); }
  size_t size() const noexcept { return MNumBytes; }

  ur_physical_mem_handle_t &getHandleRef() { return MPhysicalMem; }
  const ur_physical_mem_handle_t &getHandleRef() const { return MPhysicalMem; }

private:
  ur_physical_mem_handle_t MPhysicalMem = nullptr;
  const std::shared_ptr<device_impl> MDevice;
  const std::shared_ptr<context_impl> MContext;
  const size_t MNumBytes;
};

} // namespace detail
} // namespace _V1
} // namespace sycl
