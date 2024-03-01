//==------------ interop_handle.cpp --- SYCL interop handle ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/accessor_impl.hpp>
#include <detail/backend_impl.hpp>
#include <detail/context_impl.hpp>
#include <detail/device_impl.hpp>
#include <detail/queue_impl.hpp>
#include <sycl/exception.hpp>
#include <sycl/interop_handle.hpp>

#include <algorithm>

namespace sycl {
inline namespace _V1 {

backend interop_handle::get_backend() const noexcept {
  return detail::getImplBackend(MQueue);
}

pi_native_handle interop_handle::getNativeMem(detail::Requirement *Req) const {
  auto Iter = std::find_if(std::begin(MMemObjs), std::end(MMemObjs),
                           [=](ReqToMem Elem) { return (Elem.first == Req); });

  if (Iter == std::end(MMemObjs)) {
    throw invalid_object_error("Invalid memory object used inside interop",
                               PI_ERROR_INVALID_MEM_OBJECT);
  }

  auto Plugin = MQueue->getPlugin();
  pi_native_handle Handle;
  Plugin->call<detail::PiApiKind::piextMemGetNativeHandle>(
      Iter->second, MDevice->getHandleRef(), &Handle);
  return Handle;
}

pi_native_handle interop_handle::getNativeDevice() const {
  return MDevice->getNative();
}

pi_native_handle interop_handle::getNativeContext() const {
  return MContext->getNative();
}

pi_native_handle
interop_handle::getNativeQueue(int32_t &NativeHandleDesc) const {
  return MQueue->getNative(NativeHandleDesc);
}

} // namespace _V1
} // namespace sycl
