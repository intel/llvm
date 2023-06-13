//==------- interop_handler.cpp - Argument for codeplay_introp_task --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/queue_impl.hpp>
#include <sycl/detail/defines.hpp>
#include <sycl/detail/pi.hpp>
#include <sycl/interop_handler.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {

pi_native_handle
interop_handler::GetNativeQueue(int32_t &NativeHandleDesc) const {
  return MQueue->getNative(NativeHandleDesc);
}

pi_native_handle interop_handler::GetNativeMem(detail::Requirement *Req) const {
  auto Iter = std::find_if(std::begin(MMemObjs), std::end(MMemObjs),
                           [=](ReqToMem Elem) { return (Elem.first == Req); });

  if (Iter == std::end(MMemObjs)) {
    throw("Invalid memory object used inside interop");
  }

  auto Plugin = MQueue->getPlugin();
  pi_native_handle Handle;
  Plugin->call<detail::PiApiKind::piextMemGetNativeHandle>(Iter->second,
                                                           &Handle);
  return Handle;
}

} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
