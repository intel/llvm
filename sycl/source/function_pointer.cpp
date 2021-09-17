//==----------- function_pointer.cpp --- SYCL Function pointers ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/device_impl.hpp>
#include <detail/program_impl.hpp>
#include <sycl/ext/oneapi/__function_pointer.hpp>

__SYCL_OPEN_NS() {
namespace detail {
ext::oneapi::device_func_ptr_holder_t
getDeviceFunctionPointerImpl(device &D, program &P, const char *FuncName) {
  ext::oneapi::device_func_ptr_holder_t FPtr = 0;
  // FIXME: return value must be checked here, but since we cannot yet check
  // if corresponding extension is supported, let's silently ignore it here.
  const detail::plugin &Plugin = detail::getSyclObjImpl(P)->getPlugin();
  Plugin.call<__sycl_ns::detail::PiApiKind::piextGetDeviceFunctionPointer>(
      detail::pi::cast<pi_device>(detail::getSyclObjImpl(D)->getHandleRef()),
      detail::pi::cast<pi_program>(detail::getSyclObjImpl(P)->getHandleRef()),
      FuncName, &FPtr);
  return FPtr;
}

} // namespace detail
} // __SYCL_OPEN_NS()
__SYCL_CLOSE_NS()
