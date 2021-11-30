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

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {
ext::oneapi::device_func_ptr_holder_t
getDeviceFunctionPointerImpl(device &D, program &P, const char *FuncName) {
  ext::oneapi::device_func_ptr_holder_t FPtr = 0;

  const detail::plugin &Plugin = detail::getSyclObjImpl(P)->getPlugin();
  pi_result Result = Plugin.call_nocheck<
      cl::sycl::detail::PiApiKind::piextGetDeviceFunctionPointer>(
      detail::pi::cast<pi_device>(detail::getSyclObjImpl(D)->getHandleRef()),
      detail::pi::cast<pi_program>(detail::getSyclObjImpl(P)->getHandleRef()),
      FuncName, &FPtr);
  // If extension is not supported, a fallback method is used which returns
  // if the function exists or not. If the return address is not valid throw
  // error.
  if (Result != PI_SUCCESS) {
    if (Result == PI_FUNCTION_ADDRESS_IS_NOT_AVAILABLE)
      throw runtime_error("Function exists but address is not available",
                          Result);
    if (Result == PI_INVALID_KERNEL_NAME)
      throw runtime_error("Function name is not valid", Result);
    else
      throw runtime_error("piextGetDeviceFunctionPointer returned error when "
                          "retrieving address",
                          Result);
  }

  return FPtr;
}

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
