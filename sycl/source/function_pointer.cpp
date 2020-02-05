//==----------- function_pointer.cpp --- SYCL Function pointers ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/intel/function_pointer.hpp>
#include <CL/sycl/detail/device_impl.hpp>
#include <CL/sycl/detail/program_impl.hpp>

__SYCL_INLINE namespace cl {
namespace sycl {
namespace intel {
namespace detail {
device_func_ptr_holder_t getDeviceFunctionPointerImpl(device &D, program &P,
                                                      const char *FuncName) {
  device_func_ptr_holder_t FPtr = 0;
  // FIXME: return value must be checked here, but since we cannot yet check
  // if corresponding extension is supported, let's silently ignore it here.
  PI_CALL(piextGetDeviceFunctionPointer)(
      sycl::detail::pi::cast<pi_device>(sycl::detail::getSyclObjImpl(D)->getHandleRef()),
      sycl::detail::pi::cast<pi_program>(sycl::detail::getSyclObjImpl(P)->getHandleRef()),
      FuncName, &FPtr);
  return FPtr;
}

} // namespace detail
} // namespace intel
} // namespace sycl
} // namespace cl