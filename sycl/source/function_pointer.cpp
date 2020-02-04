//==----------- function_pointer.cpp --- SYCL Function pointers ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/detail/program_impl.hpp>
#include <CL/sycl/intel/function_pointer.hpp>

__SYCL_INLINE namespace cl {
  namespace sycl {
  namespace detail {
  pi_device getRawDevice(device &D) {
    return detail::pi::cast<pi_device>(
        detail::getSyclObjImpl(D)->getHandleRef());
  }
  pi_program getRawProgram(program &P) {
    return detail::pi::cast<pi_program>(
        detail::getSyclObjImpl(P)->getHandleRef());
  }
  } // namespace detail
  } // namespace sycl
}