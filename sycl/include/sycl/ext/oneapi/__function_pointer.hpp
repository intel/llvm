//==----------- function_pointer.hpp --- SYCL Function pointers ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#ifdef __SYCL_INTERNAL_API

#include <sycl/detail/export.hpp>
#include <sycl/detail/stl_type_traits.hpp>
#include <sycl/device.hpp>
#include <sycl/program.hpp>
#include <sycl/stl.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {
__SYCL_EXPORT cl_ulong getDeviceFunctionPointerImpl(device &D, program &P,
                                                    const char *FuncName);
}
namespace ext {
namespace oneapi {

// This is a preview extension implementation, intended to provide early
// access to a feature for review and community feedback.
//
// Because the interfaces defined by this header file are not final and are
// subject to change they are not intended to be used by shipping software
// products. If you are interested in using this feature in your software
// product, please let us know!

using device_func_ptr_holder_t = cl_ulong;

/// \brief this function performs a cast from device_func_ptr_holder_t type
/// to the provided function pointer type.
template <
    class FuncType,
    typename FuncPtrType = typename detail::add_pointer_t<FuncType>,
    typename detail::enable_if_t<std::is_function<FuncType>::value, int> = 0>
inline FuncPtrType to_device_func_ptr(device_func_ptr_holder_t FptrHolder) {
  return reinterpret_cast<FuncPtrType>(FptrHolder);
}

template <class FuncType>
using enable_if_is_function_pointer_t = typename detail::enable_if_t<
    std::is_pointer<FuncType>::value &&
        std::is_function<typename std::remove_pointer<FuncType>::type>::value,
    int>;

/// \brief this function can be used only on host side to obtain device
/// function pointer for the specified function.
///
/// \param F - pointer to function to make it work for SYCL Host device
/// \param FuncName - name of the function. Please note that by default names
/// of functions are mangled since SYCL is a C++. To avoid the need ot
/// specifying mangled name here, use `extern "C"` \param P - sycl::program
/// object which will be used to extract device function pointer \param D -
/// sycl::device object which will be used to extract device function pointer
///
/// \returns device_func_ptr_holder_t object which can be used inside a device
/// code. This object must be converted back to a function pointer using
/// `to_device_func_ptr` prior to actual usage.
///
/// Returned value is valid only within device code which was compiled for the
/// specified program and device. Returned value invalidates whenever program
/// is released or re-built
template <class FuncType, enable_if_is_function_pointer_t<FuncType> = 0>
device_func_ptr_holder_t get_device_func_ptr(FuncType F, const char *FuncName,
                                             program &P, device &D) {
  if (program_state::linked != P.get_state()) {
    throw invalid_parameter_error(
        "Program must be built before passing to get_device_func_ptr",
        PI_ERROR_INVALID_OPERATION);
  }

  return sycl::detail::getDeviceFunctionPointerImpl(D, P, FuncName);
}
} // namespace oneapi
} // namespace ext

} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl

#endif
