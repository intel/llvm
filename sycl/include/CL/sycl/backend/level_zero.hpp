//==--------- level_zero.hpp - SYCL Level-Zero backend ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl.hpp>
// This header should be included by users.
//#include <level_zero/ze_api.h>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

template <> struct interop<backend::level_zero, platform> {
  using type = ze_driver_handle_t;
};

template <> struct interop<backend::level_zero, device> {
  using type = ze_device_handle_t;
};

template <> struct interop<backend::level_zero, context> {
  using type = ze_context_handle_t;
};

template <> struct interop<backend::level_zero, queue> {
  using type = ze_command_queue_handle_t;
};

template <> struct interop<backend::level_zero, program> {
  using type = ze_module_handle_t;
};

template <typename DataT, int Dimensions, access::mode AccessMode>
struct interop<backend::level_zero, accessor<DataT, Dimensions, AccessMode,
                                             access::target::global_buffer,
                                             access::placeholder::false_t>> {
  using type = char *;
};

template <typename DataT, int Dimensions, access::mode AccessMode>
struct interop<backend::level_zero, accessor<DataT, Dimensions, AccessMode,
                                             access::target::constant_buffer,
                                             access::placeholder::false_t>> {
  using type = char *;
};

namespace level_zero {

// Implementation of various "make" functions resides in libsycl.so
__SYCL_EXPORT platform make_platform(pi_native_handle NativeHandle);
__SYCL_EXPORT device make_device(const platform &Platform,
                                 pi_native_handle NativeHandle);
__SYCL_EXPORT context make_context(const vector_class<device> &DeviceList,
                                   pi_native_handle NativeHandle);
__SYCL_EXPORT program make_program(const context &Context,
                                   pi_native_handle NativeHandle);
__SYCL_EXPORT queue make_queue(const context &Context,
                               pi_native_handle InteropHandle);

// Construction of SYCL platform.
template <typename T, typename detail::enable_if_t<
                          detail::is_same_v<T, platform>> * = nullptr>
T make(typename interop<backend::level_zero, T>::type Interop) {
  return make_platform(reinterpret_cast<pi_native_handle>(Interop));
}

// Construction of SYCL device.
template <typename T, typename detail::enable_if_t<detail::is_same_v<T, device>>
                          * = nullptr>
T make(const platform &Platform,
       typename interop<backend::level_zero, T>::type Interop) {
  return make_device(Platform, reinterpret_cast<pi_native_handle>(Interop));
}

/// Construction of SYCL context.
/// \param DeviceList is a vector of devices which must be encapsulated by
///        created SYCL context. Provided devices and native context handle must
///        be associated with the same platform.
/// \param Interop is a Level Zero native context handle.
template <typename T, typename std::enable_if<
                          std::is_same<T, context>::value>::type * = nullptr>
T make(const vector_class<device> &DeviceList,
       typename interop<backend::level_zero, T>::type Interop) {
  return make_context(DeviceList, detail::pi::cast<pi_native_handle>(Interop));
}

// Construction of SYCL program.
template <typename T, typename detail::enable_if_t<
                          detail::is_same_v<T, program>> * = nullptr>
T make(const context &Context,
       typename interop<backend::level_zero, T>::type Interop) {
  return make_program(Context, reinterpret_cast<pi_native_handle>(Interop));
}

// Construction of SYCL queue.
template <typename T,
          typename detail::enable_if_t<detail::is_same_v<T, queue>> * = nullptr>
T make(const context &Context,
       typename interop<backend::level_zero, T>::type Interop) {
  return make_queue(Context, reinterpret_cast<pi_native_handle>(Interop));
}

} // namespace level_zero
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
