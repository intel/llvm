//==--------- level_zero.hpp - SYCL Level-Zero backend ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/backend.hpp>
#include <CL/sycl/program.hpp>

#include <vector>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace oneapi {
namespace level_zero {
// Implementation of various "make" functions resides in libsycl.so and thus
// their interface needs to be backend agnostic.
// TODO: remove/merge with similar functions in sycl::detail
__SYCL_EXPORT platform make_platform(pi_native_handle NativeHandle);
__SYCL_EXPORT device make_device(const platform &Platform,
                                 pi_native_handle NativeHandle);
__SYCL_EXPORT context make_context(const std::vector<device> &DeviceList,
                                   pi_native_handle NativeHandle,
                                   bool keep_ownership = false);
#ifdef __SYCL_INTERNAL_API
__SYCL_EXPORT program make_program(const context &Context,
                                   pi_native_handle NativeHandle);
#endif
__SYCL_EXPORT queue make_queue(const context &Context,
                               pi_native_handle InteropHandle,
                               bool keep_ownership = false);
__SYCL_EXPORT event make_event(const context &Context,
                               pi_native_handle InteropHandle,
                               bool keep_ownership = false);

// Construction of SYCL platform.
template <typename T, typename sycl::detail::enable_if_t<
                          std::is_same<T, platform>::value> * = nullptr>
__SYCL_DEPRECATED("Use SYCL 2020 sycl::make_platform free function")
T make(typename sycl::detail::interop<backend::ext_oneapi_level_zero, T>::type
           Interop) {
  return make_platform(reinterpret_cast<pi_native_handle>(Interop));
}

// Construction of SYCL device.
template <typename T, typename sycl::detail::enable_if_t<
                          std::is_same<T, device>::value> * = nullptr>
__SYCL_DEPRECATED("Use SYCL 2020 sycl::make_device free function")
T make(const platform &Platform,
       typename sycl::detail::interop<backend::ext_oneapi_level_zero, T>::type
           Interop) {
  return make_device(Platform, reinterpret_cast<pi_native_handle>(Interop));
}

/// Construction of SYCL context.
/// \param DeviceList is a vector of devices which must be encapsulated by
///        created SYCL context. Provided devices and native context handle must
///        be associated with the same platform.
/// \param Interop is a Level Zero native context handle.
/// \param Ownership (optional) specifies who will assume ownership of the
///        native context handle. Default is that SYCL RT does, so it destroys
///        the native handle when the created SYCL object goes out of life.
///
template <typename T, typename std::enable_if<
                          std::is_same<T, context>::value>::type * = nullptr>
__SYCL_DEPRECATED("Use SYCL 2020 sycl::make_context free function")
T make(const std::vector<device> &DeviceList,
       typename sycl::detail::interop<backend::ext_oneapi_level_zero, T>::type
           Interop,
       ownership Ownership = ownership::transfer) {
  return make_context(DeviceList,
                      sycl::detail::pi::cast<pi_native_handle>(Interop),
                      Ownership == ownership::keep);
}

// Construction of SYCL program.
#ifdef __SYCL_INTERNAL_API
template <typename T, typename sycl::detail::enable_if_t<
                          std::is_same<T, program>::value> * = nullptr>
__SYCL_DEPRECATED("Use SYCL 2020 sycl::make_kernel_bundle free function")
T make(const context &Context,
       typename sycl::detail::interop<backend::ext_oneapi_level_zero, T>::type
           Interop) {
  return make_program(Context, reinterpret_cast<pi_native_handle>(Interop));
}
#endif

// Construction of SYCL queue.
template <typename T, typename sycl::detail::enable_if_t<
                          std::is_same<T, queue>::value> * = nullptr>
__SYCL_DEPRECATED("Use SYCL 2020 sycl::make_queue free function")
T make(const context &Context,
       typename sycl::detail::interop<backend::ext_oneapi_level_zero, T>::type
           Interop,
       ownership Ownership = ownership::transfer) {
  return make_queue(Context, reinterpret_cast<pi_native_handle>(Interop),
                    Ownership == ownership::keep);
}

// Construction of SYCL event.
template <typename T, typename sycl::detail::enable_if_t<
                          std::is_same<T, event>::value> * = nullptr>
__SYCL_DEPRECATED("Use SYCL 2020 sycl::make_event free function")
T make(const context &Context,
       typename sycl::detail::interop<backend::ext_oneapi_level_zero, T>::type
           Interop,
       ownership Ownership = ownership::transfer) {
  return make_event(Context, reinterpret_cast<pi_native_handle>(Interop),
                    Ownership == ownership::keep);
}
} // namespace level_zero
} // namespace oneapi
} // namespace ext

// Specialization of sycl::make_context for Level-Zero backend.
template <>
inline context make_context<backend::ext_oneapi_level_zero>(
    const backend_input_t<backend::ext_oneapi_level_zero, context>
        &BackendObject,
    const async_handler &Handler) {
  (void)Handler;
  return ext::oneapi::level_zero::make_context(
      BackendObject.DeviceList,
      detail::pi::cast<pi_native_handle>(BackendObject.NativeHandle),
      BackendObject.Ownership == ext::oneapi::level_zero::ownership::keep);
}

// Specialization of sycl::make_queue for Level-Zero backend.
template <>
inline queue make_queue<backend::ext_oneapi_level_zero>(
    const backend_input_t<backend::ext_oneapi_level_zero, queue> &BackendObject,
    const context &TargetContext, const async_handler Handler) {
  (void)Handler;
  return ext::oneapi::level_zero::make_queue(
      TargetContext,
      detail::pi::cast<pi_native_handle>(BackendObject.NativeHandle),
      BackendObject.Ownership == ext::oneapi::level_zero::ownership::keep);
}

// Specialization of sycl::make_event for Level-Zero backend.
template <>
inline event make_event<backend::ext_oneapi_level_zero>(
    const backend_input_t<backend::ext_oneapi_level_zero, event> &BackendObject,
    const context &TargetContext) {
  return ext::oneapi::level_zero::make_event(
      TargetContext,
      detail::pi::cast<pi_native_handle>(BackendObject.NativeHandle),
      BackendObject.Ownership == ext::oneapi::level_zero::ownership::keep);
}

// Specialization of sycl::make_kernel_bundle for Level-Zero backend.
template <>
inline kernel_bundle<bundle_state::executable>
make_kernel_bundle<backend::ext_oneapi_level_zero, bundle_state::executable>(
    const backend_input_t<backend::ext_oneapi_level_zero,
                          kernel_bundle<bundle_state::executable>>
        &BackendObject,
    const context &TargetContext) {
  std::shared_ptr<detail::kernel_bundle_impl> KBImpl =
      detail::make_kernel_bundle(
          detail::pi::cast<pi_native_handle>(BackendObject.NativeHandle),
          TargetContext,
          BackendObject.Ownership == ext::oneapi::level_zero::ownership::keep,
          bundle_state::executable, backend::ext_oneapi_level_zero);
  return detail::createSyclObjFromImpl<kernel_bundle<bundle_state::executable>>(
      KBImpl);
}

// Specialization of sycl::make_kernel for Level-Zero backend.
template <>
inline kernel make_kernel<backend::ext_oneapi_level_zero>(
    const backend_input_t<backend::ext_oneapi_level_zero, kernel>
        &BackendObject,
    const context &TargetContext) {
  return detail::make_kernel(
      TargetContext, BackendObject.KernelBundle,
      detail::pi::cast<pi_native_handle>(BackendObject.NativeHandle),
      BackendObject.Ownership == ext::oneapi::level_zero::ownership::keep,
      backend::ext_oneapi_level_zero);
}

// TODO: remove this specialization when generic is changed to call
// .GetNative() instead of .get_native() member of kernel_bundle.
template <>
inline auto get_native<backend::ext_oneapi_level_zero>(
    const kernel_bundle<bundle_state::executable> &Obj)
    -> backend_return_t<backend::ext_oneapi_level_zero,
                        kernel_bundle<bundle_state::executable>> {
  // TODO use SYCL 2020 exception when implemented
  if (Obj.get_backend() != backend::ext_oneapi_level_zero)
    throw runtime_error(errc::backend_mismatch, "Backends mismatch",
                        PI_INVALID_OPERATION);

  return Obj.template getNative<backend::ext_oneapi_level_zero>();
}

namespace __SYCL2020_DEPRECATED("use 'ext::oneapi::level_zero' instead")
    level_zero {
  using namespace ext::oneapi::level_zero;
}

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
