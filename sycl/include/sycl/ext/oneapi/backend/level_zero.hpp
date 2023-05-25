//==--------- level_zero.hpp - SYCL Level-Zero backend ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/backend.hpp>

#include <vector>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext::oneapi::level_zero {
// Implementation of various "make" functions resides in libsycl.so and thus
// their interface needs to be backend agnostic.
// TODO: remove/merge with similar functions in sycl::detail
__SYCL_EXPORT platform make_platform(pi_native_handle NativeHandle);
__SYCL_EXPORT device make_device(const platform &Platform,
                                 pi_native_handle NativeHandle);
__SYCL_EXPORT context make_context(const std::vector<device> &DeviceList,
                                   pi_native_handle NativeHandle,
                                   bool keep_ownership = false);
__SYCL_EXPORT queue make_queue(const context &Context, const device &Device,
                               pi_native_handle InteropHandle,
                               bool IsImmCmdList, bool keep_ownership,
                               const property_list &Properties);
__SYCL_EXPORT event make_event(const context &Context,
                               pi_native_handle InteropHandle,
                               bool keep_ownership = false);

// Construction of SYCL platform.
template <typename T,
          typename std::enable_if_t<std::is_same_v<T, platform>> * = nullptr>
__SYCL_DEPRECATED("Use SYCL 2020 sycl::make_platform free function")
T make(typename sycl::detail::interop<backend::ext_oneapi_level_zero, T>::type
           Interop) {
  return make_platform(reinterpret_cast<pi_native_handle>(Interop));
}

// Construction of SYCL device.
template <typename T,
          typename std::enable_if_t<std::is_same_v<T, device>> * = nullptr>
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
template <typename T, std::enable_if_t<std::is_same_v<T, context>> * = nullptr>
__SYCL_DEPRECATED("Use SYCL 2020 sycl::make_context free function")
T make(const std::vector<device> &DeviceList,
       typename sycl::detail::interop<backend::ext_oneapi_level_zero, T>::type
           Interop,
       ownership Ownership = ownership::transfer) {
  return make_context(DeviceList,
                      sycl::detail::pi::cast<pi_native_handle>(Interop),
                      Ownership == ownership::keep);
}

// Construction of SYCL event.
template <typename T,
          typename std::enable_if_t<std::is_same_v<T, event>> * = nullptr>
__SYCL_DEPRECATED("Use SYCL 2020 sycl::make_event free function")
T make(const context &Context,
       typename sycl::detail::interop<backend::ext_oneapi_level_zero, T>::type
           Interop,
       ownership Ownership = ownership::transfer) {
  return make_event(Context, reinterpret_cast<pi_native_handle>(Interop),
                    Ownership == ownership::keep);
}

} // namespace ext::oneapi::level_zero

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
  const device Device = device{BackendObject.Device};
  bool IsImmCmdList = std::holds_alternative<ze_command_list_handle_t>(
      BackendObject.NativeHandle);
  pi_native_handle Handle = IsImmCmdList
                                ? reinterpret_cast<pi_native_handle>(
                                      *(std::get_if<ze_command_list_handle_t>(
                                          &BackendObject.NativeHandle)))
                                : reinterpret_cast<pi_native_handle>(
                                      *(std::get_if<ze_command_queue_handle_t>(
                                          &BackendObject.NativeHandle)));
  return ext::oneapi::level_zero::make_queue(
      TargetContext, Device, Handle, IsImmCmdList,
      BackendObject.Ownership == ext::oneapi::level_zero::ownership::keep,
      BackendObject.Properties);
}

// Specialization of sycl::get_native for Level-Zero backend.
template <>
inline auto get_native<backend::ext_oneapi_level_zero, queue>(const queue &Obj)
    -> backend_return_t<backend::ext_oneapi_level_zero, queue> {
  int32_t IsImmCmdList;
  pi_native_handle Handle = Obj.getNative(IsImmCmdList);
  return IsImmCmdList
             ? backend_return_t<
                   backend::ext_oneapi_level_zero,
                   queue>{reinterpret_cast<ze_command_list_handle_t>(Handle)}
             : backend_return_t<backend::ext_oneapi_level_zero, queue>{
                   reinterpret_cast<ze_command_queue_handle_t>(Handle)};
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

// Specialization of sycl::make_buffer with event for Level-Zero backend.
template <backend Backend, typename T, int Dimensions = 1,
          typename AllocatorT = buffer_allocator<std::remove_const_t<T>>>
std::enable_if_t<Backend == backend::ext_oneapi_level_zero,
                 buffer<T, Dimensions, AllocatorT>>
make_buffer(
    const backend_input_t<backend::ext_oneapi_level_zero,
                          buffer<T, Dimensions, AllocatorT>> &BackendObject,
    const context &TargetContext, event AvailableEvent) {
  return detail::make_buffer_helper<T, Dimensions, AllocatorT>(
      detail::pi::cast<pi_native_handle>(BackendObject.NativeHandle),
      TargetContext, AvailableEvent,
      !(BackendObject.Ownership == ext::oneapi::level_zero::ownership::keep));
}

// Specialization of sycl::make_buffer for Level-Zero backend.
template <backend Backend, typename T, int Dimensions = 1,
          typename AllocatorT = buffer_allocator<std::remove_const_t<T>>>
std::enable_if_t<Backend == backend::ext_oneapi_level_zero,
                 buffer<T, Dimensions, AllocatorT>>
make_buffer(
    const backend_input_t<backend::ext_oneapi_level_zero,
                          buffer<T, Dimensions, AllocatorT>> &BackendObject,
    const context &TargetContext) {
  return detail::make_buffer_helper<T, Dimensions, AllocatorT>(
      detail::pi::cast<pi_native_handle>(BackendObject.NativeHandle),
      TargetContext, event{},
      !(BackendObject.Ownership == ext::oneapi::level_zero::ownership::keep));
}

// Specialization of sycl::make_image for Level-Zero backend.
template <backend Backend, int Dimensions = 1,
          typename AllocatorT = image_allocator>
std::enable_if_t<Backend == backend::ext_oneapi_level_zero,
                 image<Dimensions, AllocatorT>>
make_image(const backend_input_t<Backend, image<Dimensions, AllocatorT>>
               &BackendObject,
           const context &TargetContext, event AvailableEvent) {

  bool OwnNativeHandle =
      (BackendObject.Ownership == ext::oneapi::level_zero::ownership::transfer);

  return image<Dimensions, AllocatorT>(
      detail::pi::cast<pi_native_handle>(BackendObject.ZeImageHandle),
      TargetContext, AvailableEvent, BackendObject.ChanOrder,
      BackendObject.ChanType, OwnNativeHandle, BackendObject.Range);
}

namespace __SYCL2020_DEPRECATED("use 'ext::oneapi::level_zero' instead")
    level_zero {
using namespace ext::oneapi::level_zero;
}

} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
