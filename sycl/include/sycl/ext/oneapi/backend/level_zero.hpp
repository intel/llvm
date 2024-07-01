//==--------- level_zero.hpp - SYCL Level-Zero backend ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/async_handler.hpp>                    // for async_han...
#include <sycl/backend.hpp>                          // for backend_i...
#include <sycl/backend_types.hpp>                    // for backend
#include <sycl/buffer.hpp>                           // for buffer_al...
#include <sycl/context.hpp>                          // for context
#include <sycl/detail/backend_traits.hpp>            // for interop
#include <sycl/detail/backend_traits_level_zero.hpp> // for ze_comman...
#include <sycl/detail/defines_elementary.hpp>        // for __SYCL_DE...
#include <sycl/detail/export.hpp>                    // for __SYCL_EX...
#include <sycl/detail/impl_utils.hpp>                // for createSyc...
#include <sycl/detail/pi.h>                          // for pi_native...
#include <sycl/detail/pi.hpp>                        // for cast
#include <sycl/device.hpp>                           // for device
#include <sycl/event.hpp>                            // for event
#include <sycl/ext/codeplay/experimental/fusion_properties.hpp> // for buffer
#include <sycl/ext/oneapi/backend/level_zero_ownership.hpp>     // for ownership
#include <sycl/image.hpp>                                       // for image
#include <sycl/kernel.hpp>                                      // for kernel
#include <sycl/kernel_bundle.hpp>               // for kernel_bu...
#include <sycl/kernel_bundle_enums.hpp>         // for bundle_state
#include <sycl/platform.hpp>                    // for platform
#include <sycl/properties/image_properties.hpp> // for image
#include <sycl/property_list.hpp>               // for property_...
#include <sycl/queue.hpp>                       // for queue

#include <memory>      // for shared_ptr
#include <stdint.h>    // for int32_t
#include <type_traits> // for enable_if_t
#include <variant>     // for get_if
#include <vector>      // for vector

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::level_zero::detail {
__SYCL_EXPORT device make_device(const platform &Platform,
                                 pi_native_handle NativeHandle);
} // namespace ext::oneapi::level_zero::detail

// Specialization of sycl::make_context for Level-Zero backend.
template <>
inline context make_context<backend::ext_oneapi_level_zero>(
    const backend_input_t<backend::ext_oneapi_level_zero, context>
        &BackendObject,
    const async_handler &Handler) {

  const std::vector<device> &DeviceList = BackendObject.DeviceList;
  pi_native_handle NativeHandle =
      detail::pi::cast<pi_native_handle>(BackendObject.NativeHandle);
  bool KeepOwnership =
      BackendObject.Ownership == ext::oneapi::level_zero::ownership::keep;

  return sycl::detail::make_context(NativeHandle, Handler,
                                    backend::ext_oneapi_level_zero,
                                    KeepOwnership, DeviceList);
}

namespace detail {
inline std::optional<sycl::device> find_matching_descendent_device(
    sycl::device d,
    const backend_input_t<backend::ext_oneapi_level_zero, device>
        &BackendObject) {
  if (get_native<backend::ext_oneapi_level_zero>(d) == BackendObject)
    return d;
  std::vector<info::partition_property> partition_props =
      d.get_info<info::device::partition_properties>();

  for (auto pp : partition_props) {
    if (pp == info::partition_property::partition_by_affinity_domain) {
      auto sub_devices = d.create_sub_devices<
          info::partition_property::partition_by_affinity_domain>(
          info::partition_affinity_domain::next_partitionable);
      for (auto sub_dev : sub_devices) {
        if (auto maybe_device =
                find_matching_descendent_device(sub_dev, BackendObject))
          return maybe_device;
      }
    }

    assert(false && "Unexpected partitioning scheme for a Level-Zero device!");
  }

  return {};
}
} // namespace detail

// Specialization of sycl::make_device for Level-Zero backend.
// Level-Zero backend specification says:
//
//  > The SYCL execution environment for the Level Zero backend contains a fixed
//  > number of devices that are enumerated via sycl::device::get_devices() and
//  > a fixed number of sub-devices that are enumerated via
//  > sycl::device::create_sub_devices(...). Calling this function does not
//  > create a new device. Rather it merely creates a sycl::device object that
//  > is a copy of one of the devices from those enumerations.
//
// Per SYCL 2020 specification, device and it's copy should be equally
// comparable and its hashes must be equal. As such, we cannot simply create a
// new `detail::device_impl` and then a `sycl::device` out of it and have to
// iterate over the existing device hierarchy and make a copy.
template <>
inline device make_device<backend::ext_oneapi_level_zero>(
    const backend_input_t<backend::ext_oneapi_level_zero, device>
        &BackendObject) {
  for (auto p : platform::get_platforms()) {
    if (p.get_backend() != backend::ext_oneapi_level_zero)
      continue;

    for (auto d : p.get_devices()) {
      if (auto maybe_device = find_matching_descendent_device(d, BackendObject))
        return *maybe_device;
    }
  }

  throw sycl::exception(make_error_code(errc::invalid),
                        "Native device isn't exposed to SYCL.");
}

// Specialization of sycl::make_queue for Level-Zero backend.
template <>
inline queue make_queue<backend::ext_oneapi_level_zero>(
    const backend_input_t<backend::ext_oneapi_level_zero, queue> &BackendObject,
    const context &TargetContext, const async_handler Handler) {
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

  return sycl::detail::make_queue(
      Handle, IsImmCmdList, TargetContext, &Device,
      BackendObject.Ownership == ext::oneapi::level_zero::ownership::keep,
      BackendObject.Properties, Handler, backend::ext_oneapi_level_zero);
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
  return sycl::detail::make_event(
      detail::pi::cast<pi_native_handle>(BackendObject.NativeHandle),
      TargetContext,
      BackendObject.Ownership == ext::oneapi::level_zero::ownership::keep,
      backend::ext_oneapi_level_zero);
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

} // namespace _V1
} // namespace sycl
