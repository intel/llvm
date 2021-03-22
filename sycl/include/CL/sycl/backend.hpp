//==---------------- backend.hpp - SYCL PI backends ------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/accessor.hpp>
#include <CL/sycl/backend_types.hpp>
#include <CL/sycl/buffer.hpp>
#include <CL/sycl/context.hpp>
#include <CL/sycl/detail/backend_traits.hpp>
#include <CL/sycl/detail/pi.hpp>
#include <CL/sycl/device.hpp>
#include <CL/sycl/event.hpp>
#include <CL/sycl/exception.hpp>
#include <CL/sycl/platform.hpp>
#include <CL/sycl/queue.hpp>

#include <type_traits>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

template <backend BackendName, class SyclObjectT>
auto get_native(const SyclObjectT &Obj) ->
    typename interop<BackendName, SyclObjectT>::type {
  // TODO use SYCL 2020 exception when implemented
  if (Obj.get_backend() != BackendName)
    throw runtime_error("Backends mismatch", PI_INVALID_OPERATION);
  return Obj.template get_native<BackendName>();
}

// Native handle of an accessor should be accessed through interop_handler
template <backend BackendName, typename DataT, int Dimensions,
          access::mode AccessMode, access::target AccessTarget,
          access::placeholder IsPlaceholder>
auto get_native(const accessor<DataT, Dimensions, AccessMode, AccessTarget,
                               IsPlaceholder> &Obj) ->
    typename interop<BackendName, accessor<DataT, Dimensions, AccessMode,
                                           AccessTarget, IsPlaceholder>>::type =
    delete;

namespace detail {
__SYCL_EXPORT platform make_platform(pi_native_handle NativeHandle,
                                     backend Backend);
__SYCL_EXPORT device make_device(pi_native_handle NativeHandle,
                                 backend Backend);
__SYCL_EXPORT context make_context(pi_native_handle NativeHandle,
                                   const async_handler &Handler,
                                   backend Backend);
__SYCL_EXPORT queue make_queue(pi_native_handle NativeHandle,
                               const context &TargetContext,
                               const async_handler &Handler, backend Backend);
__SYCL_EXPORT event make_event(pi_native_handle NativeHandle,
                               const context &TargetContext, backend Backend);
} // namespace detail

template <backend Backend>
typename std::enable_if<
    detail::InteropFeatureSupportMap<Backend>::MakePlatform == true,
    platform>::type
make_platform(const typename interop<Backend, platform>::type &BackendObject) {
  return detail::make_platform(
      detail::pi::cast<pi_native_handle>(BackendObject), Backend);
}

template <backend Backend>
typename std::enable_if<
    detail::InteropFeatureSupportMap<Backend>::MakeDevice == true, device>::type
make_device(const typename interop<Backend, device>::type &BackendObject) {
  return detail::make_device(detail::pi::cast<pi_native_handle>(BackendObject),
                             Backend);
}

template <backend Backend>
typename std::enable_if<
    detail::InteropFeatureSupportMap<Backend>::MakeContext == true,
    context>::type
make_context(const typename interop<Backend, context>::type &BackendObject,
             const async_handler &Handler = {}) {
  return detail::make_context(detail::pi::cast<pi_native_handle>(BackendObject),
                              Handler, Backend);
}

template <backend Backend>
typename std::enable_if<
    detail::InteropFeatureSupportMap<Backend>::MakeQueue == true, queue>::type
make_queue(const typename interop<Backend, queue>::type &BackendObject,
           const context &TargetContext, const async_handler Handler = {}) {
  return detail::make_queue(detail::pi::cast<pi_native_handle>(BackendObject),
                            TargetContext, Handler, Backend);
}

template <backend Backend>
typename std::enable_if<
    detail::InteropFeatureSupportMap<Backend>::MakeEvent == true, event>::type
make_event(const typename interop<Backend, event>::type &BackendObject,
           const context &TargetContext) {
  return detail::make_event(detail::pi::cast<pi_native_handle>(BackendObject),
                            TargetContext, Backend);
}

template <backend Backend, typename T, int Dimensions = 1,
          typename AllocatorT = buffer_allocator>
typename std::enable_if<detail::InteropFeatureSupportMap<Backend>::MakeBuffer ==
                            true,
                        buffer<T, Dimensions, AllocatorT>>::type
make_buffer(
    const typename interop<Backend, buffer<T, Dimensions, AllocatorT>>::type
        &BackendObject,
    const context &TargetContext, event AvailableEvent = {}) {
  return buffer<T, Dimensions, AllocatorT>(
      reinterpret_cast<cl_mem>(BackendObject), TargetContext, AvailableEvent);
}
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
