//==---------------- backend.hpp - SYCL PI backends ------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/access/access.hpp>             // for mode, placeholder
#include <sycl/accessor.hpp>                  // for accessor
#include <sycl/async_handler.hpp>             // for async_handler
#include <sycl/backend_types.hpp>             // for backend
#include <sycl/buffer.hpp>                    // for buffer_allocator
#include <sycl/context.hpp>                   // for context, get_na...
#include <sycl/detail/backend_traits.hpp>     // for InteropFeatureS...
#include <sycl/detail/cl.h>                   // for _cl_event
#include <sycl/detail/defines_elementary.hpp> // for __SYCL_DEPRECATED
#include <sycl/detail/export.hpp>             // for __SYCL_EXPORT
#include <sycl/detail/impl_utils.hpp>         // for createSyclObjFr...
#include <sycl/detail/pi.h>                   // for pi_native_handle
#include <sycl/device.hpp>                    // for device, get_native
#include <sycl/event.hpp>                     // for event, get_native
#include <sycl/exception.hpp>                 // for make_error_code
#include <sycl/feature_test.hpp>              // for SYCL_BACKEND_OP...
#include <sycl/handler.hpp>                   // for buffer
#include <sycl/image.hpp>                     // for image, image_al...
#include <sycl/kernel.hpp>                    // for kernel, get_native
#include <sycl/kernel_bundle.hpp>             // for kernel_bundle
#include <sycl/kernel_bundle_enums.hpp>       // for bundle_state
#include <sycl/platform.hpp>                  // for platform, get_n...
#include <sycl/property_list.hpp>             // for property_list
#include <sycl/queue.hpp>                     // for queue, get_native

#if SYCL_BACKEND_OPENCL
#include <sycl/detail/backend_traits_opencl.hpp> // for interop
#endif
#if SYCL_EXT_ONEAPI_BACKEND_CUDA
#ifdef SYCL_EXT_ONEAPI_BACKEND_CUDA_EXPERIMENTAL
#include <sycl/ext/oneapi/experimental/backend/backend_traits_cuda.hpp>
#else
#include <sycl/detail/backend_traits_cuda.hpp>
#endif
#endif
#if SYCL_EXT_ONEAPI_BACKEND_HIP
#include <sycl/detail/backend_traits_hip.hpp>
#endif
#if SYCL_EXT_ONEAPI_BACKEND_LEVEL_ZERO
#include <sycl/detail/backend_traits_level_zero.hpp> // for _ze_command_lis...
#endif

#include <memory>      // for shared_ptr
#include <stdint.h>    // for int32_t
#include <type_traits> // for enable_if_t
#include <vector>      // for vector

namespace sycl {
inline namespace _V1 {

namespace detail {
// Convert from PI backend to SYCL backend enum
backend convertBackend(pi_platform_backend PiBackend);
} // namespace detail

template <backend Backend> class backend_traits {
public:
  template <class T>
  using input_type = typename detail::BackendInput<Backend, T>::type;

  template <class T>
  using return_type = typename detail::BackendReturn<Backend, T>::type;
};

template <backend Backend, typename SyclType>
using backend_input_t =
    typename backend_traits<Backend>::template input_type<SyclType>;

template <backend Backend, typename SyclType>
using backend_return_t =
    typename backend_traits<Backend>::template return_type<SyclType>;

namespace detail {
template <backend Backend, typename DataT, int Dimensions, typename AllocatorT>
struct BufferInterop {
  using ReturnType =
      backend_return_t<Backend, buffer<DataT, Dimensions, AllocatorT>>;

  static ReturnType GetNativeObjs(const std::vector<pi_native_handle> &Handle) {
    ReturnType ReturnValue = 0;
    if (Handle.size()) {
      ReturnValue = detail::pi::cast<ReturnType>(Handle[0]);
    }
    return ReturnValue;
  }
};

template <typename DataT, int Dimensions, typename AllocatorT>
struct BufferInterop<backend::opencl, DataT, Dimensions, AllocatorT> {
  using ReturnType =
      backend_return_t<backend::opencl, buffer<DataT, Dimensions, AllocatorT>>;

  static ReturnType GetNativeObjs(const std::vector<pi_native_handle> &Handle) {
    ReturnType ReturnValue{};
    for (auto &Obj : Handle) {
      ReturnValue.push_back(
          detail::pi::cast<typename decltype(ReturnValue)::value_type>(Obj));
    }
    return ReturnValue;
  }
};

#if SYCL_EXT_ONEAPI_BACKEND_LEVEL_ZERO
template <backend BackendName, typename DataT, int Dimensions,
          typename AllocatorT>
auto get_native_buffer(const buffer<DataT, Dimensions, AllocatorT, void> &Obj)
    -> backend_return_t<BackendName,
                        buffer<DataT, Dimensions, AllocatorT, void>> {
  // No check for backend mismatch because buffer can be allocated on different
  // backends
  if (BackendName == backend::ext_oneapi_level_zero)
    throw sycl::exception(make_error_code(errc::feature_not_supported),
                          "Buffer interop is not supported by level zero yet");
  return Obj.template getNative<BackendName>();
}
#endif
} // namespace detail

template <backend BackendName, class SyclObjectT>
auto get_native(const SyclObjectT &Obj)
    -> backend_return_t<BackendName, SyclObjectT> {
  if (Obj.get_backend() != BackendName) {
    throw sycl::exception(make_error_code(errc::backend_mismatch),
                          "Backends mismatch");
  }
  return reinterpret_cast<backend_return_t<BackendName, SyclObjectT>>(
      Obj.getNative());
}

template <backend BackendName>
auto get_native(const queue &Obj) -> backend_return_t<BackendName, queue> {
  if (Obj.get_backend() != BackendName) {
    throw sycl::exception(make_error_code(errc::backend_mismatch),
                          "Backends mismatch");
  }
  int32_t IsImmCmdList;
  pi_native_handle Handle = Obj.getNative(IsImmCmdList);
  backend_return_t<BackendName, queue> RetVal;
  if constexpr (BackendName == backend::ext_oneapi_level_zero)
    RetVal = IsImmCmdList
                 ? backend_return_t<BackendName, queue>{reinterpret_cast<
                       ze_command_list_handle_t>(Handle)}
                 : backend_return_t<BackendName, queue>{
                       reinterpret_cast<ze_command_queue_handle_t>(Handle)};
  else
    RetVal = reinterpret_cast<backend_return_t<BackendName, queue>>(Handle);

  return RetVal;
}

template <backend BackendName, bundle_state State>
auto get_native(const kernel_bundle<State> &Obj)
    -> backend_return_t<BackendName, kernel_bundle<State>> {
  if (Obj.get_backend() != BackendName) {
    throw sycl::exception(make_error_code(errc::backend_mismatch),
                          "Backends mismatch");
  }
  return Obj.template getNative<BackendName>();
}

template <backend BackendName, typename DataT, int Dimensions,
          typename AllocatorT>
auto get_native(const buffer<DataT, Dimensions, AllocatorT> &Obj)
    -> backend_return_t<BackendName, buffer<DataT, Dimensions, AllocatorT>> {
  return detail::get_native_buffer<BackendName>(Obj);
}

#if SYCL_BACKEND_OPENCL
template <>
inline backend_return_t<backend::opencl, event>
get_native<backend::opencl, event>(const event &Obj) {
  if (Obj.get_backend() != backend::opencl) {
    throw sycl::exception(make_error_code(errc::backend_mismatch),
                          "Backends mismatch");
  }
  backend_return_t<backend::opencl, event> ReturnValue;
  for (auto const &element : Obj.getNativeVector()) {
    ReturnValue.push_back(
        reinterpret_cast<
            typename detail::interop<backend::opencl, event>::value_type>(
            element));
  }
  return ReturnValue;
}
#endif

#if SYCL_EXT_ONEAPI_BACKEND_CUDA
template <>
inline backend_return_t<backend::ext_oneapi_cuda, device>
get_native<backend::ext_oneapi_cuda, device>(const device &Obj) {
  if (Obj.get_backend() != backend::ext_oneapi_cuda) {
    throw sycl::exception(make_error_code(errc::backend_mismatch),
                          "Backends mismatch");
  }
  // CUDA uses a 32-bit int instead of an opaque pointer like other backends,
  // so we need a specialization with static_cast instead of reinterpret_cast.
  return static_cast<backend_return_t<backend::ext_oneapi_cuda, device>>(
      Obj.getNative());
}

#ifndef SYCL_EXT_ONEAPI_BACKEND_CUDA_EXPERIMENTAL
template <>
__SYCL_DEPRECATED(
    "Context interop is deprecated for CUDA. If a native context is required,"
    " use cuDevicePrimaryCtxRetain with a native device")
inline backend_return_t<backend::ext_oneapi_cuda, context> get_native<
    backend::ext_oneapi_cuda, context>(const context &Obj) {
  if (Obj.get_backend() != backend::ext_oneapi_cuda) {
    throw sycl::exception(make_error_code(errc::backend_mismatch),
                          "Backends mismatch");
  }
  return reinterpret_cast<backend_return_t<backend::ext_oneapi_cuda, context>>(
      Obj.getNative());
}

#endif // SYCL_EXT_ONEAPI_BACKEND_CUDA_EXPERIMENTAL
#endif // SYCL_EXT_ONEAPI_BACKEND_CUDA

#if SYCL_EXT_ONEAPI_BACKEND_HIP

template <>
__SYCL_DEPRECATED(
    "Context interop is deprecated for HIP. If a native context is required,"
    " use hipDevicePrimaryCtxRetain with a native device")
inline backend_return_t<backend::ext_oneapi_hip, context> get_native<
    backend::ext_oneapi_hip, context>(const context &Obj) {
  if (Obj.get_backend() != backend::ext_oneapi_hip) {
    throw sycl::exception(make_error_code(errc::backend_mismatch),
                          "Backends mismatch");
  }
  return reinterpret_cast<backend_return_t<backend::ext_oneapi_hip, context>>(
      Obj.getNative());
}

#endif // SYCL_EXT_ONEAPI_BACKEND_HIP

template <backend BackendName, typename DataT, int Dimensions,
          access::mode AccessMode, access::target AccessTarget,
          access::placeholder IsPlaceholder>
auto get_native(const accessor<DataT, Dimensions, AccessMode, AccessTarget,
                               IsPlaceholder> &Obj) ->
    typename detail::interop<
        BackendName, accessor<DataT, Dimensions, AccessMode, AccessTarget,
                              IsPlaceholder>>::type = delete;

namespace detail {
// Forward declaration
class kernel_bundle_impl;

__SYCL_EXPORT platform make_platform(pi_native_handle NativeHandle,
                                     backend Backend);
__SYCL_EXPORT device make_device(pi_native_handle NativeHandle,
                                 backend Backend);
__SYCL_EXPORT context make_context(pi_native_handle NativeHandle,
                                   const async_handler &Handler,
                                   backend Backend, bool KeepOwnership,
                                   const std::vector<device> &DeviceList = {});
__SYCL_EXPORT queue make_queue(pi_native_handle NativeHandle,
                               int32_t nativeHandleDesc,
                               const context &TargetContext,
                               const device *TargetDevice, bool KeepOwnership,
                               const property_list &PropList,
                               const async_handler &Handler, backend Backend);
__SYCL_EXPORT event make_event(pi_native_handle NativeHandle,
                               const context &TargetContext, backend Backend);
__SYCL_EXPORT event make_event(pi_native_handle NativeHandle,
                               const context &TargetContext, bool KeepOwnership,
                               backend Backend);
// TODO: Unused. Remove when allowed.
__SYCL_EXPORT kernel make_kernel(pi_native_handle NativeHandle,
                                 const context &TargetContext, backend Backend);
__SYCL_EXPORT kernel make_kernel(
    const context &TargetContext,
    const kernel_bundle<bundle_state::executable> &KernelBundle,
    pi_native_handle NativeKernelHandle, bool KeepOwnership, backend Backend);
// TODO: Unused. Remove when allowed.
__SYCL_EXPORT std::shared_ptr<detail::kernel_bundle_impl>
make_kernel_bundle(pi_native_handle NativeHandle, const context &TargetContext,
                   bundle_state State, backend Backend);
__SYCL_EXPORT std::shared_ptr<detail::kernel_bundle_impl>
make_kernel_bundle(pi_native_handle NativeHandle, const context &TargetContext,
                   bool KeepOwnership, bundle_state State, backend Backend);
} // namespace detail

template <backend Backend>
std::enable_if_t<
    detail::InteropFeatureSupportMap<Backend>::MakePlatform == true, platform>
make_platform(
    const typename backend_traits<Backend>::template input_type<platform>
        &BackendObject) {
  return detail::make_platform(
      detail::pi::cast<pi_native_handle>(BackendObject), Backend);
}

template <backend Backend>
std::enable_if_t<detail::InteropFeatureSupportMap<Backend>::MakeDevice == true,
                 device>
make_device(const typename backend_traits<Backend>::template input_type<device>
                &BackendObject) {
  for (auto p : platform::get_platforms()) {
    if (p.get_backend() != Backend)
      continue;

    for (auto d : p.get_devices()) {
      if (get_native<Backend>(d) == BackendObject)
        return d;
    }
  }

  return detail::make_device(detail::pi::cast<pi_native_handle>(BackendObject),
                             Backend);
}

template <backend Backend>
std::enable_if_t<detail::InteropFeatureSupportMap<Backend>::MakeContext == true,
                 context>
make_context(
    const typename backend_traits<Backend>::template input_type<context>
        &BackendObject,
    const async_handler &Handler = {}) {
  return detail::make_context(detail::pi::cast<pi_native_handle>(BackendObject),
                              Handler, Backend, false /* KeepOwnership */);
}

template <backend Backend>
std::enable_if_t<detail::InteropFeatureSupportMap<Backend>::MakeQueue == true,
                 queue>
make_queue(const typename backend_traits<Backend>::template input_type<queue>
               &BackendObject,
           const context &TargetContext, const async_handler Handler = {}) {
  auto KeepOwnership =
      Backend == backend::ext_oneapi_cuda || Backend == backend::ext_oneapi_hip;
  return detail::make_queue(detail::pi::cast<pi_native_handle>(BackendObject),
                            false, TargetContext, nullptr, KeepOwnership, {},
                            Handler, Backend);
}

template <backend Backend>
std::enable_if_t<detail::InteropFeatureSupportMap<Backend>::MakeEvent == true,
                 event>
make_event(const typename backend_traits<Backend>::template input_type<event>
               &BackendObject,
           const context &TargetContext) {
  return detail::make_event(detail::pi::cast<pi_native_handle>(BackendObject),
                            TargetContext, Backend);
}

template <backend Backend>
__SYCL_DEPRECATED("Use SYCL 2020 sycl::make_event free function")
std::enable_if_t<detail::InteropFeatureSupportMap<Backend>::MakeEvent == true,
                 event> make_event(const typename backend_traits<Backend>::
                                       template input_type<event>
                                           &BackendObject,
                                   const context &TargetContext,
                                   bool KeepOwnership) {
  return detail::make_event(detail::pi::cast<pi_native_handle>(BackendObject),
                            TargetContext, KeepOwnership, Backend);
}

template <backend Backend, typename T, int Dimensions = 1,
          typename AllocatorT = buffer_allocator<std::remove_const_t<T>>>
std::enable_if_t<detail::InteropFeatureSupportMap<Backend>::MakeBuffer ==
                         true &&
                     Backend != backend::ext_oneapi_level_zero,
                 buffer<T, Dimensions, AllocatorT>>
make_buffer(const typename backend_traits<Backend>::template input_type<
                buffer<T, Dimensions, AllocatorT>> &BackendObject,
            const context &TargetContext, event AvailableEvent = {}) {
  return detail::make_buffer_helper<T, Dimensions, AllocatorT>(
      detail::pi::cast<pi_native_handle>(BackendObject), TargetContext,
      AvailableEvent);
}

template <backend Backend, int Dimensions = 1,
          typename AllocatorT = image_allocator>
std::enable_if_t<detail::InteropFeatureSupportMap<Backend>::MakeImage == true &&
                     Backend != backend::ext_oneapi_level_zero,
                 image<Dimensions, AllocatorT>>
make_image(const typename backend_traits<Backend>::template input_type<
               image<Dimensions, AllocatorT>> &BackendObject,
           const context &TargetContext, event AvailableEvent = {}) {
  return image<Dimensions, AllocatorT>(
      detail::pi::cast<pi_native_handle>(BackendObject), TargetContext,
      AvailableEvent);
}

template <backend Backend>
kernel
make_kernel(const typename backend_traits<Backend>::template input_type<kernel>
                &BackendObject,
            const context &TargetContext) {
  return detail::make_kernel(detail::pi::cast<pi_native_handle>(BackendObject),
                             TargetContext, Backend);
}

template <backend Backend, bundle_state State>
std::enable_if_t<detail::InteropFeatureSupportMap<Backend>::MakeKernelBundle ==
                     true,
                 kernel_bundle<State>>
make_kernel_bundle(const typename backend_traits<Backend>::template input_type<
                       kernel_bundle<State>> &BackendObject,
                   const context &TargetContext) {
  std::shared_ptr<detail::kernel_bundle_impl> KBImpl =
      detail::make_kernel_bundle(
          detail::pi::cast<pi_native_handle>(BackendObject), TargetContext,
          false, State, Backend);
  return detail::createSyclObjFromImpl<kernel_bundle<State>>(KBImpl);
}
} // namespace _V1
} // namespace sycl
