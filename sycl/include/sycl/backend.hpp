//==---------------- backend.hpp - SYCL PI backends ------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/accessor.hpp>
#include <sycl/backend_types.hpp>
#include <sycl/buffer.hpp>
#include <sycl/context.hpp>
#include <sycl/detail/backend_traits.hpp>
#include <sycl/feature_test.hpp>
#if SYCL_BACKEND_OPENCL
#include <sycl/detail/backend_traits_opencl.hpp>
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
#include <sycl/detail/backend_traits_level_zero.hpp>
#endif
#include <sycl/detail/common.hpp>
#include <sycl/detail/export.hpp>
#include <sycl/detail/pi.h>
#include <sycl/detail/pi.hpp>
#include <sycl/device.hpp>
#include <sycl/event.hpp>
#include <sycl/exception.hpp>
#include <sycl/kernel_bundle.hpp>
#include <sycl/platform.hpp>
#include <sycl/queue.hpp>

#include <type_traits>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {

namespace detail {
// TODO each backend can have its own custom errc enumeration
// but the details for this are not fully specified yet
enum class backend_errc : unsigned int {};
} // namespace detail

template <backend Backend> class backend_traits {
public:
  template <class T>
  using input_type = typename detail::BackendInput<Backend, T>::type;

  template <class T>
  using return_type = typename detail::BackendReturn<Backend, T>::type;

  using errc = detail::backend_errc;
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
    throw sycl::runtime_error(
        errc::feature_not_supported,
        "Buffer interop is not supported by level zero yet",
        PI_ERROR_INVALID_OPERATION);
  return Obj.template getNative<BackendName>();
}
#endif
} // namespace detail

template <backend BackendName, class SyclObjectT>
auto get_native(const SyclObjectT &Obj)
    -> backend_return_t<BackendName, SyclObjectT> {
  // TODO use SYCL 2020 exception when implemented
  if (Obj.get_backend() != BackendName) {
    throw sycl::runtime_error(errc::backend_mismatch, "Backends mismatch",
                              PI_ERROR_INVALID_OPERATION);
  }
  return reinterpret_cast<backend_return_t<BackendName, SyclObjectT>>(
      Obj.getNative());
}

template <backend BackendName, bundle_state State>
auto get_native(const kernel_bundle<State> &Obj)
    -> backend_return_t<BackendName, kernel_bundle<State>> {
  // TODO use SYCL 2020 exception when implemented
  if (Obj.get_backend() != BackendName) {
    throw sycl::runtime_error(errc::backend_mismatch, "Backends mismatch",
                              PI_ERROR_INVALID_OPERATION);
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
  // TODO use SYCL 2020 exception when implemented
  if (Obj.get_backend() != backend::opencl) {
    throw sycl::runtime_error(errc::backend_mismatch, "Backends mismatch",
                              PI_ERROR_INVALID_OPERATION);
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
  // TODO use SYCL 2020 exception when implemented
  if (Obj.get_backend() != backend::ext_oneapi_cuda) {
    throw sycl::runtime_error(errc::backend_mismatch, "Backends mismatch",
                              PI_ERROR_INVALID_OPERATION);
  }
  // CUDA uses a 32-bit int instead of an opaque pointer like other backends,
  // so we need a specialization with static_cast instead of reinterpret_cast.
  return static_cast<backend_return_t<backend::ext_oneapi_cuda, device>>(
      Obj.getNative());
}
#endif

// Native handle of an accessor should be accessed through interop_handler
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
                                   backend Backend);
__SYCL_EXPORT queue make_queue(pi_native_handle NativeHandle,
                               const context &TargetContext,
                               const device *TargetDevice, bool KeepOwnership,
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
typename std::enable_if<
    detail::InteropFeatureSupportMap<Backend>::MakePlatform == true,
    platform>::type
make_platform(
    const typename backend_traits<Backend>::template input_type<platform>
        &BackendObject) {
  return detail::make_platform(
      detail::pi::cast<pi_native_handle>(BackendObject), Backend);
}

template <backend Backend>
typename std::enable_if<
    detail::InteropFeatureSupportMap<Backend>::MakeDevice == true, device>::type
make_device(const typename backend_traits<Backend>::template input_type<device>
                &BackendObject) {
  return detail::make_device(detail::pi::cast<pi_native_handle>(BackendObject),
                             Backend);
}

template <backend Backend>
typename std::enable_if<
    detail::InteropFeatureSupportMap<Backend>::MakeContext == true,
    context>::type
make_context(
    const typename backend_traits<Backend>::template input_type<context>
        &BackendObject,
    const async_handler &Handler = {}) {
  return detail::make_context(detail::pi::cast<pi_native_handle>(BackendObject),
                              Handler, Backend);
}

template <backend Backend>
typename std::enable_if<
    detail::InteropFeatureSupportMap<Backend>::MakeQueue == true, queue>::type
make_queue(const typename backend_traits<Backend>::template input_type<queue>
               &BackendObject,
           const context &TargetContext, const async_handler Handler = {}) {
  return detail::make_queue(detail::pi::cast<pi_native_handle>(BackendObject),
                            TargetContext, nullptr, false, Handler, Backend);
}

template <backend Backend>
typename std::enable_if<
    detail::InteropFeatureSupportMap<Backend>::MakeEvent == true, event>::type
make_event(const typename backend_traits<Backend>::template input_type<event>
               &BackendObject,
           const context &TargetContext) {
  return detail::make_event(detail::pi::cast<pi_native_handle>(BackendObject),
                            TargetContext, Backend);
}

template <backend Backend>
__SYCL_DEPRECATED("Use SYCL 2020 sycl::make_event free function")
typename std::enable_if<
    detail::InteropFeatureSupportMap<Backend>::MakeEvent == true, event>::type
    make_event(
        const typename backend_traits<Backend>::template input_type<event>
            &BackendObject,
        const context &TargetContext, bool KeepOwnership) {
  return detail::make_event(detail::pi::cast<pi_native_handle>(BackendObject),
                            TargetContext, KeepOwnership, Backend);
}

template <backend Backend, typename T, int Dimensions = 1,
          typename AllocatorT = buffer_allocator<std::remove_const_t<T>>>
typename std::enable_if<detail::InteropFeatureSupportMap<Backend>::MakeBuffer ==
                                true &&
                            Backend != backend::ext_oneapi_level_zero,
                        buffer<T, Dimensions, AllocatorT>>::type
make_buffer(const typename backend_traits<Backend>::template input_type<
                buffer<T, Dimensions, AllocatorT>> &BackendObject,
            const context &TargetContext, event AvailableEvent = {}) {
  return detail::make_buffer_helper<T, Dimensions, AllocatorT>(
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
typename std::enable_if<
    detail::InteropFeatureSupportMap<Backend>::MakeKernelBundle == true,
    kernel_bundle<State>>::type
make_kernel_bundle(const typename backend_traits<Backend>::template input_type<
                       kernel_bundle<State>> &BackendObject,
                   const context &TargetContext) {
  std::shared_ptr<detail::kernel_bundle_impl> KBImpl =
      detail::make_kernel_bundle(
          detail::pi::cast<pi_native_handle>(BackendObject), TargetContext,
          false, State, Backend);
  return detail::createSyclObjFromImpl<kernel_bundle<State>>(KBImpl);
}
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
