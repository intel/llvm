//==------------ interop_handle.hpp --- SYCL interop handle ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/access/access.hpp>
#include <sycl/accessor.hpp>
#include <sycl/backend_types.hpp>
#include <sycl/detail/common.hpp>
#include <sycl/detail/defines.hpp>
#include <sycl/detail/pi.hpp>

#include <memory>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {

namespace detail {
class AccessorBaseHost;
class ExecCGCommand;
class DispatchHostTask;
class queue_impl;
class device_impl;
class context_impl;
} // namespace detail

class queue;
class device;
class context;

class interop_handle {
public:
  interop_handle() = delete;

  /// Returns a backend associated with the queue associated with this
  /// interop_handle.
  __SYCL_EXPORT backend get_backend() const noexcept;

  /// Receives a SYCL accessor that has been defined as a requirement for the
  /// command group, and returns the underlying OpenCL memory object that is
  /// used by the SYCL runtime. If the accessor passed as parameter is not part
  /// of the command group requirements (e.g. it is an unregistered placeholder
  /// accessor), the exception `sycl::invalid_object` is thrown
  /// asynchronously.
  template <backend Backend = backend::opencl, typename DataT, int Dims,
            access::mode Mode, access::target Target, access::placeholder IsPlh,
            typename PropertyListT = ext::oneapi::accessor_property_list<>>
  std::enable_if_t<Target != access::target::image,
                   backend_return_t<Backend, buffer<DataT, Dims>>>
  get_native_mem(const accessor<DataT, Dims, Mode, Target, IsPlh, PropertyListT>
                     &Acc) const {
    static_assert(Target == access::target::device ||
                      Target == access::target::constant_buffer,
                  "The method is available only for target::device accessors");
#ifndef __SYCL_DEVICE_ONLY__
    if (Backend != get_backend())
      throw invalid_object_error("Incorrect backend argument was passed",
                                 PI_ERROR_INVALID_MEM_OBJECT);
    const auto *AccBase = static_cast<const detail::AccessorBaseHost *>(&Acc);
    return getMemImpl<Backend, DataT, Dims>(
        detail::getSyclObjImpl(*AccBase).get());
#else
    (void)Acc;
    // we believe this won't be ever called on device side
    return backend_return_t<Backend, buffer<DataT, Dims>>{0};
#endif
  }

  /// Receives a SYCL accessor that has been defined as a requirement for the
  /// command group, and returns the underlying OpenCL memory object that is
  /// used by the SYCL runtime. If the accessor passed as parameter is not part
  /// of the command group requirements (e.g. it is an unregistered placeholder
  /// accessor), the exception `sycl::invalid_object` is thrown
  /// asynchronously.
  template <backend Backend = backend::opencl, typename DataT, int Dims,
            access::mode Mode, access::target Target, access::placeholder IsPlh>
  backend_return_t<Backend, image<Dims>> get_native_mem(
      const detail::image_accessor<DataT, Dims, Mode,
                                   /*access::target::image*/ Target,
                                   IsPlh /*, PropertyListT */> &Acc) const {
#ifndef __SYCL_DEVICE_ONLY__
    if (Backend != get_backend())
      throw invalid_object_error("Incorrect backend argument was passed",
                                 PI_ERROR_INVALID_MEM_OBJECT);
    const auto *AccBase = static_cast<const detail::AccessorBaseHost *>(&Acc);
    return getMemImpl<Backend, Dims>(detail::getSyclObjImpl(*AccBase).get());
#else
    (void)Acc;
    // we believe this won't be ever called on device side
    return backend_return_t<Backend, image<Dims>>{0};
#endif
  }

  /// Returns an underlying native backend object associated with teh queue
  /// that the host task was submitted to. If the command group was submitted
  /// with a secondary queue and the fall-back was triggered, the queue that
  /// is associated with the interop_handle must be the fall-back queue.
  /// The native backend object returned must be in a state where it is capable
  /// of being used in a way appropriate for the associated SYCL backend. It is
  /// implementation-defined in cases where the SYCL queue maps to multiple
  /// underlying backend objects. It is responsibility of the SYCL runtime to
  /// ensure the backend queue returned is in a state that can be used to
  /// dispatch work, and that other potential backend command queues associated
  /// with the same SYCL command queue are not executing commands while the host
  /// task is executing.
  template <backend Backend = backend::opencl>
  backend_return_t<Backend, queue> get_native_queue() const {
#ifndef __SYCL_DEVICE_ONLY__
    // TODO: replace the exception thrown below with the SYCL 2020 exception
    // with the error code 'errc::backend_mismatch' when those new exceptions
    // are ready to be used.
    if (Backend != get_backend())
      throw invalid_object_error("Incorrect backend argument was passed",
                                 PI_ERROR_INVALID_MEM_OBJECT);
    int32_t NativeHandleDesc;
    return reinterpret_cast<backend_return_t<Backend, queue>>(
        getNativeQueue(NativeHandleDesc));
#else
    // we believe this won't be ever called on device side
    return 0;
#endif
  }

  /// Returns the SYCL application interoperability native backend object
  /// associated with the device associated with the SYCL queue that the host
  /// task was submitted to. The native backend object returned must be in
  /// a state where it is capable of being used in a way appropriate for
  /// the associated SYCL backend.
  template <backend Backend = backend::opencl>
  backend_return_t<Backend, device> get_native_device() const {
#ifndef __SYCL_DEVICE_ONLY__
    // TODO: replace the exception thrown below with the SYCL 2020 exception
    // with the error code 'errc::backend_mismatch' when those new exceptions
    // are ready to be used.
    if (Backend != get_backend())
      throw invalid_object_error("Incorrect backend argument was passed",
                                 PI_ERROR_INVALID_MEM_OBJECT);
    // C-style cast required to allow various native types
    return (backend_return_t<Backend, device>)getNativeDevice();
#else
    // we believe this won't be ever called on device side
    return 0;
#endif
  }

  /// Returns the SYCL application interoperability native backend object
  /// associated with the context associated with the SYCL queue that the host
  /// task was submitted to. The native backend object returned must be in
  /// a state where it is capable of being used in a way appropriate for
  /// the associated SYCL backend.
  template <backend Backend = backend::opencl>
  backend_return_t<Backend, context> get_native_context() const {
#ifndef __SYCL_DEVICE_ONLY__
    // TODO: replace the exception thrown below with the SYCL 2020 exception
    // with the error code 'errc::backend_mismatch' when those new exceptions
    // are ready to be used.
    if (Backend != get_backend())
      throw invalid_object_error("Incorrect backend argument was passed",
                                 PI_ERROR_INVALID_MEM_OBJECT);
    return reinterpret_cast<backend_return_t<Backend, context>>(
        getNativeContext());
#else
    // we believe this won't be ever called on device side
    return 0;
#endif
  }

private:
  friend class detail::ExecCGCommand;
  friend class detail::DispatchHostTask;
  using ReqToMem = std::pair<detail::AccessorImplHost *, pi_mem>;

  interop_handle(std::vector<ReqToMem> MemObjs,
                 const std::shared_ptr<detail::queue_impl> &Queue,
                 const std::shared_ptr<detail::device_impl> &Device,
                 const std::shared_ptr<detail::context_impl> &Context)
      : MQueue(Queue), MDevice(Device), MContext(Context),
        MMemObjs(std::move(MemObjs)) {}

  template <backend Backend, typename DataT, int Dims>
  backend_return_t<Backend, buffer<DataT, Dims>>
  getMemImpl(detail::AccessorImplHost *Req) const {
    std::vector<pi_native_handle> NativeHandles{getNativeMem(Req)};
    return detail::BufferInterop<Backend, DataT, Dims>::GetNativeObjs(
        NativeHandles);
  }

  template <backend Backend, int Dims>
  backend_return_t<Backend, image<Dims>>
  getMemImpl(detail::AccessorImplHost *Req) const {
    using image_return_t = backend_return_t<Backend, image<Dims>>;
    return reinterpret_cast<image_return_t>(getNativeMem(Req));
  }

  __SYCL_EXPORT pi_native_handle
  getNativeMem(detail::AccessorImplHost *Req) const;
  __SYCL_EXPORT pi_native_handle
  getNativeQueue(int32_t &NativeHandleDesc) const;
  __SYCL_EXPORT pi_native_handle getNativeDevice() const;
  __SYCL_EXPORT pi_native_handle getNativeContext() const;

  std::shared_ptr<detail::queue_impl> MQueue;
  std::shared_ptr<detail::device_impl> MDevice;
  std::shared_ptr<detail::context_impl> MContext;

  std::vector<ReqToMem> MMemObjs;
};

} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
