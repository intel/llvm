//==------------ interop_handle.hpp --- SYCL interop handle ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/access/access.hpp>
#include <CL/sycl/accessor.hpp>
#include <CL/sycl/backend_types.hpp>
#include <CL/sycl/detail/accessor_impl.hpp>
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/defines.hpp>
#include <CL/sycl/detail/pi.hpp>

#include <memory>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

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
  /// accessor), the exception `cl::sycl::invalid_object` is thrown
  /// asynchronously.
  template <backend Backend = backend::opencl, typename DataT, int Dims,
            access::mode Mode, access::target Target, access::placeholder IsPlh>
  backend_return_t<Backend, buffer<DataT, Dims>>
  get_native_mem(const accessor<DataT, Dims, Mode, Target, IsPlh> &Acc) const {
    // TODO: the method is available when the target is target::device. Add it
    // to the assert below when target::device enum is created.
    static_assert(Target == access::target::global_buffer ||
                      Target == access::target::constant_buffer,
                  "The method is available only for target::device accessors");
#ifndef __SYCL_DEVICE_ONLY__
    if (Backend != get_backend())
      throw invalid_object_error("Incorrect backend argument was passed",
                                 PI_INVALID_MEM_OBJECT);
    const auto *AccBase = static_cast<const detail::AccessorBaseHost *>(&Acc);
    return getMemImpl<Backend, DataT, Dims>(
        detail::getSyclObjImpl(*AccBase).get());
#else
    (void)Acc;
    // we believe this won't be ever called on device side
    return 0;
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
    // TODO: replace the exception thrown below with the SYCL-2020 exception
    // with the error code 'errc::backend_mismatch' when those new exceptions
    // are ready to be used.
    if (Backend != get_backend())
      throw invalid_object_error("Incorrect backend argument was passed",
                                 PI_INVALID_MEM_OBJECT);
    return reinterpret_cast<backend_return_t<Backend, queue>>(getNativeQueue());
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
    // TODO: replace the exception thrown below with the SYCL-2020 exception
    // with the error code 'errc::backend_mismatch' when those new exceptions
    // are ready to be used.
    if (Backend != get_backend())
      throw invalid_object_error("Incorrect backend argument was passed",
                                 PI_INVALID_MEM_OBJECT);
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
    // TODO: replace the exception thrown below with the SYCL-2020 exception
    // with the error code 'errc::backend_mismatch' when those new exceptions
    // are ready to be used.
    if (Backend != get_backend())
      throw invalid_object_error("Incorrect backend argument was passed",
                                 PI_INVALID_MEM_OBJECT);
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
  using ReqToMem = std::pair<detail::Requirement *, pi_mem>;

  interop_handle(std::vector<ReqToMem> MemObjs,
                 const std::shared_ptr<detail::queue_impl> &Queue,
                 const std::shared_ptr<detail::device_impl> &Device,
                 const std::shared_ptr<detail::context_impl> &Context)
      : MQueue(Queue), MDevice(Device), MContext(Context),
        MMemObjs(std::move(MemObjs)) {}

  template <backend Backend, typename DataT, int Dims>
  backend_return_t<Backend, buffer<DataT, Dims>>
  getMemImpl(detail::Requirement *Req) const {
    /*
      Do not update this cast: a C-style cast is required here.

      This function tries to cast pi_native_handle to the native handle type.
      pi_native_handle is a typedef of uintptr_t. It is used to store opaque
      pointers, such as cl_device, and integer handles, such as CUdevice. To
      convert a uintptr_t to a pointer type, such as cl_device, reinterpret_cast
      must be used. However, reinterpret_cast cannot be used to convert
      uintptr_t to a different integer type, such as CUdevice. For this,
      static_cast must be used. This function must employ a cast that is capable
      of reinterpret_cast and static_cast depending on the arguments passed to
      it. A C-style cast will achieve this. The compiler will attempt to
      interpret it as a static_cast, and will fall back to reinterpret_cast
      where appropriate.

      https://en.cppreference.com/w/cpp/language/reinterpret_cast
      https://en.cppreference.com/w/cpp/language/explicit_cast
      */
    return (backend_return_t<Backend, buffer<DataT, Dims>>)(getNativeMem(Req));
  }

  __SYCL_EXPORT pi_native_handle getNativeMem(detail::Requirement *Req) const;
  __SYCL_EXPORT pi_native_handle getNativeQueue() const;
  __SYCL_EXPORT pi_native_handle getNativeDevice() const;
  __SYCL_EXPORT pi_native_handle getNativeContext() const;

  std::shared_ptr<detail::queue_impl> MQueue;
  std::shared_ptr<detail::device_impl> MDevice;
  std::shared_ptr<detail::context_impl> MContext;

  std::vector<ReqToMem> MMemObjs;
};

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
