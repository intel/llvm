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
  /// Receives a SYCL accessor that has been defined as a requirement for the
  /// command group, and returns the underlying OpenCL memory object that is
  /// used by the SYCL runtime. If the accessor passed as parameter is not part
  /// of the command group requirements (e.g. it is an unregistered placeholder
  /// accessor), the exception `cl::sycl::invalid_object` is thrown
  /// asynchronously.
  template <backend BackendName = backend::opencl, typename DataT, int Dims,
            access::mode Mode, access::target Target, access::placeholder IsPlh>
  typename std::enable_if<
      Target != access::target::host_buffer,
      typename interop<BackendName,
                       accessor<DataT, Dims, Mode, Target, IsPlh>>::type>::type
  get_native_mem(const accessor<DataT, Dims, Mode, Target, IsPlh> &Acc) const {
#ifndef __SYCL_DEVICE_ONLY__
    const auto *AccBase = static_cast<const detail::AccessorBaseHost *>(&Acc);
    return getMemImpl<BackendName, DataT, Dims, Mode, Target, IsPlh>(
        detail::getSyclObjImpl(*AccBase).get());
#else
    (void)Acc;
    // we believe this won't be ever called on device side
    return nullptr;
#endif
  }

  template <backend BackendName = backend::opencl, typename DataT, int Dims,
            access::mode Mode, access::target Target, access::placeholder IsPlh>
  typename std::enable_if<
      Target == access::target::host_buffer,
      typename interop<BackendName,
                       accessor<DataT, Dims, Mode, Target, IsPlh>>::type>::type
  get_native_mem(const accessor<DataT, Dims, Mode, Target, IsPlh> &) const {
    throw invalid_object_error("Getting memory object out of host accessor is "
                               "not allowed",
                               PI_INVALID_MEM_OBJECT);
  }

  /// Returns an underlying OpenCL queue for the SYCL queue used to submit the
  /// command group, or the fallback queue if this command-group is re-trying
  /// execution on an OpenCL queue. The OpenCL command queue returned is
  /// implementation-defined in cases where the SYCL queue maps to multiple
  /// underlying OpenCL objects. It is responsibility of the SYCL runtime to
  /// ensure the OpenCL queue returned is in a state that can be used to
  /// dispatch work, and that other potential OpenCL command queues associated
  /// with the same SYCL command queue are not executing commands while the host
  /// task is executing.
  template <backend BackendName = backend::opencl>
  auto get_native_queue() const noexcept ->
      typename interop<BackendName, queue>::type {
    return reinterpret_cast<typename interop<BackendName, queue>::type>(
        getNativeQueue());
  }

  /// Returns an underlying OpenCL device associated with the SYCL queue used
  /// to submit the command group, or the fallback queue if this command-group
  /// is re-trying execution on an OpenCL queue.
  template <backend BackendName = backend::opencl>
  auto get_native_device() const noexcept ->
      typename interop<BackendName, device>::type {
    return reinterpret_cast<typename interop<BackendName, device>::type>(
        getNativeDevice());
  }

  /// Returns an underlying OpenCL context associated with the SYCL queue used
  /// to submit the command group, or the fallback queue if this command-group
  /// is re-trying execution on an OpenCL queue.
  template <backend BackendName = backend::opencl>
  auto get_native_context() const noexcept ->
      typename interop<BackendName, context>::type {
    return reinterpret_cast<typename interop<BackendName, context>::type>(
        getNativeContext());
  }

private:
  using ReqToMem = std::pair<detail::Requirement *, pi_mem>;

public:
  // TODO set c-tor private
  interop_handle(std::vector<ReqToMem> MemObjs,
                 const std::shared_ptr<detail::queue_impl> &Queue,
                 const std::shared_ptr<detail::device_impl> &Device,
                 const std::shared_ptr<detail::context_impl> &Context)
      : MQueue(Queue), MDevice(Device), MContext(Context),
        MMemObjs(std::move(MemObjs)) {}

private:
  template <backend BackendName, typename DataT, int Dims, access::mode Mode,
            access::target Target, access::placeholder IsPlh>
  auto getMemImpl(detail::Requirement *Req) const ->
      typename interop<BackendName,
                       accessor<DataT, Dims, Mode, Target, IsPlh>>::type {
    return reinterpret_cast<typename interop<
        BackendName, accessor<DataT, Dims, Mode, Target, IsPlh>>::type>(
        getNativeMem(Req));
  }

  pi_native_handle getNativeMem(detail::Requirement *Req) const;
  pi_native_handle getNativeQueue() const;
  pi_native_handle getNativeDevice() const;
  pi_native_handle getNativeContext() const;

  std::shared_ptr<detail::queue_impl> MQueue;
  std::shared_ptr<detail::device_impl> MDevice;
  std::shared_ptr<detail::context_impl> MContext;

  std::vector<ReqToMem> MMemObjs;
};

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
