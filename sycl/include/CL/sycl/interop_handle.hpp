//==------------ interop_handle.hpp --- SYCL interop handle ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/access/access.hpp>
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
} // namespace detail

template <typename DataT, int Dims, access::mode AccMode,
          access::target AccTarget, access::placeholder isPlaceholder>
class accessor;

class interop_handle {
public:
  /// Receives a SYCL accessor that has been defined is a requirement for the
  /// command group, and returns the underlying OpenCL memory object that is
  /// used by the SYCL runtime. If the accessor passed as parameter is not part
  /// of the command group requirements (e.g. it is an unregistered placeholder
  /// accessor), the exception `cl::sycl::invalid_object` is thrown
  /// asynchronously.
  template <typename dataT, int dimensions, access::mode accessmode,
            access::target accessTarget, access::placeholder isPlaceholder>
  typename std::enable_if<accessTarget != access::target::host_buffer,
                          cl_mem>::type
  get_native_mem(const accessor<dataT, dimensions, accessmode, accessTarget,
                                isPlaceholder> &Acc) const {
#ifndef __SYCL_DEVICE_ONLY__
    // employ reinterpret_cast instead of static_cast due to cycle in includes
    // involving CL/sycl/accessor.hpp
    auto *AccBase = const_cast<detail::AccessorBaseHost *>(
        reinterpret_cast<const detail::AccessorBaseHost *>(&Acc));
    return getMemImpl(detail::getSyclObjImpl(*AccBase).get());
#else
    (void)Acc;
    // we believe this won't be ever called on device side
    return static_cast<cl_mem>(0x0);
#endif
  }

  template <typename dataT, int dimensions, access::mode accessmode,
            access::target accessTarget, access::placeholder isPlaceholder>
  typename std::enable_if<accessTarget == access::target::host_buffer,
                          cl_mem>::type
  get_native_mem(const accessor<dataT, dimensions, accessmode, accessTarget,
                                isPlaceholder> &) const {
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
  cl_command_queue get_native_queue() const noexcept { return MQueue; }

  /// Returns an underlying OpenCL device associated with the SYCL queue used
  /// to submit the command group, or the fallback queue if this command-group
  /// is re-trying execution on an OpenCL queue.
  cl_device_id get_native_device() const noexcept { return MDeviceId; }

  /// Returns an underlying OpenCL context associated with the SYCL queue used
  /// to submit the command group, or the fallback queue if this command-group
  /// is re-trying execution on an OpenCL queue.
  cl_context get_native_context() const noexcept { return MContext; }

private:
  using ReqToMem = std::pair<detail::Requirement *, pi_mem>;

public:
  // TODO set c-tor private
  interop_handle(std::vector<ReqToMem> MemObjs, cl_command_queue Queue,
                 cl_device_id DeviceId, cl_context Context)
      : MQueue(Queue), MDeviceId(DeviceId), MContext(Context),
        MMemObjs(std::move(MemObjs)) {}

private:
  cl_mem getMemImpl(detail::Requirement *Req) const;

  cl_command_queue MQueue;
  cl_device_id MDeviceId;
  cl_context MContext;
  std::vector<ReqToMem> MMemObjs;
};

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
