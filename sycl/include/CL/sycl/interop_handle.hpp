//==------------ interop_handle.hpp --- SYCL interop handle ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/access/access.hpp>
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/defines.hpp>
#include <CL/sycl/detail/pi.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

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
                          pi_mem>::type
  get_native_mem(const accessor<dataT, dimensions, accessmode,
                                accessTarget, isPlaceholder> &Acc) const {
    auto *AccBase = static_cast<detail::AccessorBaseHost *>(&Acc);
    return getMemImpl(detail::getSyclObjImpl(*AccBase).get());
  }

  template <typename dataT, int dimensions, access::mode accessmode,
            access::target accessTarget, access::placeholder isPlaceholder>
  typename std::enable_if<accessTarget == access::target::host_buffer, 
                          pi_mem>::type
  get_native_mem(const accessor<dataT, dimensions, accessmode,
                                accessTarget, isPlaceholder> &Acc) const {
    throw invalid_object_error("Getting memory object out of host accessor is "
                               "not allowed", PI_INVALID_MEM_OBJECT);
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
  pi_queue get_native_queue() const noexcept {
    return MQueue;
  }

  /// Returns an underlying OpenCL device associated with the SYCL queue used
  /// to submit the command group, or the fallback queue if this command-group
  /// is re-trying execution on an OpenCL queue.
  cl_device_id get_native_device() const noexcept {
    return MDeviceId;
  }

  /// Returns an underlying OpenCL context associated with the SYCL queue used
  /// to submit the command group, or the fallback queue if this command-group
  /// is re-trying execution on an OpenCL queue.
  pi_context get_native_context() const noexcept {
    return MContext;
  }

private:
  using ReqToMem = std::pair<detail::Requirement*, pi_mem>;

  template <typename DataT, int Dims, access::mode AccMode,
            access::target AccTarget, access::placeholder isPlaceholder>
  friend class accessor;


  interop_handle(std::vector<ReqToMem> MemObjs, pi_queue Queue,
                 cl_device_id DeviceId, pi_context Context)
      : MQueue(Queue), MDeviceId(DeviceId),
        MContext(Context), MMemObjs(std::move(MemObjs)) {}

  pi_mem getMemImpl(detail::Requirement* Req) const;

  pi_queue MQueue;
  cl_device_id MDeviceId;
  pi_context MContext;
  std::vector<ReqToMem> MMemObjs;
};

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
