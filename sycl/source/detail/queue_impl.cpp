//==------------------ queue_impl.cpp - SYCL queue -------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/context.hpp>
#include <CL/sycl/detail/clusm.hpp>
#include <CL/sycl/detail/pi.hpp>
#include <CL/sycl/detail/queue_impl.hpp>
#include <CL/sycl/device.hpp>

namespace cl {
namespace sycl {
namespace detail {
template <> cl_uint queue_impl::get_info<info::queue::reference_count>() const {
  RT::PiResult result = PI_SUCCESS;
  if (!is_host())
    PI_CALL(RT::piQueueGetInfo(m_CommandQueue,
                               PI_QUEUE_INFO_REFERENCE_COUNT,
                               sizeof(result), &result, nullptr));
  return result;
}

template <> context queue_impl::get_info<info::queue::context>() const {
  return get_context();
}

template <> device queue_impl::get_info<info::queue::device>() const {
  return get_device();
}

// TODO: Update with PI interfaces
event queue_impl::memset(void* ptr, int value, size_t count) {
  cl_event e;
  cl_int error;
  cl_command_queue q = pi::cast<cl_command_queue>(getHandleRef());

  error = clEnqueueMemsetINTEL(q, ptr, value, count,
                               /* sizeof waitlist */ 0, nullptr, &e);

  CHECK_OCL_CODE_THROW(error, runtime_error);

  return event(e, get_context());
}

event queue_impl::memcpy(void* dest, const void* src, size_t count) {
  cl_event e;
  cl_int error;
  cl_command_queue q = pi::cast<cl_command_queue>(getHandleRef());

  error = clEnqueueMemcpyINTEL(q,
                               /* blocking */ false, dest, src, count,
                               /* sizeof waitlist */ 0, nullptr, &e);

  CHECK_OCL_CODE_THROW(error, runtime_error);

  return event(e, get_context());
}
} // namespace detail
} // namespace sycl
} // namespace cl
