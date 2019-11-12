//==-------------- ordered_queue.cpp ---------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/exception_list.hpp>
#include <CL/sycl/ordered_queue.hpp>

#include <algorithm>

namespace cl {
namespace sycl {
ordered_queue::ordered_queue(const context &syclContext,
                             const device_selector &deviceSelector,
                             const async_handler &asyncHandler,
                             const property_list &propList) {

  const vector_class<device> Devs = syclContext.get_devices();

  auto Comp = [&deviceSelector](const device &d1, const device &d2) {
    return deviceSelector(d1) < deviceSelector(d2);
  };

  const device &syclDevice = *std::max_element(Devs.begin(), Devs.end(), Comp);
  impl = std::make_shared<detail::queue_impl>(
      syclDevice, syclContext, asyncHandler,
      cl::sycl::detail::QueueOrder::Ordered, propList);
}

ordered_queue::ordered_queue(const device &syclDevice,
                             const async_handler &asyncHandler,
                             const property_list &propList) {
  impl = std::make_shared<detail::queue_impl>(
      syclDevice, asyncHandler, cl::sycl::detail::QueueOrder::Ordered,
      propList);
}

ordered_queue::ordered_queue(cl_command_queue clQueue,
                             const context &syclContext,
                             const async_handler &asyncHandler) {
  cl_command_queue_properties reportedProps;
  RT::PiQueue m_CommandQueue = detail::pi::cast<detail::RT::PiQueue>(clQueue);
  PI_CALL(RT::piQueueGetInfo, m_CommandQueue, PI_QUEUE_INFO_DEVICE,
          sizeof(reportedProps), &reportedProps, nullptr);
  if (reportedProps & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)
    throw runtime_error(
        "Failed to build a sycl ordered queue from a cl OOO queue.");

  impl =
      std::make_shared<detail::queue_impl>(clQueue, syclContext, asyncHandler);
}

} // namespace sycl
} // namespace cl
