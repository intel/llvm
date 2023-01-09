//==------------ QueueApiFailures.cpp --- XPTI integration unit tests------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <helpers/PiMock.hpp>
#include <helpers/ScopedEnvVar.hpp>
#include <helpers/TestKernel.hpp>

#include <detail/xpti_registry.hpp>

#include <gtest/gtest.h>

using namespace sycl;     

pi_result redefinedEnqueueKernelLaunch(pi_queue queue, pi_kernel kernel, pi_uint32 work_dim,
    const size_t *global_work_offset, const size_t *global_work_size,
    const size_t *local_work_size, pi_uint32 num_events_in_wait_list,
    const pi_event *event_wait_list, pi_event *event) {
                                      std::cout << "redefined called" << std::endl;
  return PI_ERROR_INVALID_VALUE;
}

bool success {false};

XPTI_CALLBACK_API void testCallback(uint16_t TraceType,
                                  xpti::trace_event_data_t * /*Parent*/,
                                  xpti::trace_event_data_t * /*Event*/,
                                  uint64_t /*Instance*/, const void *UserData) {
  if (TraceType == xpti::trace_diagnostics) {
    const char* message = static_cast<const char*>(UserData);
    std::cout << message << std::endl;
    success = true;
  }
  else
    success = false;
}

void RegisterXPTIHandler()
{
  uint16_t StreamID = xptiRegisterStream(sycl::detail::SYCL_SYCLCALL_STREAM_NAME);
  xptiRegisterCallback(StreamID, xpti::trace_diagnostics,
                         testCallback);
}

TEST(QueueApiFailures, QueueCreation) {
  unittest::ScopedEnvVar XPTIenabling{
      "XPTI_TRACE_ENABLE", "1",[]{}};

  RegisterXPTIHandler();

  sycl::unittest::PiMock Mock;
  Mock.redefineAfter<detail::PiApiKind::piEnqueueKernelLaunch>(
      redefinedEnqueueKernelLaunch);

  sycl::queue Q;
  buffer<int, 1> buf{range<1>(1)};
  Q.submit([&](handler &Cgh) {
    auto Acc = buf.template get_access<access::mode::read_write>(Cgh);
    constexpr size_t KS = sizeof(decltype(Acc));
    Cgh.single_task<TestKernel<KS>>([=]() { Acc[0] = 4; });
  });
  Q.wait();

  EXPECT_TRUE(success);
}