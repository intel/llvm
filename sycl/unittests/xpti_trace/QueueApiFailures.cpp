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
XPTI_CALLBACK_API bool queryReceivedNotifications(uint16_t &TraceType,
                                                  std::string &Message);

inline pi_result redefinedPluginGetLastError(char **message) {
  return PI_ERROR_INVALID_VALUE;
}

pi_result redefinedEnqueueKernelLaunch(
    pi_queue queue, pi_kernel kernel, pi_uint32 work_dim,
    const size_t *global_work_offset, const size_t *global_work_size,
    const size_t *local_work_size, pi_uint32 num_events_in_wait_list,
    const pi_event *event_wait_list, pi_event *event) {
  std::cout << "redefined called" << std::endl;
  return PI_ERROR_PLUGIN_SPECIFIC_ERROR;
}

TEST(QueueApiFailures, QueueSubmit) {
  unittest::ScopedEnvVar XPTIenabling{"XPTI_TRACE_ENABLE", "1", [] {}};
  unittest::ScopedEnvVar PathToXPTIFW{"XPTI_FRAMEWORK_DISPATCHER",
                                      "libxptifw.so", [] {}};
  unittest::ScopedEnvVar XPTISubscriber{"XPTI_SUBSCRIBERS",
                                        "libxptitest_subscriber.so", [] {}};
  std::cout << "vars is set" << std::endl;

  xptiTraceTryToEnable();

  sycl::unittest::PiMock Mock;
  Mock.redefine<detail::PiApiKind::piEnqueueKernelLaunch>(
      redefinedEnqueueKernelLaunch);
  Mock.redefine<detail::PiApiKind::piPluginGetLastError>(
      redefinedPluginGetLastError);

  sycl::queue Q;
  buffer<int, 1> buf{range<1>(1)};
  try {
    Q.submit([&](handler &Cgh) {
      auto Acc = buf.template get_access<access::mode::read_write>(Cgh);
      constexpr size_t KS = sizeof(decltype(Acc));
      Cgh.single_task<TestKernel<KS>>([=]() { Acc[0] = 4; });
    });
  } catch (...) {
  }
  Q.wait();
  uint16_t TraceType = 0;
  std::string Message;
  EXPECT_TRUE(queryReceivedNotifications(TraceType, Message));
  EXPECT_EQ(TraceType, xpti::trace_diagnostics);
  EXPECT_EQ(TraceType, xpti::trace_diagnostics);
  EXPECT_EQ(Message, "No code location data is available.");
  EXPECT_FALSE(queryReceivedNotifications(TraceType, Message));
}