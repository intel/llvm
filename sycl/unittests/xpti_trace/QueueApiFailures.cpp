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

#include <sycl/sycl.hpp>

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
  return PI_ERROR_PLUGIN_SPECIFIC_ERROR;
}

class QueueApiFailures : public ::testing::Test {
protected:
  void SetUp() { xptiTraceTryToEnable(); }

  void TearDown() {}

public:
  unittest::ScopedEnvVar XPTIenabling{"XPTI_TRACE_ENABLE", "1", [] {}};
  unittest::ScopedEnvVar PathToXPTIFW{"XPTI_FRAMEWORK_DISPATCHER",
                                      "libxptifw.so", [] {}};
  unittest::ScopedEnvVar XPTISubscriber{"XPTI_SUBSCRIBERS",
                                        "libxptitest_subscriber.so", [] {}};
  sycl::unittest::PiMock MockPlugin;

  static constexpr char FileName[] = "QueueApiFailures.cpp";
  static constexpr char FunctionName[] = "TestCaseExecution";
  static constexpr unsigned int LineNumber = 8;
  static constexpr unsigned int ColumnNumber = 13;
  static constexpr sycl::detail::code_location TestCodeLocation = {
      FileName, FunctionName, LineNumber, ColumnNumber};
  static const std::string TestCodeLocationMessage;
  static const std::string TestKernelLocationMessage;
  static const size_t KernelSize = 1;
  static constexpr char UnknownCodeLocation[] = "unknown";
};

const std::string QueueApiFailures::TestCodeLocationMessage = {
    std::string(FileName)
        .append(":")
        .append(FunctionName)
        .append(":ln")
        .append(std::to_string(LineNumber))
        .append(":col")
        .append(std::to_string(ColumnNumber))};
const std::string QueueApiFailures::TestKernelLocationMessage = {
    std::string(detail::KernelInfo<TestKernel<KernelSize>>::getFileName())
        .append(":")
        .append(detail::KernelInfo<TestKernel<KernelSize>>::getFunctionName())
        .append(":ln")
        .append(std::to_string(
            detail::KernelInfo<TestKernel<KernelSize>>::getLineNumber()))
        .append(":col")
        .append(std::to_string(
            detail::KernelInfo<TestKernel<KernelSize>>::getColumnNumber()))};

TEST_F(QueueApiFailures, QueueSubmit) {
  MockPlugin.redefine<detail::PiApiKind::piEnqueueKernelLaunch>(
      redefinedEnqueueKernelLaunch);
  MockPlugin.redefine<detail::PiApiKind::piPluginGetLastError>(
      redefinedPluginGetLastError);
  sycl::queue Q;
  bool ExceptionCaught = false;
  try {
    Q.submit(
        [&](handler &Cgh) {
          Cgh.single_task<TestKernel<KernelSize>>([=]() {});
        },
        TestCodeLocation);
  } catch (sycl::exception &e) {
    ExceptionCaught = true;
  }
  Q.wait();
  EXPECT_TRUE(ExceptionCaught);
  uint16_t TraceType = 0;
  std::string Message;
  EXPECT_TRUE(queryReceivedNotifications(TraceType, Message));
  EXPECT_EQ(TraceType, xpti::trace_diagnostics);
  EXPECT_TRUE(Message.find(TestCodeLocationMessage) == 0);
  EXPECT_FALSE(queryReceivedNotifications(TraceType, Message));
}

TEST_F(QueueApiFailures, QueueSingleTask) {
  MockPlugin.redefine<detail::PiApiKind::piEnqueueKernelLaunch>(
      redefinedEnqueueKernelLaunch);
  MockPlugin.redefine<detail::PiApiKind::piPluginGetLastError>(
      redefinedPluginGetLastError);
  sycl::queue Q;
  bool ExceptionCaught = false;
  try {
    Q.single_task<TestKernel<KernelSize>>([=]() {}, TestCodeLocation);
  } catch (sycl::exception &e) {
    ExceptionCaught = true;
  }
  Q.wait();
  EXPECT_TRUE(ExceptionCaught);
  uint16_t TraceType = 0;
  std::string Message;
  EXPECT_TRUE(queryReceivedNotifications(TraceType, Message));
  EXPECT_EQ(TraceType, xpti::trace_diagnostics);
  EXPECT_TRUE(Message.find(TestCodeLocationMessage) == 0);
  EXPECT_FALSE(queryReceivedNotifications(TraceType, Message));
}

pi_result redefinedUSMEnqueueMemset(pi_queue Queue, void *Ptr, pi_int32 Value,
                                    size_t Count,
                                    pi_uint32 Num_events_in_waitlist,
                                    const pi_event *Events_waitlist,
                                    pi_event *Event) {
  return PI_ERROR_PLUGIN_SPECIFIC_ERROR;
}

TEST_F(QueueApiFailures, QueueMemset) {
  MockPlugin.redefine<detail::PiApiKind::piextUSMEnqueueMemset>(
      redefinedUSMEnqueueMemset);
  MockPlugin.redefine<detail::PiApiKind::piPluginGetLastError>(
      redefinedPluginGetLastError);
  sycl::queue Q;
  bool ExceptionCaught = false;
  unsigned char *HostAlloc = (unsigned char *)sycl::malloc_host(1, Q);
  try {
    Q.memset(HostAlloc, 42, 1);
  } catch (sycl::exception &e) {
    ExceptionCaught = true;
  }
  Q.wait();
  sycl::free(HostAlloc, Q);
  EXPECT_TRUE(ExceptionCaught);
  uint16_t TraceType = 0;
  std::string Message;
  ASSERT_TRUE(queryReceivedNotifications(TraceType, Message));
  EXPECT_EQ(TraceType, xpti::trace_diagnostics);
  EXPECT_TRUE(Message.find(UnknownCodeLocation) == 0);
  EXPECT_FALSE(queryReceivedNotifications(TraceType, Message));
}

pi_result redefinedUSMEnqueueMemcpy(pi_queue queue, pi_bool blocking,
                                    void *dst_ptr, const void *src_ptr,
                                    size_t size,
                                    pi_uint32 num_events_in_waitlist,
                                    const pi_event *events_waitlist,
                                    pi_event *event) {
  return PI_ERROR_PLUGIN_SPECIFIC_ERROR;
}

TEST_F(QueueApiFailures, QueueMemcpy) {
  MockPlugin.redefine<detail::PiApiKind::piextUSMEnqueueMemcpy>(
      redefinedUSMEnqueueMemcpy);
  MockPlugin.redefine<detail::PiApiKind::piPluginGetLastError>(
      redefinedPluginGetLastError);
  sycl::queue Q;
  bool ExceptionCaught = false;
  unsigned char *HostAllocSrc = (unsigned char *)sycl::malloc_host(1, Q);
  unsigned char *HostAllocDst = (unsigned char *)sycl::malloc_host(1, Q);
  try {
    Q.memcpy(HostAllocDst, HostAllocSrc, 1);
  } catch (sycl::exception &e) {
    ExceptionCaught = true;
  }
  Q.wait();
  sycl::free(HostAllocSrc, Q);
  sycl::free(HostAllocDst, Q);
  EXPECT_TRUE(ExceptionCaught);
  uint16_t TraceType = 0;
  std::string Message;
  ASSERT_TRUE(queryReceivedNotifications(TraceType, Message));
  EXPECT_EQ(TraceType, xpti::trace_diagnostics);
  EXPECT_TRUE(Message.find(UnknownCodeLocation) == 0);
  EXPECT_FALSE(queryReceivedNotifications(TraceType, Message));
}

TEST_F(QueueApiFailures, QueueCopy) {
  MockPlugin.redefine<detail::PiApiKind::piextUSMEnqueueMemcpy>(
      redefinedUSMEnqueueMemcpy);
  MockPlugin.redefine<detail::PiApiKind::piPluginGetLastError>(
      redefinedPluginGetLastError);
  sycl::queue Q;
  bool ExceptionCaught = false;
  unsigned char *HostAllocSrc = (unsigned char *)sycl::malloc_host(1, Q);
  unsigned char *HostAllocDst = (unsigned char *)sycl::malloc_host(1, Q);
  try {
    Q.copy(HostAllocDst, HostAllocSrc, 1, TestCodeLocation);
  } catch (sycl::exception &e) {
    ExceptionCaught = true;
  }
  Q.wait();
  sycl::free(HostAllocSrc, Q);
  sycl::free(HostAllocDst, Q);
  EXPECT_TRUE(ExceptionCaught);
  uint16_t TraceType = 0;
  std::string Message;
  ASSERT_TRUE(queryReceivedNotifications(TraceType, Message));
  EXPECT_EQ(TraceType, xpti::trace_diagnostics);
  EXPECT_TRUE(Message.find(TestCodeLocationMessage) == 0);
  EXPECT_FALSE(queryReceivedNotifications(TraceType, Message));
}

pi_result redefinedEnqueueMemBufferFill(pi_queue Queue, pi_mem Buffer,
                                        const void *Pattern, size_t PatternSize,
                                        size_t Offset, size_t Size,
                                        pi_uint32 NumEventsInWaitList,
                                        const pi_event *EventWaitList,
                                        pi_event *Event) {
  return PI_ERROR_PLUGIN_SPECIFIC_ERROR;
}

TEST_F(QueueApiFailures, QueueFill) {
  MockPlugin.redefine<detail::PiApiKind::piEnqueueMemBufferFill>(
      redefinedEnqueueMemBufferFill);
  MockPlugin.redefine<detail::PiApiKind::piPluginGetLastError>(
      redefinedPluginGetLastError);
  sycl::queue Q;
  bool ExceptionCaught = false;
  unsigned char *HostAlloc = (unsigned char *)sycl::malloc_host(1, Q);
  try {
    Q.fill(HostAlloc, 42, 1);
  } catch (sycl::exception &e) {
    ExceptionCaught = true;
  }
  Q.wait();
  sycl::free(HostAlloc, Q);
  EXPECT_TRUE(ExceptionCaught);
  uint16_t TraceType = 0;
  std::string Message;
  ASSERT_TRUE(queryReceivedNotifications(TraceType, Message));
  EXPECT_EQ(TraceType, xpti::trace_diagnostics);
  EXPECT_TRUE(Message.find(UnknownCodeLocation) == 0);
  EXPECT_FALSE(queryReceivedNotifications(TraceType, Message));
}

inline pi_result redefinedUSMEnqueuePrefetch(pi_queue queue, const void *ptr,
                                             size_t size,
                                             pi_usm_migration_flags flags,
                                             pi_uint32 num_events_in_waitlist,
                                             const pi_event *events_waitlist,
                                             pi_event *event) {
  return PI_ERROR_PLUGIN_SPECIFIC_ERROR;
}

TEST_F(QueueApiFailures, QueuePrefetch) {
  MockPlugin.redefine<detail::PiApiKind::piextUSMEnqueuePrefetch>(
      redefinedUSMEnqueuePrefetch);
  MockPlugin.redefine<detail::PiApiKind::piPluginGetLastError>(
      redefinedPluginGetLastError);
  sycl::queue Q;
  bool ExceptionCaught = false;
  unsigned char *HostAlloc = (unsigned char *)sycl::malloc_host(4, Q);
  try {
    Q.prefetch(HostAlloc, 2);
  } catch (sycl::exception &e) {
    ExceptionCaught = true;
  }
  Q.wait();
  sycl::free(HostAlloc, Q);
  EXPECT_TRUE(ExceptionCaught);
  uint16_t TraceType = 0;
  std::string Message;
  ASSERT_TRUE(queryReceivedNotifications(TraceType, Message));
  EXPECT_EQ(TraceType, xpti::trace_diagnostics);
  EXPECT_TRUE(Message.find(UnknownCodeLocation) == 0);
  EXPECT_FALSE(queryReceivedNotifications(TraceType, Message));
}

inline pi_result redefinedUSMEnqueueMemAdvise(pi_queue queue, const void *ptr,
                                              size_t length,
                                              pi_mem_advice advice,
                                              pi_event *event) {
  return PI_ERROR_PLUGIN_SPECIFIC_ERROR;
}

TEST_F(QueueApiFailures, QueueMemAdvise) {
  MockPlugin.redefine<detail::PiApiKind::piextUSMEnqueueMemAdvise>(
      redefinedUSMEnqueueMemAdvise);
  MockPlugin.redefine<detail::PiApiKind::piPluginGetLastError>(
      redefinedPluginGetLastError);
  sycl::queue Q;
  bool ExceptionCaught = false;
  unsigned char *HostAlloc = (unsigned char *)sycl::malloc_host(1, Q);
  try {
    Q.mem_advise(HostAlloc, 1, 0 /*default*/);
  } catch (sycl::exception &e) {
    ExceptionCaught = true;
  }
  Q.wait();
  sycl::free(HostAlloc, Q);
  EXPECT_TRUE(ExceptionCaught);
  uint16_t TraceType = 0;
  std::string Message;
  ASSERT_TRUE(queryReceivedNotifications(TraceType, Message));
  EXPECT_EQ(TraceType, xpti::trace_diagnostics);
  EXPECT_TRUE(Message.find(UnknownCodeLocation) == 0);
  EXPECT_FALSE(queryReceivedNotifications(TraceType, Message));
}

TEST_F(QueueApiFailures, QueueParallelFor) {
  MockPlugin.redefine<detail::PiApiKind::piEnqueueKernelLaunch>(
      redefinedEnqueueKernelLaunch);
  MockPlugin.redefine<detail::PiApiKind::piPluginGetLastError>(
      redefinedPluginGetLastError);
  sycl::queue Q;
  bool ExceptionCaught = false;
  const int globalWIs{512};
  try {
    Q.parallel_for<TestKernel<KernelSize>>(globalWIs, [=](sycl::id<1> idx) {});
  } catch (sycl::exception &e) {
    ExceptionCaught = true;
  }
  Q.wait();
  EXPECT_TRUE(ExceptionCaught);
  uint16_t TraceType = 0;
  std::string Message;
  EXPECT_TRUE(queryReceivedNotifications(TraceType, Message));
  EXPECT_EQ(TraceType, xpti::trace_diagnostics);
  EXPECT_TRUE(Message.find(TestKernelLocationMessage) == 0);
  EXPECT_FALSE(queryReceivedNotifications(TraceType, Message));
}