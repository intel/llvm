//==------------ QueueApiFailures.cpp --- XPTI integration unit tests ------==//
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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <sycl/sycl.hpp>

#include <condition_variable>

using ::testing::HasSubstr;
using namespace sycl;
XPTI_CALLBACK_API bool queryReceivedNotifications(uint16_t &TraceType,
                                                  std::string &Message);
XPTI_CALLBACK_API void resetReceivedNotifications();
XPTI_CALLBACK_API void addAnalyzedTraceType(uint16_t);

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
  void SetUp() {
    xptiForceSetTraceEnabled(true);
    xptiTraceTryToEnable();
    addAnalyzedTraceType(xpti::trace_diagnostics);
  }

  void TearDown() {
    resetReceivedNotifications();
    xptiForceSetTraceEnabled(false);
  }

public:
  static std::string BuildCodeLocationMessage(const char *FileName,
                                              const char *FunctionName,
                                              int LineNumber,
                                              int ColumnNumber) {
    const char Delimiter[] = ";";
    std::string Result(FunctionName);
    Result.append(Delimiter);
    Result.append(FileName);
    Result.append(Delimiter);
    Result.append(std::to_string(LineNumber) + Delimiter +
                  std::to_string(ColumnNumber));
    return Result;
  }

  unittest::ScopedEnvVar PathToXPTIFW{"XPTI_FRAMEWORK_DISPATCHER",
                                      "libxptifw.so", [] {}};
  unittest::ScopedEnvVar XPTISubscriber{"XPTI_SUBSCRIBERS",
                                        "libxptitest_subscriber.so", [] {}};
  sycl::unittest::PiMock MockPlugin;

  static constexpr char FileName[] = "QueueApiFailures.cpp";
  static constexpr char FunctionName[] = "TestCaseExecution";
  static constexpr char ExtraFunctionName[] = "ExtraTestCaseExecution";
  static constexpr int LineNumber = 8;
  static constexpr int ExtraLineNumber = 7;
  static constexpr int ColumnNumber = 13;
  const sycl::detail::code_location TestCodeLocation = {
      FileName, FunctionName, LineNumber, ColumnNumber};
  const sycl::detail::code_location ExtraTestCodeLocation = {
      FileName, ExtraFunctionName, ExtraLineNumber, ColumnNumber};
  static constexpr size_t KernelSize = 1;
  using TestKI = detail::KernelInfo<TestKernel<KernelSize>>;

  const std::string TestCodeLocationMessage = BuildCodeLocationMessage(
      FileName, FunctionName, LineNumber, ColumnNumber);
  const std::string ExtraTestCodeLocationMessage = BuildCodeLocationMessage(
      FileName, ExtraFunctionName, ExtraLineNumber, ColumnNumber);
  const std::string TestKernelLocationMessage = BuildCodeLocationMessage(
      TestKI::getFileName(), TestKI::getFunctionName(), TestKI::getLineNumber(),
      TestKI::getColumnNumber());
  const std::string PiLevelFailMessage = "Native API failed";
};

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
    std::ignore = e;
    ExceptionCaught = true;
  }
  Q.wait();
  ASSERT_TRUE(ExceptionCaught);
  uint16_t TraceType = 0;
  std::string Message;
  EXPECT_TRUE(queryReceivedNotifications(TraceType, Message));
  EXPECT_EQ(TraceType, xpti::trace_diagnostics);
  EXPECT_THAT(Message, HasSubstr(TestCodeLocationMessage));
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
    std::ignore = e;
    ExceptionCaught = true;
  }
  Q.wait();
  ASSERT_TRUE(ExceptionCaught);
  uint16_t TraceType = 0;
  std::string Message;
  EXPECT_TRUE(queryReceivedNotifications(TraceType, Message));
  EXPECT_EQ(TraceType, xpti::trace_diagnostics);
  EXPECT_THAT(Message, HasSubstr(TestCodeLocationMessage));
  EXPECT_FALSE(queryReceivedNotifications(TraceType, Message));
}

pi_result redefinedUSMEnqueueMemset(pi_queue Queue, void *Ptr,
                                    const void *Pattern, size_t PatternSize,
                                    size_t Count,
                                    pi_uint32 Num_events_in_waitlist,
                                    const pi_event *Events_waitlist,
                                    pi_event *Event) {
  return PI_ERROR_PLUGIN_SPECIFIC_ERROR;
}

TEST_F(QueueApiFailures, QueueMemset) {
  MockPlugin.redefine<detail::PiApiKind::piextUSMEnqueueFill>(
      redefinedUSMEnqueueMemset);
  MockPlugin.redefine<detail::PiApiKind::piPluginGetLastError>(
      redefinedPluginGetLastError);
  sycl::queue Q;
  bool ExceptionCaught = false;
  unsigned char *HostAlloc = (unsigned char *)sycl::malloc_host(1, Q);
  try {
    Q.memset(HostAlloc, 42, 1, TestCodeLocation);
  } catch (sycl::exception &e) {
    std::ignore = e;
    ExceptionCaught = true;
  }
  Q.wait();
  sycl::free(HostAlloc, Q);
  ASSERT_TRUE(ExceptionCaught);
  uint16_t TraceType = 0;
  std::string Message;
  ASSERT_TRUE(queryReceivedNotifications(TraceType, Message));
  EXPECT_EQ(TraceType, xpti::trace_diagnostics);
  EXPECT_THAT(Message, HasSubstr(TestCodeLocationMessage));
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
    Q.memcpy(HostAllocDst, HostAllocSrc, 1, TestCodeLocation);
  } catch (sycl::exception &e) {
    std::ignore = e;
    ExceptionCaught = true;
  }
  Q.wait();
  sycl::free(HostAllocSrc, Q);
  sycl::free(HostAllocDst, Q);
  ASSERT_TRUE(ExceptionCaught);
  uint16_t TraceType = 0;
  std::string Message;
  ASSERT_TRUE(queryReceivedNotifications(TraceType, Message));
  EXPECT_EQ(TraceType, xpti::trace_diagnostics);
  EXPECT_THAT(Message, HasSubstr(TestCodeLocationMessage));
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
    std::ignore = e;
    ExceptionCaught = true;
  }
  Q.wait();
  sycl::free(HostAllocSrc, Q);
  sycl::free(HostAllocDst, Q);
  ASSERT_TRUE(ExceptionCaught);
  uint16_t TraceType = 0;
  std::string Message;
  ASSERT_TRUE(queryReceivedNotifications(TraceType, Message));
  EXPECT_EQ(TraceType, xpti::trace_diagnostics);
  EXPECT_THAT(Message, HasSubstr(TestCodeLocationMessage));
  EXPECT_FALSE(queryReceivedNotifications(TraceType, Message));
}

pi_result redefinedUSMEnqueueFill(pi_queue Queue, void *Ptr,
                                  const void *Pattern, size_t PatternSize,
                                  size_t Count, pi_uint32 NumEventsInWaitList,
                                  const pi_event *EventWaitList,
                                  pi_event *Event) {
  return PI_ERROR_PLUGIN_SPECIFIC_ERROR;
}

TEST_F(QueueApiFailures, QueueFill) {
  MockPlugin.redefine<detail::PiApiKind::piextUSMEnqueueFill>(
      redefinedUSMEnqueueFill);
  MockPlugin.redefine<detail::PiApiKind::piPluginGetLastError>(
      redefinedPluginGetLastError);
  sycl::queue Q;
  bool ExceptionCaught = false;
  unsigned char *HostAlloc = (unsigned char *)sycl::malloc_host(1, Q);
  try {
    Q.fill(HostAlloc, 42, 1, TestCodeLocation);
  } catch (sycl::exception &e) {
    std::ignore = e;
    ExceptionCaught = true;
  }
  Q.wait();
  sycl::free(HostAlloc, Q);
  ASSERT_TRUE(ExceptionCaught);
  uint16_t TraceType = 0;
  std::string Message;
  ASSERT_TRUE(queryReceivedNotifications(TraceType, Message));
  EXPECT_EQ(TraceType, xpti::trace_diagnostics);
  EXPECT_THAT(Message, HasSubstr(TestCodeLocationMessage));
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
    Q.prefetch(HostAlloc, 2, TestCodeLocation);
  } catch (sycl::exception &e) {
    std::ignore = e;
    ExceptionCaught = true;
  }
  Q.wait();
  sycl::free(HostAlloc, Q);
  ASSERT_TRUE(ExceptionCaught);
  uint16_t TraceType = 0;
  std::string Message;
  ASSERT_TRUE(queryReceivedNotifications(TraceType, Message));
  EXPECT_EQ(TraceType, xpti::trace_diagnostics);
  EXPECT_THAT(Message, HasSubstr(TestCodeLocationMessage));
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
    Q.mem_advise(HostAlloc, 1, 0 /*default*/, TestCodeLocation);
  } catch (sycl::exception &e) {
    std::ignore = e;
    ExceptionCaught = true;
  }
  Q.wait();
  sycl::free(HostAlloc, Q);
  ASSERT_TRUE(ExceptionCaught);
  uint16_t TraceType = 0;
  std::string Message;
  ASSERT_TRUE(queryReceivedNotifications(TraceType, Message));
  EXPECT_EQ(TraceType, xpti::trace_diagnostics);
  EXPECT_THAT(Message, HasSubstr(TestCodeLocationMessage));
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
    std::ignore = e;
    ExceptionCaught = true;
  }
  Q.wait();
  ASSERT_TRUE(ExceptionCaught);
  uint16_t TraceType = 0;
  std::string Message;
  EXPECT_TRUE(queryReceivedNotifications(TraceType, Message));
  EXPECT_EQ(TraceType, xpti::trace_diagnostics);
  EXPECT_THAT(Message, HasSubstr(TestKernelLocationMessage));
  EXPECT_FALSE(queryReceivedNotifications(TraceType, Message));
}

inline pi_result redefinedEventsWait(pi_uint32 num_events,
                                     const pi_event *event_list) {
  return PI_ERROR_PLUGIN_SPECIFIC_ERROR;
}

inline void silentAsyncHandler(exception_list Exceptions) {
  std::ignore = Exceptions;
}

TEST_F(QueueApiFailures, QueueHostTaskWaitFail) {
  MockPlugin.redefine<detail::PiApiKind::piEventsWait>(redefinedEventsWait);
  MockPlugin.redefine<detail::PiApiKind::piPluginGetLastError>(
      redefinedPluginGetLastError);
  sycl::queue Q(default_selector(), silentAsyncHandler);
  bool ExceptionCaught = false;
  event EventToDepend;
  try {
    EventToDepend =
        Q.single_task<TestKernel<KernelSize>>([=]() {}, TestCodeLocation);
  } catch (sycl::exception &e) {
    std::ignore = e;
    ExceptionCaught = true;
  }
  EXPECT_FALSE(ExceptionCaught);
  const std::string HostTaskExceptionStr = "Host task exception";
  try {
    Q.submit(
        [&](handler &Cgh) {
          Cgh.depends_on(EventToDepend);
          Cgh.host_task(
              [=]() { throw std::runtime_error(HostTaskExceptionStr); });
        },
        ExtraTestCodeLocation);
  } catch (sycl::exception &e) {
    std::ignore = e;
    ExceptionCaught = true;
  }
  EXPECT_FALSE(ExceptionCaught);
  Q.wait();

  uint16_t TraceType = 0;
  std::string Message;
  EXPECT_TRUE(queryReceivedNotifications(TraceType, Message));
  EXPECT_EQ(TraceType, xpti::trace_diagnostics);
  EXPECT_THAT(Message, HasSubstr(ExtraTestCodeLocationMessage));
  EXPECT_THAT(Message, Not(HasSubstr(HostTaskExceptionStr)));
}

TEST_F(QueueApiFailures, QueueHostTaskFail) {
  MockPlugin.redefine<detail::PiApiKind::piPluginGetLastError>(
      redefinedPluginGetLastError);
  enum ExceptionType { STD_EXCEPTION = 0, SYCL_EXCEPTION };
  auto Test = [&](ExceptionType ExType) {
    sycl::queue Q(default_selector(), silentAsyncHandler);
    bool ExceptionCaught = false;
    event EventToDepend;
    const std::string HostTaskExeptionStr = "Host task exception";
    try {
      EventToDepend =
          Q.single_task<TestKernel<KernelSize>>([=]() {}, TestCodeLocation);
    } catch (sycl::exception &e) {
      std::ignore = e;
      ExceptionCaught = true;
    }
    EXPECT_FALSE(ExceptionCaught);
    try {
      Q.submit(
          [&](handler &Cgh) {
            Cgh.depends_on(EventToDepend);
            Cgh.host_task([=]() {
              if (ExType == SYCL_EXCEPTION)
                throw sycl::exception(sycl::make_error_code(errc::invalid),
                                      HostTaskExeptionStr);
              else
                throw std::runtime_error(HostTaskExeptionStr);
            });
          },
          ExtraTestCodeLocation);
    } catch (sycl::exception &e) {
      std::ignore = e;
      ExceptionCaught = true;
    }
    EXPECT_FALSE(ExceptionCaught);
    Q.wait();

    uint16_t TraceType = 0;
    std::string Message;
    EXPECT_TRUE(queryReceivedNotifications(TraceType, Message));
    EXPECT_EQ(TraceType, xpti::trace_diagnostics);
    EXPECT_THAT(Message, HasSubstr(ExtraTestCodeLocationMessage));
    EXPECT_THAT(Message, HasSubstr(HostTaskExeptionStr));
    resetReceivedNotifications();
  };
  Test(SYCL_EXCEPTION);
  Test(STD_EXCEPTION);
}

std::mutex m;
std::condition_variable cv;
bool EnqueueKernelLaunchCalled = false;

pi_result redefinedEnqueueKernelLaunchWithStatus(
    pi_queue queue, pi_kernel kernel, pi_uint32 work_dim,
    const size_t *global_work_offset, const size_t *global_work_size,
    const size_t *local_work_size, pi_uint32 num_events_in_wait_list,
    const pi_event *event_wait_list, pi_event *event) {
  {
    std::lock_guard<std::mutex> lk(m);
    EnqueueKernelLaunchCalled = true;
  }
  cv.notify_one();
  return PI_ERROR_PLUGIN_SPECIFIC_ERROR;
}

TEST_F(QueueApiFailures, QueueKernelAsync) {
  MockPlugin.redefine<detail::PiApiKind::piEnqueueKernelLaunch>(
      redefinedEnqueueKernelLaunchWithStatus);
  MockPlugin.redefine<detail::PiApiKind::piPluginGetLastError>(
      redefinedPluginGetLastError);

  sycl::queue Q(default_selector(), silentAsyncHandler);
  bool ExceptionCaught = false;
  event EventToDepend;

  std::mutex m;
  std::function<void()> CustomHostLambda = [&m]() {
    std::unique_lock<std::mutex> InsideHostTaskLock(m);
  };
  std::unique_lock<std::mutex> TestLock(m, std::defer_lock);
  TestLock.lock();
  try {
    EventToDepend = Q.submit(
        [&](handler &Cgh) {
          Cgh.depends_on(EventToDepend);
          Cgh.host_task(CustomHostLambda);
        },
        TestCodeLocation);
  } catch (sycl::exception &e) {
    std::ignore = e;
    ExceptionCaught = true;
  }
  EXPECT_FALSE(ExceptionCaught);

  try {
    Q.submit(
        [&](handler &Cgh) {
          Cgh.depends_on(EventToDepend);
          Cgh.single_task<TestKernel<KernelSize>>([=]() {});
        },
        ExtraTestCodeLocation);
  } catch (sycl::exception &e) {
    std::ignore = e;
    ExceptionCaught = true;
  }
  EXPECT_FALSE(ExceptionCaught);
  TestLock.unlock();

  // Need to wait till host task enqueue kernel to check code location report.
  {
    std::unique_lock<std::mutex> lk(m);
    cv.wait(lk, [] { return EnqueueKernelLaunchCalled; });
  }

  try {
    Q.wait();
  } catch (...) {
    // kernel enqueue leads to exception throw
  }

  uint16_t TraceType = 0;
  std::string Message;
  EXPECT_TRUE(queryReceivedNotifications(TraceType, Message));
  EXPECT_EQ(TraceType, xpti::trace_diagnostics);
  EXPECT_THAT(Message, HasSubstr(ExtraTestCodeLocationMessage));
}
