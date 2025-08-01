//==------------ QueueApiFailures.cpp --- XPTI integration unit tests ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <helpers/ScopedEnvVar.hpp>
#include <helpers/TestKernel.hpp>
#include <helpers/UrMock.hpp>

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

inline ur_result_t redefinedAdapterGetLastError(void *) {
  return UR_RESULT_ERROR_INVALID_VALUE;
}

ur_result_t redefinedEnqueueKernelLaunchWithArgsExp(void *) {
  return UR_RESULT_ERROR_ADAPTER_SPECIFIC;
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
  sycl::unittest::UrMock<> MockAdapter;

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
  using TestKI = detail::KernelInfo<TestKernel>;

  const std::string TestCodeLocationMessage = BuildCodeLocationMessage(
      FileName, FunctionName, LineNumber, ColumnNumber);
  const std::string ExtraTestCodeLocationMessage = BuildCodeLocationMessage(
      FileName, ExtraFunctionName, ExtraLineNumber, ColumnNumber);
  const std::string TestKernelLocationMessage = BuildCodeLocationMessage(
      TestKI::getFileName(), TestKI::getFunctionName(), TestKI::getLineNumber(),
      TestKI::getColumnNumber());
  const std::string URLevelFailMessage = " backend failed with error: ";
  const std::string SYCLLevelFailMessage = "Enqueue process failed";
};

TEST_F(QueueApiFailures, QueueSubmit) {
  mock::getCallbacks().set_replace_callback(
      "urEnqueueKernelLaunchWithArgsExp",
      &redefinedEnqueueKernelLaunchWithArgsExp);
  mock::getCallbacks().set_replace_callback("urAdapterGetLastError",
                                            &redefinedAdapterGetLastError);
  sycl::queue Q;
  bool ExceptionCaught = false;
  try {
    Q.submit([&](handler &Cgh) { Cgh.single_task<TestKernel>([=]() {}); },
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
  mock::getCallbacks().set_replace_callback(
      "urEnqueueKernelLaunchWithArgsExp",
      &redefinedEnqueueKernelLaunchWithArgsExp);
  mock::getCallbacks().set_replace_callback("urAdapterGetLastError",
                                            &redefinedAdapterGetLastError);
  sycl::queue Q;
  bool ExceptionCaught = false;
  try {
    Q.single_task<TestKernel>([=]() {}, TestCodeLocation);
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

ur_result_t redefinedEnqueueUSMFill(void *) {
  return UR_RESULT_ERROR_ADAPTER_SPECIFIC;
}

TEST_F(QueueApiFailures, QueueMemset) {
  mock::getCallbacks().set_replace_callback("urEnqueueUSMFill",
                                            &redefinedEnqueueUSMFill);
  mock::getCallbacks().set_replace_callback("urAdapterGetLastError",
                                            &redefinedAdapterGetLastError);
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

ur_result_t redefinedUSMEnqueueMemcpy(void *) {
  return UR_RESULT_ERROR_ADAPTER_SPECIFIC;
}

TEST_F(QueueApiFailures, QueueMemcpy) {
  mock::getCallbacks().set_replace_callback("urEnqueueUSMMemcpy",
                                            &redefinedUSMEnqueueMemcpy);
  mock::getCallbacks().set_replace_callback("urAdapterGetLastError",
                                            &redefinedAdapterGetLastError);
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
  mock::getCallbacks().set_replace_callback("urEnqueueUSMMemcpy",
                                            &redefinedUSMEnqueueMemcpy);
  mock::getCallbacks().set_replace_callback("urAdapterGetLastError",
                                            &redefinedAdapterGetLastError);
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

TEST_F(QueueApiFailures, QueueFill) {
  mock::getCallbacks().set_replace_callback("urEnqueueUSMFill",
                                            &redefinedEnqueueUSMFill);
  mock::getCallbacks().set_replace_callback("urAdapterGetLastError",
                                            &redefinedAdapterGetLastError);
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
  EXPECT_THAT(Message, HasSubstr(URLevelFailMessage));
  ASSERT_TRUE(queryReceivedNotifications(TraceType, Message));
  EXPECT_EQ(TraceType, xpti::trace_diagnostics);
  EXPECT_THAT(Message, HasSubstr(TestCodeLocationMessage));
  EXPECT_THAT(Message, HasSubstr(SYCLLevelFailMessage));
  EXPECT_FALSE(queryReceivedNotifications(TraceType, Message));
}

inline ur_result_t redefinedUSMEnqueuePrefetch(void *) {
  return UR_RESULT_ERROR_ADAPTER_SPECIFIC;
}

TEST_F(QueueApiFailures, QueuePrefetch) {
  mock::getCallbacks().set_replace_callback("urEnqueueUSMPrefetch",
                                            &redefinedUSMEnqueuePrefetch);
  mock::getCallbacks().set_replace_callback("urAdapterGetLastError",
                                            &redefinedAdapterGetLastError);
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
  EXPECT_THAT(Message, HasSubstr(URLevelFailMessage));
  ASSERT_TRUE(queryReceivedNotifications(TraceType, Message));
  EXPECT_EQ(TraceType, xpti::trace_diagnostics);
  EXPECT_THAT(Message, HasSubstr(TestCodeLocationMessage));
  EXPECT_THAT(Message, HasSubstr(SYCLLevelFailMessage));
  EXPECT_FALSE(queryReceivedNotifications(TraceType, Message));
}

inline ur_result_t redefinedUSMEnqueueMemAdvise(void *) {
  return UR_RESULT_ERROR_ADAPTER_SPECIFIC;
}

TEST_F(QueueApiFailures, QueueMemAdvise) {
  mock::getCallbacks().set_replace_callback("urEnqueueUSMAdvise",
                                            &redefinedUSMEnqueueMemAdvise);
  mock::getCallbacks().set_replace_callback("urAdapterGetLastError",
                                            &redefinedAdapterGetLastError);
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
  mock::getCallbacks().set_replace_callback(
      "urEnqueueKernelLaunchWithArgsExp",
      &redefinedEnqueueKernelLaunchWithArgsExp);
  mock::getCallbacks().set_replace_callback("urAdapterGetLastError",
                                            &redefinedAdapterGetLastError);
  sycl::queue Q;
  bool ExceptionCaught = false;
  const int globalWIs{512};
  try {
    Q.parallel_for<TestKernel>(globalWIs, [=](sycl::id<1> idx) {});
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

inline ur_result_t redefinedEventWait(void *) {
  return UR_RESULT_ERROR_ADAPTER_SPECIFIC;
}

inline void silentAsyncHandler(exception_list Exceptions) {
  std::ignore = Exceptions;
}

TEST_F(QueueApiFailures, QueueHostTaskWaitFail) {
  mock::getCallbacks().set_replace_callback("urEventWait", &redefinedEventWait);
  mock::getCallbacks().set_replace_callback("urAdapterGetLastError",
                                            &redefinedAdapterGetLastError);
  sycl::queue Q(default_selector(), silentAsyncHandler);
  bool ExceptionCaught = false;
  event EventToDepend;
  try {
    EventToDepend = Q.single_task<TestKernel>([=]() {}, TestCodeLocation);
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
  mock::getCallbacks().set_replace_callback("urAdapterGetLastError",
                                            &redefinedAdapterGetLastError);
  enum ExceptionType { STD_EXCEPTION = 0, SYCL_EXCEPTION };
  auto Test = [&](ExceptionType ExType) {
    sycl::queue Q(default_selector(), silentAsyncHandler);
    bool ExceptionCaught = false;
    event EventToDepend;
    const std::string HostTaskExeptionStr = "Host task exception";
    try {
      EventToDepend = Q.single_task<TestKernel>([=]() {}, TestCodeLocation);
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

ur_result_t redefinedEnqueueKernelLaunchWithStatus(void *) {
  {
    std::lock_guard<std::mutex> lk(m);
    EnqueueKernelLaunchCalled = true;
  }
  cv.notify_one();
  return UR_RESULT_ERROR_ADAPTER_SPECIFIC;
}

TEST_F(QueueApiFailures, QueueKernelAsync) {
  mock::getCallbacks().set_replace_callback(
      "urEnqueueKernelLaunchWithArgsExp",
      &redefinedEnqueueKernelLaunchWithStatus);
  mock::getCallbacks().set_replace_callback("urAdapterGetLastError",
                                            &redefinedAdapterGetLastError);

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
          Cgh.single_task<TestKernel>([=]() {});
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
