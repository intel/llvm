//==------------ QueueIDCheck.cpp --- XPTI integration unit tests ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <helpers/ScopedEnvVar.hpp>
#include <helpers/TestKernel.hpp>
#include <helpers/UrMock.hpp>

#include <detail/queue_impl.hpp>
#include <detail/xpti_registry.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <sycl/sycl.hpp>

using ::testing::HasSubstr;
using namespace sycl;
XPTI_CALLBACK_API bool queryReceivedNotifications(uint16_t &TraceType,
                                                  std::string &Message);
XPTI_CALLBACK_API void resetReceivedNotifications();
XPTI_CALLBACK_API void addAnalyzedTraceType(uint16_t);
XPTI_CALLBACK_API void clearAnalyzedTraceTypes();

class QueueID : public ::testing::Test {
protected:
  void SetUp() {
    xptiForceSetTraceEnabled(true);
    xptiTraceTryToEnable();
    addAnalyzedTraceType(xpti::trace_task_begin);
    addAnalyzedTraceType(xpti::trace_task_end);
  }

  void TearDown() {
    clearAnalyzedTraceTypes();
    resetReceivedNotifications();
    xptiForceSetTraceEnabled(false);
  }

public:
  unittest::ScopedEnvVar PathToXPTIFW{"XPTI_FRAMEWORK_DISPATCHER",
                                      "libxptifw.so", [] {}};
  unittest::ScopedEnvVar XPTISubscriber{"XPTI_SUBSCRIBERS",
                                        "libxptitest_subscriber.so", [] {}};
  sycl::unittest::UrMock<> MockAdapter;

  static constexpr size_t KernelSize = 1;

  static constexpr char FileName[] = "QueueIDCheck.cpp";
  static constexpr char FunctionName[] = "TestCaseExecution";

  void checkTaskBeginEnd(const std::string &QueueIDStr) {
    uint16_t TraceType = 0;
    std::string Message;
    ASSERT_TRUE(queryReceivedNotifications(TraceType, Message));
    EXPECT_EQ(TraceType, xpti::trace_task_begin);
    EXPECT_THAT(Message, HasSubstr("task_begin:queue_id:" + QueueIDStr));
    ASSERT_TRUE(queryReceivedNotifications(TraceType, Message));
    EXPECT_EQ(TraceType, xpti::trace_task_end);
    EXPECT_THAT(Message, HasSubstr("task_end:queue_id:" + QueueIDStr));
  }
};

ur_queue_handle_t QueueHandle = nullptr;
inline ur_result_t redefinedQueueCreate(void *pParams) {
  auto params = *static_cast<ur_queue_create_params_t *>(pParams);
  QueueHandle = nullptr;
  if (*params.pphQueue)
    QueueHandle = **params.pphQueue;
  return UR_RESULT_SUCCESS;
}

TEST_F(QueueID, QueueID_QueueCreationAndDestroy) {
  sycl::platform Plt{sycl::platform()};
  mock::getCallbacks().set_after_callback("urQueueCreate",
                                          &redefinedQueueCreate);
  sycl::context Context{Plt};
  addAnalyzedTraceType(xpti::trace_queue_create);
  addAnalyzedTraceType(xpti::trace_queue_destroy);
  uint16_t TraceType = 0;
  std::string Message;
  std::string Queue0IDSTr;
  std::string Queue1IDSTr;
  {
    sycl::queue Q0{Context, sycl::default_selector{}};
    sycl::detail::queue_impl &Queue0Impl = *sycl::detail::getSyclObjImpl(Q0);
    Queue0IDSTr = std::to_string(Queue0Impl.getQueueID());
    ASSERT_TRUE(queryReceivedNotifications(TraceType, Message));
    EXPECT_EQ(TraceType, xpti::trace_queue_create);
    EXPECT_THAT(Message, HasSubstr("create:queue_id:" + Queue0IDSTr));
    ASSERT_NE(QueueHandle, nullptr);
    EXPECT_THAT(Message, HasSubstr("queue_handle:" +
                                   std::to_string(size_t(QueueHandle))));

    sycl::queue Q1{Context, sycl::default_selector{}};
    sycl::detail::queue_impl &Queue1Impl = *sycl::detail::getSyclObjImpl(Q1);
    Queue1IDSTr = std::to_string(Queue1Impl.getQueueID());
    ASSERT_TRUE(queryReceivedNotifications(TraceType, Message));
    EXPECT_EQ(TraceType, xpti::trace_queue_create);
    EXPECT_THAT(Message, HasSubstr("create:queue_id:" + Queue1IDSTr));
    ASSERT_NE(QueueHandle, nullptr);
    EXPECT_THAT(Message, HasSubstr("queue_handle:" +
                                   std::to_string(size_t(QueueHandle))));
  }

  ASSERT_TRUE(queryReceivedNotifications(TraceType, Message));
  EXPECT_EQ(TraceType, xpti::trace_queue_destroy);
  EXPECT_THAT(Message, HasSubstr("destroy:queue_id:" + Queue1IDSTr));
  ASSERT_TRUE(queryReceivedNotifications(TraceType, Message));
  EXPECT_EQ(TraceType, xpti::trace_queue_destroy);
  EXPECT_THAT(Message, HasSubstr("destroy:queue_id:" + Queue0IDSTr));
}

TEST_F(QueueID, QueueCreationAndKernelWithDeps) {
  sycl::queue Q0;
  sycl::queue Q1;
  sycl::detail::queue_impl &Queue0Impl = *sycl::detail::getSyclObjImpl(Q0);
  sycl::detail::queue_impl &Queue1Impl = *sycl::detail::getSyclObjImpl(Q1);
  sycl::buffer<int, 1> buf(sycl::range<1>(1));
  Q1.submit(
        [&](handler &Cgh) {
          sycl::accessor acc(buf, Cgh, sycl::read_write);
          Cgh.parallel_for<TestKernel>(1, [=](sycl::id<1> idx) {});
        },
        {FileName, FunctionName, 1, 0})
      .wait();
  EXPECT_NE(Queue1Impl.getQueueID(), Queue0Impl.getQueueID());
  auto QueueIDSTr = std::to_string(Queue1Impl.getQueueID());
  // alloca
  checkTaskBeginEnd(QueueIDSTr);
  // kernel
  checkTaskBeginEnd(QueueIDSTr);
}

// Re-enable this test after fixing
// https://github.com/intel/llvm/issues/12963
TEST_F(QueueID, DISABLED_QueueCreationUSMOperations) {
  sycl::queue Q0;
  sycl::detail::queue_impl &Queue0Impl = *sycl::detail::getSyclObjImpl(Q0);
  auto QueueIDSTr = std::to_string(Queue0Impl.getQueueID());

  unsigned char *AllocSrc = (unsigned char *)sycl::malloc_device(1, Q0);
  unsigned char *AllocDst = (unsigned char *)sycl::malloc_device(1, Q0);
  Q0.memset(AllocSrc, 42, 1).wait();
  checkTaskBeginEnd(QueueIDSTr);

  Q0.memcpy(AllocDst, AllocSrc, 1).wait();
  checkTaskBeginEnd(QueueIDSTr);

  Q0.submit([&](handler &Cgh) { Cgh.memset(AllocSrc, 42, 1); }).wait();
  checkTaskBeginEnd(QueueIDSTr);

  Q0.submit([&](handler &Cgh) { Cgh.memcpy(AllocDst, AllocSrc, 1); }).wait();
  checkTaskBeginEnd(QueueIDSTr);

  sycl::free(AllocSrc, Q0);
  sycl::free(AllocDst, Q0);
}

TEST_F(QueueID, QueueCreationAndKernelNoDeps) {
  sycl::queue Q0;
  sycl::queue Q1;

  sycl::detail::queue_impl &Queue0Impl = *sycl::detail::getSyclObjImpl(Q0);
  auto Queue0IDSTr = std::to_string(Queue0Impl.getQueueID());

  sycl::detail::queue_impl &Queue1Impl = *sycl::detail::getSyclObjImpl(Q1);
  auto Queue1IDSTr = std::to_string(Queue1Impl.getQueueID());

  Q0.submit(
        [&](handler &Cgh) {
          Cgh.parallel_for<TestKernel>(1, [=](sycl::id<1> idx) {});
        },
        {FileName, FunctionName, 2, 0})
      .wait();
  checkTaskBeginEnd(Queue0IDSTr);

  Q1.submit(
        [&](handler &Cgh) {
          Cgh.parallel_for<TestKernel>(1, [=](sycl::id<1> idx) {});
        },
        {FileName, FunctionName, 3, 0})
      .wait();
  checkTaskBeginEnd(Queue1IDSTr);
}

// host + kernel tasks
