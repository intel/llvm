//==------------ NodeCreation.cpp --- XPTI integration unit tests ----------==//
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

using ::testing::HasSubstr;
using namespace sycl;
XPTI_CALLBACK_API bool queryReceivedNotifications(uint16_t &TraceType,
                                                  std::string &Message);
XPTI_CALLBACK_API void resetReceivedNotifications();
XPTI_CALLBACK_API void addAnalyzedTraceType(uint16_t);

class QueueID : public ::testing::Test {
protected:
  void SetUp() {
    xptiForceSetTraceEnabled(true);
    xptiTraceTryToEnable();
    addAnalyzedTraceType(xpti::trace_queue_create);
    addAnalyzedTraceType(xpti::trace_queue_destroy);
    addAnalyzedTraceType(xpti::trace_task_begin);
    addAnalyzedTraceType(xpti::trace_task_end);
  }

  void TearDown() {
    resetReceivedNotifications();
    xptiForceSetTraceEnabled(false);
  }

public:
  unittest::ScopedEnvVar PathToXPTIFW{"XPTI_FRAMEWORK_DISPATCHER",
                                      "libxptifw.so", [] {}};
  unittest::ScopedEnvVar XPTISubscriber{"XPTI_SUBSCRIBERS",
                                        "libxptitest_subscriber.so", [] {}};
  sycl::unittest::PiMock MockPlugin;

  static constexpr size_t KernelSize = 1;
};

TEST_F(QueueID, QueueCreateAndDestroy) {
  uint16_t TraceType = 0;
  std::string Message;
  {
  sycl::queue Q0; 
  ASSERT_TRUE(queryReceivedNotifications(TraceType, Message));
  EXPECT_EQ(TraceType, xpti::trace_queue_create);
  EXPECT_THAT(Message, HasSubstr("create:queue_id:0"));
  sycl::queue Q1;
  ASSERT_TRUE(queryReceivedNotifications(TraceType, Message));
  EXPECT_EQ(TraceType, xpti::trace_queue_create);
  EXPECT_THAT(Message, HasSubstr("create:queue_id:1"));

  static constexpr char FileName[] = "QueueIDCheck.cpp";
  static constexpr char FunctionName[] = "TestCaseExecution";
  Q0.submit(
        [&](handler &Cgh) {
          Cgh.parallel_for<TestKernel<1>>(1, [=](sycl::id<1> idx) {});
        }, { FileName, FunctionName, 1, 0});
  //host?
  ASSERT_TRUE(queryReceivedNotifications(TraceType, Message));
  EXPECT_EQ(TraceType, xpti::trace_queue_create);
  EXPECT_THAT(Message, HasSubstr("create:queue_id:2"));

  ASSERT_TRUE(queryReceivedNotifications(TraceType, Message));
  EXPECT_EQ(TraceType, xpti::trace_task_begin);
  EXPECT_THAT(Message, HasSubstr("task_begin:queue_id:0"));
  ASSERT_TRUE(queryReceivedNotifications(TraceType, Message));
  EXPECT_EQ(TraceType, xpti::trace_task_end);
  EXPECT_THAT(Message, HasSubstr("task_end:queue_id:0"));
  Q1.submit(
        [&](handler &Cgh) {
          Cgh.parallel_for<TestKernel<1>>(1, [=](sycl::id<1> idx) {});
        }, { FileName, FunctionName, 2, 0}).wait();

  ASSERT_TRUE(queryReceivedNotifications(TraceType, Message));
  EXPECT_EQ(TraceType, xpti::trace_task_begin);
  EXPECT_THAT(Message, HasSubstr("task_begin:queue_id:1"));
  ASSERT_TRUE(queryReceivedNotifications(TraceType, Message));
  EXPECT_EQ(TraceType, xpti::trace_task_end);
  EXPECT_THAT(Message, HasSubstr("task_end:queue_id:1"));
  }
  ASSERT_TRUE(queryReceivedNotifications(TraceType, Message));
  EXPECT_EQ(TraceType, xpti::trace_queue_destroy);
  EXPECT_THAT(Message, HasSubstr("destroy:queue_id:1"));
  ASSERT_TRUE(queryReceivedNotifications(TraceType, Message));
  EXPECT_EQ(TraceType, xpti::trace_queue_destroy);
  EXPECT_THAT(Message, HasSubstr("destroy:queue_id:0"));
}