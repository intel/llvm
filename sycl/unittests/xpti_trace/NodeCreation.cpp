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

class NodeCreation : public ::testing::Test {
protected:
  void SetUp() {
    xptiForceSetTraceEnabled(true);
    xptiTraceTryToEnable();
    addAnalyzedTraceType(xpti::trace_node_create);
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

  static constexpr char FileName[] = "NodeCreation.cpp";
  static constexpr char FunctionName[] = "TestCaseExecution";
  static constexpr int LineNumber = 8;
  static constexpr int ColumnNumber = 13;
  const sycl::detail::code_location TestCodeLocation = {
      FileName, FunctionName, LineNumber, ColumnNumber};
  static constexpr size_t KernelSize = 1;
};

TEST_F(NodeCreation, QueueParallelForWithGraphNode) {
  sycl::queue Q;
  try {
    sycl::buffer<int, 1> buf(sycl::range<1>(1));
    Q.submit(
        [&](handler &Cgh) {
          sycl::accessor acc(buf, Cgh, sycl::read_write);
          Cgh.parallel_for<TestKernel<KernelSize>>(1, [=](sycl::id<1> idx) {});
        },
        TestCodeLocation);
  } catch (sycl::exception &e) {
    std::ignore = e;
  }
  Q.wait();
  uint16_t TraceType = 0;
  std::string Message;
  ASSERT_TRUE(queryReceivedNotifications(TraceType, Message));
  EXPECT_EQ(TraceType, xpti::trace_node_create);
  EXPECT_THAT(Message, HasSubstr("TestKernel"));
}

TEST_F(NodeCreation, QueueParallelForWithNoGraphNode) {
  sycl::queue Q;
  try {
    Q.parallel_for<TestKernel<KernelSize>>(1, [=](sycl::id<1> idx) {});
  } catch (sycl::exception &e) {
    std::ignore = e;
  }
  Q.wait();
  uint16_t TraceType = 0;
  std::string Message;
  ASSERT_TRUE(queryReceivedNotifications(TraceType, Message));
  EXPECT_EQ(TraceType, xpti::trace_node_create);
  EXPECT_THAT(Message, HasSubstr("TestKernel"));
}

TEST_F(NodeCreation, QueueMemcpyNode) {
  sycl::queue Q;

  constexpr int n = 10 * sizeof(double);
  double HostPtr[n];
  double *DeviceUSMPtr = (double *)sycl::malloc_device(n, Q);
  Q.memcpy(DeviceUSMPtr, HostPtr, n).wait();
  sycl::free(DeviceUSMPtr, Q);

  uint16_t TraceType = 0;
  std::string Message;
  ASSERT_TRUE(queryReceivedNotifications(TraceType, Message));
  EXPECT_EQ(TraceType, xpti::trace_node_create);
  EXPECT_THAT(Message, HasSubstr("memory_transfer_node"));
}

TEST_F(NodeCreation, QueueMemsetNode) {
  sycl::queue Q;

  constexpr int n = 10 * sizeof(double);
  double *DeviceUSMPtr = (double *)sycl::malloc_device(n, Q);
  Q.memset(DeviceUSMPtr, 0, n).wait();
  sycl::free(DeviceUSMPtr, Q);

  uint16_t TraceType = 0;
  std::string Message;
  ASSERT_TRUE(queryReceivedNotifications(TraceType, Message));
  EXPECT_EQ(TraceType, xpti::trace_node_create);
  EXPECT_THAT(Message, HasSubstr("memory_transfer_node"));
}