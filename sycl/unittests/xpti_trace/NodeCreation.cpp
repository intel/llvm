//==------------ NodeCreation.cpp --- XPTI integration unit tests ----------==//
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

#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <sycl/sycl.hpp>

using ::testing::HasSubstr;
using namespace sycl;
XPTI_CALLBACK_API bool queryReceivedNotifications(uint16_t &TraceType,
                                                  std::string &Message);
XPTI_CALLBACK_API void resetReceivedNotifications();
XPTI_CALLBACK_API void addAnalyzedTraceType(uint16_t);
XPTI_CALLBACK_API void clearAnalyzedTraceTypes();

class NodeCreation : public ::testing::Test {
protected:
  void SetUp() {
    xptiForceSetTraceEnabled(true);
    xptiTraceTryToEnable();
    addAnalyzedTraceType(xpti::trace_node_create);
  }

  void TearDown() {
    resetReceivedNotifications();
    clearAnalyzedTraceTypes();
    xptiForceSetTraceEnabled(false);
  }

public:
  unittest::ScopedEnvVar PathToXPTIFW{"XPTI_FRAMEWORK_DISPATCHER",
                                      "libxptifw.so", [] {}};
  unittest::ScopedEnvVar XPTISubscriber{"XPTI_SUBSCRIBERS",
                                        "libxptitest_subscriber.so", [] {}};
  sycl::unittest::UrMock<> MockAdapter;

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

TEST_F(NodeCreation, QueueParallelForWithUserCodeLoc) {
  sycl::queue Q;
  try {
    sycl::buffer<int, 1> buf(sycl::range<1>(1));
    sycl::detail::tls_code_loc_t myLoc(
        {"LOCAL_CODELOC_FILE", "LOCAL_CODELOC_NAME", 1, 1});
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
  EXPECT_THAT(Message, HasSubstr("LOCAL_CODELOC_NAME"));
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

TEST_F(NodeCreation, CommandGraphRecord) {
  sycl::queue Q;
  try {
    sycl::ext::oneapi::experimental::command_graph cmdGraph(Q.get_context(),
                                                            Q.get_device());

    cmdGraph.begin_recording(Q);

    {
      sycl::detail::tls_code_loc_t myLoc(
          {"LOCAL_CODELOC_FILE", "LOCAL_CODELOC_NAME", 1, 1});
      Q.submit([&](handler &Cgh) {
        Cgh.parallel_for<TestKernel<KernelSize>>(1, [=](sycl::id<1> idx) {});
      });
    }

    cmdGraph.end_recording(Q);

    addAnalyzedTraceType(xpti::trace_task_begin);
    addAnalyzedTraceType(xpti::trace_task_end);

    auto exeGraph = cmdGraph.finalize();

    // Notifications should have been generated during finalize
    uint16_t TraceType = 0;
    std::string Message;
    ASSERT_TRUE(queryReceivedNotifications(TraceType, Message));
    EXPECT_EQ(TraceType, xpti::trace_node_create);
    EXPECT_THAT(Message, HasSubstr("LOCAL_CODELOC_NAME"));

    ASSERT_TRUE(queryReceivedNotifications(TraceType, Message));
    EXPECT_EQ(TraceType, xpti::trace_task_begin);

    ASSERT_TRUE(queryReceivedNotifications(TraceType, Message));
    EXPECT_EQ(TraceType, xpti::trace_task_end);

  } catch (sycl::exception &e) {
    FAIL() << "sycl::exception what=" << e.what();
  }
}

TEST_F(NodeCreation, CommandGraphAddAPI) {
  sycl::queue Q;
  try {
    sycl::ext::oneapi::experimental::command_graph cmdGraph(Q.get_context(),
                                                            Q.get_device());

    auto doAddNode = [&](const sycl::detail::code_location &loc) {
      sycl::detail::tls_code_loc_t codeLoc(loc);
      return cmdGraph.add([&](handler &Cgh) {
        Cgh.parallel_for<TestKernel<KernelSize>>(1, [=](sycl::id<1> idx) {});
      });
    };

    auto node1 = doAddNode({"LOCAL_CODELOC_FILE", "LOCAL_NODE_1", 1, 1});
    auto node2 = doAddNode({"LOCAL_CODELOC_FILE", "LOCAL_NODE_2", 2, 1});
    cmdGraph.make_edge(node1, node2);

    addAnalyzedTraceType(xpti::trace_task_begin);
    addAnalyzedTraceType(xpti::trace_task_end);

    auto exeGraph = cmdGraph.finalize();

    // Notifications should have get generated during finalize
    //
    uint16_t TraceType = 0;
    std::string Message;
    ASSERT_TRUE(queryReceivedNotifications(TraceType, Message));
    EXPECT_EQ(TraceType, xpti::trace_node_create);
    EXPECT_THAT(Message, HasSubstr("LOCAL_NODE_1"));

    ASSERT_TRUE(queryReceivedNotifications(TraceType, Message));
    EXPECT_EQ(TraceType, xpti::trace_task_begin);

    ASSERT_TRUE(queryReceivedNotifications(TraceType, Message));
    EXPECT_EQ(TraceType, xpti::trace_task_end);

    ASSERT_TRUE(queryReceivedNotifications(TraceType, Message));
    EXPECT_EQ(TraceType, xpti::trace_node_create);
    EXPECT_THAT(Message, HasSubstr("LOCAL_NODE_2"));

    ASSERT_TRUE(queryReceivedNotifications(TraceType, Message));
    EXPECT_EQ(TraceType, xpti::trace_task_begin);

    ASSERT_TRUE(queryReceivedNotifications(TraceType, Message));
    EXPECT_EQ(TraceType, xpti::trace_task_end);

  } catch (sycl::exception &e) {
    FAIL() << "sycl::exception what=" << e.what();
  }
}
