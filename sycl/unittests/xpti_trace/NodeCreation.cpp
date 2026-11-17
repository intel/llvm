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

#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <sycl/sycl.hpp>

#include <chrono>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <optional>
#include <string_view>
#include <thread>

using ::testing::HasSubstr;
using namespace sycl;
XPTI_CALLBACK_API bool queryReceivedNotifications(uint16_t &TraceType,
                                                  std::string &Message);
XPTI_CALLBACK_API void resetReceivedNotifications();
XPTI_CALLBACK_API void addAnalyzedTraceType(uint16_t);
XPTI_CALLBACK_API void clearAnalyzedTraceTypes();
XPTI_CALLBACK_API void testCallback(uint16_t, xpti_td *, xpti_td *, uint64_t,
                                    const void *);

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
};

TEST_F(NodeCreation, QueueParallelForWithGraphNode) {
  sycl::queue Q;
  try {
    sycl::buffer<int, 1> buf(sycl::range<1>(1));
    Q.submit(
        [&](handler &Cgh) {
          sycl::accessor acc(buf, Cgh, sycl::read_write);
          Cgh.parallel_for<TestKernel>(1, [=](sycl::id<1> idx) {});
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
    Q.parallel_for<TestKernel>(1, [=](sycl::id<1> idx) {});
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

namespace {
struct KernelTraceState {
  uint64_t ID = 0;
  uint64_t Instance = 0;
  size_t Range = 1;
  unsigned Notifications = 0;
  bool DisableAtBegin = false;
  bool OverlapAtBegin = false;
  std::optional<int> Argument;
};
thread_local KernelTraceState KernelTrace;
std::mutex OverlapMutex;
std::condition_variable OverlapCV;
unsigned EnqueuesInProgress = 0;
uint64_t OverlapIDs[2] = {};

void kernelLifetimeCallback(uint16_t Type, xpti_td *, xpti_td *Event,
                            uint64_t Instance, const void *) {
  ASSERT_NE(Event, nullptr);
  ASSERT_EQ(xptiFindEvent(Event->unique_id), Event);
  auto *Metadata = xptiQueryMetadata(Event);
  ASSERT_NE(Metadata, nullptr);
  bool FoundKernel = false;
  bool FoundRange = false;
  bool FoundArgument = false;
  for (const auto &Item : *Metadata) {
    if (std::string_view(xptiLookupString(Item.first)) == "kernel_name")
      FoundKernel = true;
    if (std::string_view(xptiLookupString(Item.first)) ==
        "enqueue_kernel_data") {
      auto Data = xpti::getMetadata<xpti::offload_kernel_enqueue_data_t>(Item);
      EXPECT_EQ(Data.second.global_size[0], KernelTrace.Range);
      FoundRange = true;
    }
    if (std::string_view(xptiLookupString(Item.first)) == "arg0" &&
        KernelTrace.Argument) {
      auto Data = xpti::getMetadata<xpti::offload_kernel_arg_data_t>(Item);
      ASSERT_EQ(Data.second.size, static_cast<int>(sizeof(int)));
      EXPECT_EQ(*static_cast<int *>(Data.second.pointer),
                *KernelTrace.Argument);
      FoundArgument = true;
    }
  }
  ASSERT_TRUE(FoundKernel);
  EXPECT_TRUE(FoundRange);
  if (KernelTrace.Argument) {
    EXPECT_TRUE(FoundArgument);
  }
  if (Type == xpti::trace_node_create) {
    KernelTrace.ID = Event->unique_id;
    KernelTrace.Instance = Instance;
  } else {
    EXPECT_EQ(Event->unique_id, KernelTrace.ID);
    EXPECT_EQ(Instance, KernelTrace.Instance);
  }
  ++KernelTrace.Notifications;
  if (Type == xpti::trace_task_begin && KernelTrace.OverlapAtBegin) {
    std::unique_lock<std::mutex> Lock(OverlapMutex);
    ASSERT_LT(EnqueuesInProgress, 2u);
    OverlapIDs[EnqueuesInProgress++] = Event->unique_id;
    OverlapCV.notify_all();
    EXPECT_TRUE(OverlapCV.wait_for(Lock, std::chrono::seconds(10),
                                   [] { return EnqueuesInProgress == 2; }));
    EXPECT_NE(OverlapIDs[0], OverlapIDs[1]);
  }
  if (Type == xpti::trace_task_begin && KernelTrace.DisableAtBegin)
    xptiForceSetTraceEnabled(false);
}

class KernelTraceLifetime : public NodeCreation {
  static constexpr uint16_t Types[] = {
      xpti::trace_node_create, xpti::trace_task_begin, xpti::trace_signal,
      xpti::trace_task_end};

protected:
  void SetUp() override {
    NodeCreation::SetUp();
    clearAnalyzedTraceTypes();
    // Initialize streams before registering our callbacks.
    sycl::queue Q;
    KernelTrace = {};
    mock::getCallbacks().set_after_callback(
        "urKernelGetInfo", [](void *Params) {
          auto &P = *static_cast<ur_kernel_get_info_params_t *>(Params);
          constexpr char Name[] = "TestKernel";
          if (*P.ppropName == UR_KERNEL_INFO_FUNCTION_NAME) {
            if (*P.ppPropValue)
              std::memcpy(*P.ppPropValue, Name, sizeof(Name));
            if (*P.ppPropSizeRet)
              **P.ppPropSizeRet = sizeof(Name);
          }
          return UR_RESULT_SUCCESS;
        });
    for (auto Stream : {detail::GSYCLStreamID, detail::GSYCLDebugStreamID})
      for (auto Type : Types)
        xptiRegisterCallback(Stream, Type, kernelLifetimeCallback);
  }
  void TearDown() override {
    for (auto Stream : {detail::GSYCLStreamID, detail::GSYCLDebugStreamID}) {
      for (auto Type : Types)
        xptiUnregisterCallback(Stream, Type, kernelLifetimeCallback);
      // Unregistering a callback clears the trace-type flag. Restore the
      // original subscriber's flags for subsequent fixtures.
      for (auto Type : {xpti::trace_node_create, xpti::trace_task_begin,
                        xpti::trace_task_end})
        xptiRegisterCallback(Stream, Type, testCallback);
    }
    NodeCreation::TearDown();
  }
};
} // namespace

// Direct submissions must keep metadata queryable through their last callback,
// then release the per-submission event rather than accumulating it in XPTI.
TEST_F(KernelTraceLifetime, LambdaAndKernelObject) {
  sycl::queue Q{property::queue::in_order{}};
  auto Bundle = get_kernel_bundle<bundle_state::executable>(Q.get_context());
  auto Kernel = Bundle.get_kernel(get_kernel_id<TestKernel>());
  for (bool UseKernelObject : {false, true}) {
    for (size_t Range : {1, 7, 13}) {
      KernelTrace = {};
      KernelTrace.Range = Range;
      if (UseKernelObject)
        Q.submit(
            [&](handler &Cgh) { Cgh.parallel_for(range<1>(Range), Kernel); });
      else
        Q.parallel_for<TestKernel>(range<1>(Range), [](id<1>) {});
      EXPECT_EQ(KernelTrace.Notifications, 4u);
      EXPECT_NE(KernelTrace.ID, 0u);
      EXPECT_EQ(xptiFindEvent(KernelTrace.ID), nullptr);
    }
  }
}

TEST_F(KernelTraceLifetime, WithoutBackendEvent) {
  sycl::queue Q{property::queue::in_order{}};
  sycl::ext::oneapi::experimental::submit(Q, [](handler &Cgh) {
    Cgh.parallel_for<TestKernel>(range<1>(1), [](id<1>) {});
  });
  EXPECT_EQ(KernelTrace.Notifications, 3u);
  EXPECT_NE(KernelTrace.ID, 0u);
  EXPECT_EQ(xptiFindEvent(KernelTrace.ID), nullptr);
}

TEST_F(KernelTraceLifetime, EnqueueFailure) {
  sycl::queue Q;
  auto Fail = [](void *) { return UR_RESULT_ERROR_INVALID_KERNEL; };
  mock::getCallbacks().set_replace_callback("urEnqueueKernelLaunch", Fail);
  mock::getCallbacks().set_replace_callback("urEnqueueKernelLaunchWithArgsExp",
                                            Fail);
  EXPECT_THROW(Q.parallel_for<TestKernel>(range<1>(1), [](id<1>) {}),
               sycl::exception);
  EXPECT_EQ(KernelTrace.Notifications, 2u);
  EXPECT_NE(KernelTrace.ID, 0u);
  EXPECT_EQ(xptiFindEvent(KernelTrace.ID), nullptr);
}

TEST_F(KernelTraceLifetime, TracingDisabledDuringEnqueue) {
  sycl::queue Q;
  KernelTrace.DisableAtBegin = true;
  Q.parallel_for<TestKernel>(range<1>(1), [](id<1>) {});
  EXPECT_EQ(KernelTrace.Notifications, 2u);
  EXPECT_NE(KernelTrace.ID, 0u);
  EXPECT_EQ(xptiFindEvent(KernelTrace.ID), nullptr);
}

TEST_F(KernelTraceLifetime, ConcurrentQueuesWithDifferentMetadata) {
  sycl::queue Q0, Q1;
  auto Bundle = get_kernel_bundle<bundle_state::executable>(Q0.get_context());
  auto Kernel = Bundle.get_kernel(get_kernel_id<TestKernel>());
  EnqueuesInProgress = 0;
  auto Submit = [&](queue &Q, size_t Range, int Argument) {
    KernelTrace = {};
    KernelTrace.Range = Range;
    KernelTrace.Argument = Argument;
    KernelTrace.OverlapAtBegin = true;
    Q.submit(
        [&](handler &Cgh) {
          Cgh.set_arg(0, Argument);
          Cgh.parallel_for(range<1>(Range), Kernel);
        },
        TestCodeLocation);
    EXPECT_EQ(KernelTrace.Notifications, 4u);
    EXPECT_NE(KernelTrace.ID, 0u);
    EXPECT_EQ(xptiFindEvent(KernelTrace.ID), nullptr);
  };
  std::thread T0([&] { Submit(Q0, 7, 42); });
  std::thread T1([&] { Submit(Q1, 13, 99); });
  T0.join();
  T1.join();
}

TEST_F(KernelTraceLifetime, CommandBufferInsertion) {
  sycl::queue Q;
  sycl::ext::oneapi::experimental::command_graph Graph(Q.get_context(),
                                                       Q.get_device());
  Graph.add([](handler &Cgh) {
    Cgh.parallel_for<TestKernel>(range<1>(1), [](id<1>) {});
  });
  auto Executable = Graph.finalize();
  EXPECT_EQ(KernelTrace.Notifications, 3u);
  EXPECT_NE(KernelTrace.ID, 0u);
  EXPECT_EQ(xptiFindEvent(KernelTrace.ID), nullptr);
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
          Cgh.parallel_for<TestKernel>(1, [=](sycl::id<1> idx) {});
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
        Cgh.parallel_for<TestKernel>(1, [=](sycl::id<1> idx) {});
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
        Cgh.parallel_for<TestKernel>(1, [=](sycl::id<1> idx) {});
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
