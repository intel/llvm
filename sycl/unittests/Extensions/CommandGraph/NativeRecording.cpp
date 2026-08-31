//==------------------------- NativeRecording.cpp --------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NativeRecordingMock.hpp"

using NativeRecordingMock::state;
using NativeRecordingMock::traceCount;
using NativeRecordingMock::traceIndex;

// Test that native recording throws when UR does not support it
TEST_F(NativeRecordingTest, NativeRecordingUnsupportedDevice) {
  state().SupportsNativeRecording = false;
  try {
    makeGraph();
    FAIL() << "Expected an exception";
  } catch (sycl::exception &E) {
    EXPECT_EQ(E.code(), sycl::errc::invalid);
  }
}

// Traces UR recording layer
TEST_F(NativeRecordingTest, RecordingUrTrace) {
  auto Graph = makeGraph();

  Graph.begin_recording(Queue);
  Queue.submit(
      [&](sycl::handler &CGH) { CGH.single_task<TestKernel>([]() {}); });
  Graph.end_recording(Queue);

  ASSERT_EQ(traceCount("urQueueBeginCaptureIntoGraphExp"), 1u);
  ASSERT_EQ(traceCount("urEnqueueKernelLaunchWithArgsExp"), 1u);
  ASSERT_EQ(traceCount("urQueueEndGraphCaptureExp"), 1u);
  EXPECT_LT(traceIndex("urQueueBeginCaptureIntoGraphExp"),
            traceIndex("urEnqueueKernelLaunchWithArgsExp"));
  EXPECT_LT(traceIndex("urEnqueueKernelLaunchWithArgsExp"),
            traceIndex("urQueueEndGraphCaptureExp"));
}

// Finalize and submission traces
TEST_F(NativeRecordingTest, FinalizeSubmitUrTrace) {
  auto Graph = makeGraph();

  Graph.begin_recording(Queue);
  Queue.submit(
      [&](sycl::handler &CGH) { CGH.single_task<TestKernel>([]() {}); });
  Graph.end_recording(Queue);

  EXPECT_EQ(traceCount("urGraphInstantiateGraphExp"), 0u);

  auto ExecGraph = Graph.finalize();

  EXPECT_EQ(traceCount("urGraphInstantiateGraphExp", nativeHandle(Graph)), 1u);
  ASSERT_NE(nativeHandle(ExecGraph), nullptr);
  EXPECT_EQ(traceCount("urEnqueueGraphExp"), 0u);

  Queue.ext_oneapi_graph(ExecGraph);
  Queue.wait();

  EXPECT_EQ(traceCount("urEnqueueGraphExp", nativeHandle(ExecGraph)), 1u);
  EXPECT_EQ(traceCount("urCommandBufferCreateExp"), 0u);
}

// The executable graph must be destroyed prior to the modifiable.
TEST_F(NativeRecordingTest, DestructionOrder) {
  ur_exp_graph_handle_t GraphHandle = nullptr;
  ur_exp_executable_graph_handle_t ExecHandle = nullptr;
  {
    auto ExecGraph = [&]() {
      auto Graph = makeGraph();
      GraphHandle = nativeHandle(Graph);

      Graph.begin_recording(Queue);
      Queue.submit(
          [&](sycl::handler &CGH) { CGH.single_task<TestKernel>([]() {}); });
      Graph.end_recording(Queue);

      return Graph.finalize();
    }();
    ExecHandle = nativeHandle(ExecGraph);

    ASSERT_NE(GraphHandle, nullptr);
    ASSERT_NE(ExecHandle, nullptr);
    EXPECT_EQ(traceCount("urGraphDestroyExp"), 0u);
    EXPECT_EQ(traceCount("urGraphExecutableGraphDestroyExp"), 0u);
  }

  EXPECT_EQ(traceCount("urGraphExecutableGraphDestroyExp", ExecHandle), 1u);
  EXPECT_EQ(traceCount("urGraphDestroyExp", GraphHandle), 1u);
  EXPECT_LT(traceIndex("urGraphExecutableGraphDestroyExp"),
            traceIndex("urGraphDestroyExp"));
}

// Check that destruction callback goes through UR and not SYCL command buffer
// path.
TEST_F(NativeRecordingTest, DestructionCallbackUrTrace) {
  bool CallbackFired1 = false;
  bool CallbackFired2 = false;
  ur_exp_graph_handle_t Handle = nullptr;
  {
    auto Graph = makeGraph();
    Handle = nativeHandle(Graph);

    EXPECT_EQ(traceCount("urGraphCreateExp", Handle), 1u);
    ASSERT_NE(Handle, nullptr);

    Graph.set_destruction_callback(
        [&CallbackFired1]() { CallbackFired1 = true; });
    Graph.set_destruction_callback(
        [&CallbackFired2]() { CallbackFired2 = true; });

    EXPECT_EQ(traceCount("urGraphSetDestructionCallbackExp", Handle), 2u);
    EXPECT_FALSE(CallbackFired1);
    EXPECT_FALSE(CallbackFired2);
    EXPECT_EQ(traceCount("urGraphDestroyExp"), 0u);
  }

  EXPECT_EQ(traceCount("urGraphDestroyExp", Handle), 1u);
  EXPECT_LT(traceIndex("urGraphCreateExp"), traceIndex("urGraphDestroyExp"));
  EXPECT_LT(traceIndex("urGraphSetDestructionCallbackExp"),
            traceIndex("urGraphDestroyExp"));
  EXPECT_TRUE(CallbackFired1);
  EXPECT_TRUE(CallbackFired2);
}

// Check that the graph ID is going through UR and not the SYCL command buffer
// or native recording fallback path.
TEST_F(NativeRecordingTest, GetIdUrTrace) {
  auto Graph = makeGraph();
  EXPECT_EQ(Graph.get_id(), NativeRecordingMock::FirstGraphId);
  EXPECT_EQ(traceCount("urGraphGetIdExp", nativeHandle(Graph)), 1u);
}

// Check UR call for get graph and graph uniqueness
TEST_F(NativeRecordingTest, GetGraphUrTrace) {
  auto Graph = makeGraph();
  auto SecondGraph = makeGraph();
  sycl::queue SecondQueue{Dev, {sycl::property::queue::in_order{}}};

  Graph.begin_recording(Queue);
  SecondGraph.begin_recording(SecondQueue);

  auto RecordedGraph = Queue.ext_oneapi_get_graph();
  auto SecondRecordedGraph = SecondQueue.ext_oneapi_get_graph();

  EXPECT_EQ(traceCount("urQueueGetGraphExp"), 2u);
  EXPECT_EQ(getSyclObjImpl(RecordedGraph), getSyclObjImpl(Graph));
  EXPECT_EQ(getSyclObjImpl(SecondRecordedGraph), getSyclObjImpl(SecondGraph));
  EXPECT_EQ(nativeHandle(RecordedGraph), nativeHandle(Graph));
  EXPECT_EQ(nativeHandle(SecondRecordedGraph), nativeHandle(SecondGraph));

  Graph.end_recording(Queue);
  SecondGraph.end_recording(SecondQueue);
}

// Check UR empty graph call
TEST_F(NativeRecordingTest, EmptyUrTrace) {
  auto Graph = makeGraph();
  ur_exp_graph_handle_t Handle = nativeHandle(Graph);

  state().graph(Handle).IsEmpty = true;
  EXPECT_TRUE(Graph.empty());
  EXPECT_EQ(traceCount("urGraphIsEmptyExp", Handle), 1u);

  state().graph(Handle).IsEmpty = false;
  EXPECT_FALSE(Graph.empty());
  EXPECT_EQ(traceCount("urGraphIsEmptyExp", Handle), 2u);
}

// Check UR call for queue state
TEST_F(NativeRecordingTest, GetStateUrTrace) {
  auto Graph = makeGraph();
  EXPECT_EQ(Queue.ext_oneapi_get_state(), experimental::queue_state::executing);

  Graph.begin_recording(Queue);
  EXPECT_EQ(Queue.ext_oneapi_get_state(), experimental::queue_state::recording);

  Graph.end_recording(Queue);
  EXPECT_EQ(Queue.ext_oneapi_get_state(), experimental::queue_state::executing);

  EXPECT_GE(traceCount("urQueueIsGraphCaptureEnabledExp"), 3u);
}
