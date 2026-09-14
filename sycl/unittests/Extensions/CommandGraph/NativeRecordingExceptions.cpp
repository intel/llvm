//==--------------------- NativeRecordingExceptions.cpp --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Checks that a failure reported by a UR native recording entry point reaches
// the user with the originating UR error code and expected message.
// Erroneous graph operations print the point of failure in the error message
// while recorded operations which fail just print the UR error code.

#include "NativeRecordingMock.hpp"

using NativeRecordingMock::expectFailure;
using NativeRecordingMock::state;
using ::testing::HasSubstr;

// Test that native recording throws when UR does not support it
TEST_F(NativeRecordingTest, NativeRecordingUnsupportedDevice) {
  state().SupportsNativeRecording = false;

  std::string Message =
      expectFailure([&]() { makeGraph(); }, std::nullopt, sycl::errc::invalid);
  EXPECT_THAT(Message, HasSubstr("does not support graph record and replay"));
}

TEST_F(NativeRecordingTest, UnjoinedForkDescriptiveError) {
  auto Graph1 = makeGraph();
  Graph1.begin_recording(Queue);

  FAIL_UR_AFTER(urQueueEndGraphCaptureExp,
                UR_RESULT_ERROR_GRAPH_UNJOINED_FORKS);
  std::string Message = expectFailure([&]() { Graph1.end_recording(Queue); },
                                      UR_RESULT_ERROR_GRAPH_UNJOINED_FORKS);
  EXPECT_THAT(Message, HasSubstr("ending native graph capture"));

  auto Graph2 = makeGraph();
  Graph2.begin_recording(Queue);
  Message = expectFailure([&]() { Graph2.end_recording(); },
                          UR_RESULT_ERROR_GRAPH_UNJOINED_FORKS);
  EXPECT_THAT(Message, HasSubstr("ending native graph capture"));
}

TEST_F(NativeRecordingTest, InternalEventDescriptiveError) {
  auto Graph = makeGraph();
  Graph.begin_recording(Queue);

  // Mock the next submission containing a wait event signaled on a
  // non-recording queue.
  FAIL_UR_BEFORE(urEnqueueKernelLaunchWithArgsExp,
                 UR_RESULT_ERROR_GRAPH_INTERNAL_EVENT);
  expectFailure(
      [&]() {
        Queue.submit(
            [&](sycl::handler &CGH) { CGH.single_task<TestKernel>([]() {}); });
      },
      UR_RESULT_ERROR_GRAPH_INTERNAL_EVENT);

  Graph.end_recording(Queue);
}

TEST_F(NativeRecordingTest, MergeAttemptDescriptiveError) {
  auto Graph = makeGraph();
  Graph.begin_recording(Queue);

  // Mock the next submission containing a wait event captured in a separate
  // graph.
  FAIL_UR_BEFORE(urEnqueueKernelLaunchWithArgsExp,
                 UR_RESULT_ERROR_GRAPH_CAPTURE_MERGE_ATTEMPT);
  expectFailure(
      [&]() {
        Queue.submit(
            [&](sycl::handler &CGH) { CGH.single_task<TestKernel>([]() {}); });
      },
      UR_RESULT_ERROR_GRAPH_CAPTURE_MERGE_ATTEMPT);

  Graph.end_recording(Queue);
}

TEST_F(NativeRecordingTest, RecordedEventHostWaitDescriptiveError) {
  auto Graph = makeGraph();
  Graph.begin_recording(Queue);

  sycl::event RecordedEvent = Queue.submit(
      [&](sycl::handler &CGH) { CGH.single_task<TestKernel>([]() {}); });

  FAIL_UR_BEFORE(urEventWait, UR_RESULT_ERROR_GRAPH_CAPTURE_UNSUPPORTED);
  expectFailure([&]() { RecordedEvent.wait(); },
                UR_RESULT_ERROR_GRAPH_CAPTURE_UNSUPPORTED);

  Graph.end_recording(Queue);
}

TEST_F(NativeRecordingTest, RecordingQueueWaitDescriptiveError) {
  auto Graph = makeGraph();
  Graph.begin_recording(Queue);

  FAIL_UR_BEFORE(urQueueFinish, UR_RESULT_ERROR_GRAPH_CAPTURE_UNSUPPORTED);
  expectFailure([&]() { Queue.wait(); },
                UR_RESULT_ERROR_GRAPH_CAPTURE_UNSUPPORTED);

  Graph.end_recording(Queue);
}

TEST_F(NativeRecordingTest, GraphCreateDescriptiveError) {
  FAIL_UR_BEFORE(urGraphCreateExp, UR_RESULT_ERROR_OUT_OF_RESOURCES);
  std::string Message =
      expectFailure([&]() { makeGraph(); }, UR_RESULT_ERROR_OUT_OF_RESOURCES);
  EXPECT_THAT(Message, HasSubstr("create native UR graph"));
}

TEST_F(NativeRecordingTest, FinalizeDescriptiveError) {
  auto Graph = makeGraph();

  Graph.begin_recording(Queue);
  Queue.submit(
      [&](sycl::handler &CGH) { CGH.single_task<TestKernel>([]() {}); });
  Graph.end_recording(Queue);

  FAIL_UR_BEFORE(urGraphInstantiateGraphExp, UR_RESULT_ERROR_INVALID_GRAPH);
  std::string Message =
      expectFailure([&]() { Graph.finalize(); }, UR_RESULT_ERROR_INVALID_GRAPH);
  EXPECT_THAT(Message, HasSubstr("instantiate native UR executable graph"));
}

TEST_F(NativeRecordingTest, BeginRecordingDescriptiveError) {
  auto Graph = makeGraph();

  FAIL_UR_BEFORE(urQueueBeginCaptureIntoGraphExp,
                 UR_RESULT_ERROR_GRAPH_CAPTURE_UNSUPPORTED);
  std::string Message =
      expectFailure([&]() { Graph.begin_recording(Queue); },
                    UR_RESULT_ERROR_GRAPH_CAPTURE_UNSUPPORTED);
  EXPECT_THAT(Message, HasSubstr("begin native UR graph capture"));
}

TEST_F(NativeRecordingTest, SetDestructionCallbackDescriptiveError) {
  bool CallbackFired = false;
  {
    auto Graph = makeGraph();

    FAIL_UR_BEFORE(urGraphSetDestructionCallbackExp,
                   UR_RESULT_ERROR_OUT_OF_RESOURCES);
    std::string Message = expectFailure(
        [&]() {
          Graph.set_destruction_callback(
              [&CallbackFired]() { CallbackFired = true; });
        },
        UR_RESULT_ERROR_OUT_OF_RESOURCES);
    EXPECT_THAT(Message, HasSubstr("register graph destruction callback"));
  }
}

TEST_F(NativeRecordingTest, EmptyDescriptiveError) {
  auto Graph = makeGraph();

  FAIL_UR_BEFORE(urGraphIsEmptyExp, UR_RESULT_ERROR_INVALID_GRAPH);
  std::string Message =
      expectFailure([&]() { Graph.empty(); }, UR_RESULT_ERROR_INVALID_GRAPH);
  EXPECT_THAT(Message, HasSubstr("check if graph is empty"));
}

TEST_F(NativeRecordingTest, PrintGraphDescriptiveError) {
  auto Graph = makeGraph();

  FAIL_UR_BEFORE(urGraphDumpContentsExp, UR_RESULT_ERROR_UNSUPPORTED_FEATURE);
  std::string Message =
      expectFailure([&]() { Graph.print_graph("native_graph.dot"); },
                    UR_RESULT_ERROR_UNSUPPORTED_FEATURE);
  EXPECT_THAT(Message, HasSubstr("dump native UR graph contents"));
}

TEST_F(NativeRecordingTest, GetGraphDescriptiveError) {
  auto Graph = makeGraph();
  Graph.begin_recording(Queue);

  FAIL_UR_BEFORE(urQueueGetGraphExp, UR_RESULT_ERROR_INVALID_GRAPH);
  std::string Message = expectFailure([&]() { Queue.ext_oneapi_get_graph(); },
                                      UR_RESULT_ERROR_INVALID_GRAPH);
  EXPECT_THAT(Message, HasSubstr("query native UR graph from queue"));

  Graph.end_recording(Queue);
}

// NOTE: The current implementation manually returns sycl::errc::invalid and the
// UR implementation manually handles this case, diverging from the native L0
// graph return.
TEST_F(NativeRecordingTest, GetGraphNotRecordingError) {
  auto Graph = makeGraph();
  Graph.begin_recording(Queue);

  FAIL_UR_BEFORE(urQueueGetGraphExp, UR_RESULT_ERROR_INVALID_OPERATION);
  std::string Message = expectFailure([&]() { Queue.ext_oneapi_get_graph(); },
                                      std::nullopt, sycl::errc::invalid);
  EXPECT_THAT(Message, HasSubstr("can only be called on recording queues"));

  Graph.end_recording(Queue);
}
