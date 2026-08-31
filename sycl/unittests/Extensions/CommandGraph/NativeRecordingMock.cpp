//==--------------------- NativeRecordingMock.cpp---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NativeRecordingMock.hpp"

#include <algorithm>

namespace NativeRecordingMock {

GraphState &MockState::graph(ur_exp_graph_handle_t Handle) {
  auto It = Graphs.find(Handle);
  assert(It != Graphs.end() && "Not a live mock graph handle");
  return It->second;
}

MockState &state() {
  static MockState State;
  return State;
}

size_t traceCount(std::string_view EntryPoint) {
  const std::vector<TraceEntry> &Trace = state().Trace;
  return std::count_if(
      Trace.begin(), Trace.end(),
      [&](const TraceEntry &Entry) { return Entry.EntryPoint == EntryPoint; });
}

size_t traceCount(std::string_view EntryPoint, const void *Handle) {
  const std::vector<TraceEntry> &Trace = state().Trace;
  return std::count_if(
      Trace.begin(), Trace.end(), [&](const TraceEntry &Entry) {
        return Entry.EntryPoint == EntryPoint && Entry.Handle == Handle;
      });
}

size_t traceIndex(std::string_view EntryPoint) {
  const std::vector<TraceEntry> &Trace = state().Trace;
  auto It =
      std::find_if(Trace.begin(), Trace.end(), [&](const TraceEntry &Entry) {
        return Entry.EntryPoint == EntryPoint;
      });
  if (It == Trace.end()) {
    ADD_FAILURE() << EntryPoint << " was never called";
    return std::string::npos;
  }
  return It - Trace.begin();
}

namespace {

void trace(std::string EntryPoint, const void *Handle = nullptr) {
  state().Trace.push_back({std::move(EntryPoint), Handle});
}

// Adds native recording support
ur_result_t mock_urDeviceGetInfoAfter(void *pParams) {
  auto Params = *static_cast<ur_device_get_info_params_t *>(pParams);
  if (*Params.ppropName == UR_DEVICE_INFO_GRAPH_RECORD_AND_REPLAY_SUPPORT_EXP) {
    if (*Params.ppPropValue)
      *static_cast<ur_bool_t *>(*Params.ppPropValue) =
          state().SupportsNativeRecording;
    if (*Params.ppPropSizeRet)
      **Params.ppPropSizeRet = sizeof(ur_bool_t);
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t mock_urGraphCreateExp(void *pParams) {
  auto Params = *static_cast<ur_graph_create_exp_params_t *>(pParams);
  MockState &State = state();
  auto Handle = mock::createDummyHandle<ur_exp_graph_handle_t>();
  State.Graphs[Handle] = GraphState{State.NextGraphId++};
  trace("urGraphCreateExp", Handle);
  **Params.pphGraph = Handle;
  return UR_RESULT_SUCCESS;
}

ur_result_t mock_urGraphDestroyExp(void *pParams) {
  auto Params = *static_cast<ur_graph_destroy_exp_params_t *>(pParams);
  MockState &State = state();
  ur_exp_graph_handle_t Handle = *Params.phGraph;
  trace("urGraphDestroyExp", Handle);
  const GraphState Destroyed = State.graph(Handle);
  State.Graphs.erase(Handle);
  mock::releaseDummyHandle(Handle);
  for (size_t i = 0; i < Destroyed.DestructionCallbacks.size(); ++i) {
    Destroyed.DestructionCallbacks[i](Destroyed.DestructionCallbacksData[i]);
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t mock_urGraphSetDestructionCallbackExp(void *pParams) {
  auto Params =
      *static_cast<ur_graph_set_destruction_callback_exp_params_t *>(pParams);
  trace("urGraphSetDestructionCallbackExp", *Params.phGraph);
  GraphState &Graph = state().graph(*Params.phGraph);
  Graph.DestructionCallbacks.push_back(*Params.ppfnCallback);
  Graph.DestructionCallbacksData.push_back(*Params.ppUserData);
  return UR_RESULT_SUCCESS;
}

ur_result_t mock_urGraphInstantiateGraphExp(void *pParams) {
  auto Params =
      *static_cast<ur_graph_instantiate_graph_exp_params_t *>(pParams);
  trace("urGraphInstantiateGraphExp", *Params.phGraph);
  **Params.pphExecGraph =
      mock::createDummyHandle<ur_exp_executable_graph_handle_t>();
  return UR_RESULT_SUCCESS;
}

ur_result_t mock_urGraphExecutableGraphDestroyExp(void *pParams) {
  auto Params =
      *static_cast<ur_graph_executable_graph_destroy_exp_params_t *>(pParams);
  trace("urGraphExecutableGraphDestroyExp", *Params.phExecutableGraph);
  mock::releaseDummyHandle(*Params.phExecutableGraph);
  return UR_RESULT_SUCCESS;
}

ur_result_t mock_urGraphIsEmptyExp(void *pParams) {
  auto Params = *static_cast<ur_graph_is_empty_exp_params_t *>(pParams);
  trace("urGraphIsEmptyExp", *Params.phGraph);
  **Params.ppResult = state().graph(*Params.phGraph).IsEmpty;
  return UR_RESULT_SUCCESS;
}

ur_result_t mock_urGraphGetIdExp(void *pParams) {
  auto Params = *static_cast<ur_graph_get_id_exp_params_t *>(pParams);
  trace("urGraphGetIdExp", *Params.phGraph);
  **Params.ppGraphId = state().graph(*Params.phGraph).Id;
  return UR_RESULT_SUCCESS;
}

ur_result_t mock_urQueueBeginCaptureIntoGraphExp(void *pParams) {
  auto Params =
      *static_cast<ur_queue_begin_capture_into_graph_exp_params_t *>(pParams);
  trace("urQueueBeginCaptureIntoGraphExp", *Params.phQueue);
  if (!state()
           .PrimaryCapturing.try_emplace(*Params.phQueue, *Params.phGraph)
           .second)
    return UR_RESULT_ERROR_INVALID_ARGUMENT;
  return UR_RESULT_SUCCESS;
}

ur_result_t mock_urQueueEndGraphCaptureExp(void *pParams) {
  auto Params =
      *static_cast<ur_queue_end_graph_capture_exp_params_t *>(pParams);
  MockState &State = state();
  trace("urQueueEndGraphCaptureExp", *Params.phQueue);
  auto It = State.PrimaryCapturing.find(*Params.phQueue);
  if (It == State.PrimaryCapturing.end())
    return UR_RESULT_ERROR_COMMAND_LIST_NOT_CAPTURING;
  **Params.pphGraph = It->second;
  State.PrimaryCapturing.erase(It);
  return UR_RESULT_SUCCESS;
}

ur_result_t mock_urQueueIsGraphCaptureEnabledExp(void *pParams) {
  auto Params =
      *static_cast<ur_queue_is_graph_capture_enabled_exp_params_t *>(pParams);
  trace("urQueueIsGraphCaptureEnabledExp", *Params.phQueue);
  **Params.ppResult = state().PrimaryCapturing.count(*Params.phQueue) != 0;
  return UR_RESULT_SUCCESS;
}

ur_result_t mock_urQueueGetGraphExp(void *pParams) {
  auto Params = *static_cast<ur_queue_get_graph_exp_params_t *>(pParams);
  trace("urQueueGetGraphExp", *Params.phQueue);
  auto It = state().PrimaryCapturing.find(*Params.phQueue);
  if (It == state().PrimaryCapturing.end())
    return UR_RESULT_ERROR_COMMAND_LIST_NOT_CAPTURING;
  **Params.pphGraph = It->second;
  return UR_RESULT_SUCCESS;
}

// A before-callback so the generated mock still produces the output event.
ur_result_t mock_urEnqueueGraphExpBefore(void *pParams) {
  auto Params = *static_cast<ur_enqueue_graph_exp_params_t *>(pParams);
  trace("urEnqueueGraphExp", *Params.phGraph);
  return UR_RESULT_SUCCESS;
}

} // namespace

// Extends entry point with UR tracing in the singleton
#define TRACE_UR_ENTRY_POINT(EntryPoint)                                       \
  mock::getCallbacks().set_before_callback(#EntryPoint,                        \
                                           [](void *) -> ur_result_t {         \
                                             trace(#EntryPoint);               \
                                             return UR_RESULT_SUCCESS;         \
                                           })

void registerDefaultCallbacks() {
  state() = MockState{};

  mock::getCallbacks().set_after_callback("urDeviceGetInfo",
                                          &mock_urDeviceGetInfoAfter);
#define REPLACE_UR_ENTRY_POINT(EntryPoint)                                     \
  mock::getCallbacks().set_replace_callback(#EntryPoint, &mock_##EntryPoint)
  REPLACE_UR_ENTRY_POINT(urGraphCreateExp);
  REPLACE_UR_ENTRY_POINT(urGraphDestroyExp);
  REPLACE_UR_ENTRY_POINT(urGraphSetDestructionCallbackExp);
  REPLACE_UR_ENTRY_POINT(urGraphInstantiateGraphExp);
  REPLACE_UR_ENTRY_POINT(urGraphExecutableGraphDestroyExp);
  REPLACE_UR_ENTRY_POINT(urGraphIsEmptyExp);
  REPLACE_UR_ENTRY_POINT(urGraphGetIdExp);
  REPLACE_UR_ENTRY_POINT(urQueueBeginCaptureIntoGraphExp);
  REPLACE_UR_ENTRY_POINT(urQueueEndGraphCaptureExp);
  REPLACE_UR_ENTRY_POINT(urQueueIsGraphCaptureEnabledExp);
  REPLACE_UR_ENTRY_POINT(urQueueGetGraphExp);
#undef REPLACE_UR_ENTRY_POINT

  mock::getCallbacks().set_before_callback("urEnqueueGraphExp",
                                           &mock_urEnqueueGraphExpBefore);

  TRACE_UR_ENTRY_POINT(urEnqueueKernelLaunchWithArgsExp);
  TRACE_UR_ENTRY_POINT(urCommandBufferCreateExp);
}
#undef TRACE_UR_ENTRY_POINT

} // namespace NativeRecordingMock

NativeRecordingTest::NativeRecordingTest()
    : Plat{sycl::platform()}, Dev{Plat.get_devices()[0]},
      Queue{Dev, {sycl::property::queue::in_order{}}} {
  NativeRecordingMock::registerDefaultCallbacks();
}

NativeRecordingTest::ModifiableGraph NativeRecordingTest::makeGraph() {
  return ModifiableGraph{
      Queue.get_context(),
      Dev,
      {experimental::property::graph::enable_native_recording{}}};
}

ur_exp_graph_handle_t
NativeRecordingTest::nativeHandle(const ModifiableGraph &Graph) {
  return getSyclObjImpl(Graph)->getNativeGraphHandle();
}

ur_exp_executable_graph_handle_t
NativeRecordingTest::nativeHandle(const ExecutableGraph &ExecGraph) {
  return getSyclObjImpl(ExecGraph)->getNativeExecutableGraphHandle();
}
