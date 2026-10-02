//==-- ReusableEventsGraph.cpp --- Signals of graph and allocation events --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// An executable graph keeps the bindings of its previous executions, and
// recording keeps graph nodes for the events it returns. The tests below
// re-signal the events of graph executions, of recorded commands and of
// asynchronous allocations, and check that replays, updates, partitions,
// native-recording boundaries and allocations keep using the signals they
// captured.
//
// The scenario numbers refer to the reusable events test plan
// (tests-10-02.md); the findings to review-10-01.md and review-10-02.md.
//
//===----------------------------------------------------------------------===//

#include "ReusableEventsFixture.hpp"

#include <detail/graph/graph_impl.hpp>
#include <sycl/ext/oneapi/experimental/async_alloc/async_alloc.hpp>
#include <sycl/ext/oneapi/experimental/async_alloc/memory_pool.hpp>

#include <atomic>
#include <string>

// exec_graph_impl befriends ::GraphImplTest; the bookkeeping of previous
// executions is read through it.
class GraphImplTest {
public:
  static std::vector<std::shared_ptr<sycl::detail::event_binding>>
  schedulerDependencies(
      const sycl::ext::oneapi::experimental::detail::exec_graph_impl &Impl) {
    sycl::ext::oneapi::experimental::detail::exec_graph_impl::ReadLock Lock(
        Impl.MMutex);
    std::vector<std::shared_ptr<sycl::detail::event_binding>> Bindings;
    for (const sycl::detail::captured_dependency &Dep :
         Impl.MSchedulerDependencies)
      Bindings.push_back(Dep.Binding);
    return Bindings;
  }
};

namespace {

using namespace reusable_events_test;

using Binding = std::shared_ptr<sycl::detail::event_binding>;
using ModifiableGraph = syclex::command_graph<syclex::graph_state::modifiable>;
using ExecutableGraph = syclex::command_graph<syclex::graph_state::executable>;

// The bindings of the previous executions an executable graph waits for.
std::vector<Binding> schedulerDependencies(const ExecutableGraph &Exec) {
  return GraphImplTest::schedulerDependencies(
      *sycl::detail::getSyclObjImpl(Exec));
}

bool contains(const std::vector<ur_event_handle_t> &List,
              ur_event_handle_t Handle) {
  return std::find(List.begin(), List.end(), Handle) != List.end();
}

bool containsBinding(const std::vector<Binding> &List, const Binding &B) {
  return std::find(List.begin(), List.end(), B) != List.end();
}

// Whether P is the only predecessor of N.
bool onlyPredecessor(const syclex::node &N, const syclex::node &P) {
  std::vector<syclex::node> Predecessors = N.get_predecessors();
  return Predecessors.size() == 1 && Predecessors[0] == P;
}

struct Rejection {
  bool Thrown = false;
  std::error_code Code;
  std::string What;
};

template <typename OperationT> Rejection rejectionOf(OperationT Operation) {
  Rejection Result;
  try {
    Operation();
  } catch (const sycl::exception &E) {
    Result.Thrown = true;
    Result.Code = E.code();
    Result.What = E.what();
  }
  return Result;
}

// Signals E on Q; the backend event of the signal stays pending until the test
// completes it.
void signalPending(sycl::queue &Q, sycl::event &E) {
  setBarriersStayPending(true);
  syclex::enqueue_signal_event(Q, E);
  setBarriersStayPending(false);
}

void addKernel(sycl::handler &CGH) {
  CGH.single_task<BindingTestKernel>([]() {});
}

// What reached the backend through the asynchronous allocations, frees and
// command-buffer updates. Guarded by BackendMutex.
std::vector<std::vector<ur_event_handle_t>> AsyncAllocWaitLists;
std::vector<ur_event_handle_t> AsyncAllocEvents;
std::vector<std::vector<ur_event_handle_t>> AsyncFreeWaitLists;
// The default mock gives an asynchronous allocation no memory.
alignas(64) unsigned char AllocStorage[8][64];
size_t NextAllocation = 0;

struct UpdateCall {
  // How many command buffers had been enqueued when the update was issued.
  size_t CommandBuffersEnqueued = 0;
  // The backend events pending when the update was issued.
  std::set<ur_event_handle_t> Pending;
};
std::vector<UpdateCall> UpdateCalls;

ur_result_t redefinedUrEnqueueUSMDeviceAllocExp(void *pParams) {
  auto params =
      *static_cast<ur_enqueue_usm_device_alloc_exp_params_t *>(pParams);
  std::lock_guard<std::mutex> Lock(BackendMutex);
  AsyncAllocWaitLists.push_back(
      recordWaitList(*params.pphEventWaitList, *params.pnumEventsInWaitList));
  **params.pppMem = AllocStorage[NextAllocation++ % 8];
  ur_event_handle_t Handle = newFakeEvent();
  producedBy(Handle, *params.phQueue, KernelsStayPending);
  AsyncAllocEvents.push_back(Handle);
  if (*params.pphEvent)
    **params.pphEvent = Handle;
  return UR_RESULT_SUCCESS;
}

ur_result_t redefinedUrEnqueueUSMFreeExp(void *pParams) {
  auto params = *static_cast<ur_enqueue_usm_free_exp_params_t *>(pParams);
  std::lock_guard<std::mutex> Lock(BackendMutex);
  AsyncFreeWaitLists.push_back(
      recordWaitList(*params.pphEventWaitList, *params.pnumEventsInWaitList));
  ur_event_handle_t Handle = newFakeEvent();
  producedBy(Handle, *params.phQueue, KernelsStayPending);
  if (*params.pphEvent)
    **params.pphEvent = Handle;
  return UR_RESULT_SUCCESS;
}

ur_result_t before_urCommandBufferUpdateKernelLaunchExp(void *) {
  std::lock_guard<std::mutex> Lock(BackendMutex);
  UpdateCalls.push_back({CommandBufferEvents.size(), PendingEvents});
  return UR_RESULT_SUCCESS;
}

class ReusableEventsGraphTest : public ReusableEventsTest {
protected:
  void SetUp() override {
    ReusableEventsTest::SetUp();
    {
      std::lock_guard<std::mutex> Lock(BackendMutex);
      AsyncAllocWaitLists.clear();
      AsyncAllocEvents.clear();
      AsyncFreeWaitLists.clear();
      NextAllocation = 0;
      UpdateCalls.clear();
    }
    mock::getCallbacks().set_replace_callback(
        "urEnqueueUSMDeviceAllocExp", &redefinedUrEnqueueUSMDeviceAllocExp);
    mock::getCallbacks().set_replace_callback("urEnqueueUSMFreeExp",
                                              &redefinedUrEnqueueUSMFreeExp);
    mock::getCallbacks().set_before_callback(
        "urCommandBufferUpdateKernelLaunchExp",
        &before_urCommandBufferUpdateKernelLaunchExp);
  }
};

// A minimal native-recording backend: graphs are dummy handles, and a queue
// captures into the graph it began capturing into until it ends. Guarded by
// NativeMutex, apart from BackendMutex, as the runtime queries the capture
// state from within enqueues.
std::mutex NativeMutex;
std::map<ur_queue_handle_t, ur_exp_graph_handle_t> CapturingQueues;
uint64_t NextGraphId = 1;

ur_result_t redefinedUrGraphCreateExp(void *pParams) {
  auto params = *static_cast<ur_graph_create_exp_params_t *>(pParams);
  **params.pphGraph = mock::createDummyHandle<ur_exp_graph_handle_t>();
  return UR_RESULT_SUCCESS;
}

ur_result_t redefinedUrGraphDestroyExp(void *pParams) {
  auto params = *static_cast<ur_graph_destroy_exp_params_t *>(pParams);
  mock::releaseDummyHandle(*params.phGraph);
  return UR_RESULT_SUCCESS;
}

ur_result_t redefinedUrGraphGetIdExp(void *pParams) {
  auto params = *static_cast<ur_graph_get_id_exp_params_t *>(pParams);
  std::lock_guard<std::mutex> Lock(NativeMutex);
  **params.ppGraphId = NextGraphId++;
  return UR_RESULT_SUCCESS;
}

ur_result_t redefinedUrGraphIsEmptyExp(void *pParams) {
  auto params = *static_cast<ur_graph_is_empty_exp_params_t *>(pParams);
  **params.ppResult = true;
  return UR_RESULT_SUCCESS;
}

ur_result_t redefinedUrQueueIsGraphCaptureEnabledExp(void *pParams) {
  auto params =
      *static_cast<ur_queue_is_graph_capture_enabled_exp_params_t *>(pParams);
  std::lock_guard<std::mutex> Lock(NativeMutex);
  **params.ppResult = CapturingQueues.count(*params.phQueue) != 0;
  return UR_RESULT_SUCCESS;
}

ur_result_t redefinedUrQueueBeginCaptureIntoGraphExp(void *pParams) {
  auto params =
      *static_cast<ur_queue_begin_capture_into_graph_exp_params_t *>(pParams);
  std::lock_guard<std::mutex> Lock(NativeMutex);
  if (!CapturingQueues.try_emplace(*params.phQueue, *params.phGraph).second)
    return UR_RESULT_ERROR_INVALID_ARGUMENT;
  return UR_RESULT_SUCCESS;
}

ur_result_t redefinedUrQueueEndGraphCaptureExp(void *pParams) {
  auto params =
      *static_cast<ur_queue_end_graph_capture_exp_params_t *>(pParams);
  std::lock_guard<std::mutex> Lock(NativeMutex);
  auto It = CapturingQueues.find(*params.phQueue);
  if (It == CapturingQueues.end())
    return UR_RESULT_ERROR_COMMAND_LIST_NOT_CAPTURING;
  **params.pphGraph = It->second;
  CapturingQueues.erase(It);
  return UR_RESULT_SUCCESS;
}

ur_result_t redefinedUrQueueGetGraphExp(void *pParams) {
  auto params = *static_cast<ur_queue_get_graph_exp_params_t *>(pParams);
  std::lock_guard<std::mutex> Lock(NativeMutex);
  auto It = CapturingQueues.find(*params.phQueue);
  if (It == CapturingQueues.end())
    return UR_RESULT_ERROR_COMMAND_LIST_NOT_CAPTURING;
  **params.pphGraph = It->second;
  return UR_RESULT_SUCCESS;
}

ur_result_t after_urDeviceGetInfoNativeRecording(void *pParams) {
  after_urDeviceGetInfo(pParams);
  auto params = *static_cast<ur_device_get_info_params_t *>(pParams);
  if (*params.ppropName == UR_DEVICE_INFO_GRAPH_RECORD_AND_REPLAY_SUPPORT_EXP) {
    if (*params.ppPropSizeRet)
      **params.ppPropSizeRet = sizeof(ur_bool_t);
    if (*params.ppPropValue)
      *static_cast<ur_bool_t *>(*params.ppPropValue) = true;
  }
  return UR_RESULT_SUCCESS;
}

class ReusableEventsNativeRecordingTest : public ReusableEventsGraphTest {
protected:
  void SetUp() override {
    ReusableEventsGraphTest::SetUp();
    {
      std::lock_guard<std::mutex> Lock(NativeMutex);
      CapturingQueues.clear();
      NextGraphId = 1;
    }
    mock::getCallbacks().set_replace_callback("urGraphCreateExp",
                                              &redefinedUrGraphCreateExp);
    mock::getCallbacks().set_replace_callback("urGraphDestroyExp",
                                              &redefinedUrGraphDestroyExp);
    mock::getCallbacks().set_replace_callback("urGraphGetIdExp",
                                              &redefinedUrGraphGetIdExp);
    mock::getCallbacks().set_replace_callback("urGraphIsEmptyExp",
                                              &redefinedUrGraphIsEmptyExp);
    mock::getCallbacks().set_replace_callback(
        "urQueueIsGraphCaptureEnabledExp",
        &redefinedUrQueueIsGraphCaptureEnabledExp);
    mock::getCallbacks().set_replace_callback(
        "urQueueBeginCaptureIntoGraphExp",
        &redefinedUrQueueBeginCaptureIntoGraphExp);
    mock::getCallbacks().set_replace_callback(
        "urQueueEndGraphCaptureExp", &redefinedUrQueueEndGraphCaptureExp);
    mock::getCallbacks().set_replace_callback("urQueueGetGraphExp",
                                              &redefinedUrQueueGetGraphExp);
    mock::getCallbacks().set_after_callback(
        "urDeviceGetInfo", &after_urDeviceGetInfoNativeRecording);
  }

  ModifiableGraph nativeGraph() {
    return ModifiableGraph{
        Ctx, Dev, {syclex::property::graph::enable_native_recording{}}};
  }
};

// tests-10-02 U31. Steps 1-4: an updatable kernel graph is executed behind a
// host task, and its returned event is re-signalled on another queue. The
// second execution and the update still wait for the original execution, not
// for the event's later signal, and the update is issued only after both
// executions reached the backend.
TEST_F(ReusableEventsGraphTest,
       ReplayAndUpdateAfterResignalWaitForOriginalExecution) {
  sycl::queue Q{Ctx, Dev};
  sycl::queue SignalQueue = inOrderQueue();
  ModifiableGraph Graph{Ctx, Dev};
  syclex::node KNode = Graph.add(addKernel);
  ExecutableGraph Exec = Graph.finalize({syclex::property::graph::updatable{}});

  auto Gate = std::make_shared<HostTaskGate>();
  OpenAtScopeExit OpenGate{Gate};
  sycl::event HT = blockQueue(Q, Gate);
  sycl::event E1 = Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on(HT);
    CGH.ext_oneapi_graph(Exec);
  });
  Binding Execution1 = bindingOf(E1);

  syclex::enqueue_signal_event(SignalQueue, E1);
  ur_event_handle_t Resignal = nullptr;
  {
    std::lock_guard<std::mutex> Lock(BackendMutex);
    ASSERT_EQ(CreatedEvents.size(), 1u);
    Resignal = CreatedEvents[0];
  }
  EXPECT_NE(bindingOf(E1), Execution1);

  sycl::event E2 = Q.ext_oneapi_graph(Exec);
  Binding Execution2 = bindingOf(E2);
  EXPECT_EQ(schedulerDependencies(Exec),
            (std::vector<Binding>{Execution1, Execution2}));

  Exec.update(KNode);
  {
    std::vector<Binding> Deps = schedulerDependencies(Exec);
    ASSERT_EQ(Deps.size(), 3u);
    EXPECT_EQ(Deps[0], Execution1);
    EXPECT_EQ(Deps[1], Execution2);
    EXPECT_FALSE(containsBinding(Deps, bindingOf(E1)));
  }
  {
    std::lock_guard<std::mutex> Lock(BackendMutex);
    EXPECT_TRUE(CommandBufferEvents.empty());
    EXPECT_TRUE(UpdateCalls.empty());
  }

  Gate->open();
  ASSERT_TRUE(eventually([] {
    return CommandBufferEvents.size() == 2 && UpdateCalls.size() == 1;
  }));
  {
    std::lock_guard<std::mutex> Lock(BackendMutex);
    EXPECT_EQ(UpdateCalls[0].CommandBuffersEnqueued, 2u);
    EXPECT_TRUE(CommandBufferWaitLists[0].empty());
    EXPECT_EQ(CommandBufferWaitLists[1],
              std::vector<ur_event_handle_t>{CommandBufferEvents[0]});
    EXPECT_TRUE(std::any_of(BarrierWaitLists.begin(), BarrierWaitLists.end(),
                            [](const auto &L) {
                              return contains(L, CommandBufferEvents[0]) &&
                                     contains(L, CommandBufferEvents[1]);
                            }));
    EXPECT_TRUE(
        std::none_of(BarrierWaitLists.begin(), BarrierWaitLists.end(),
                     [&](const auto &L) { return contains(L, Resignal); }));
    EXPECT_EQ(Execution1->getHandle(), CommandBufferEvents[0]);
  }

  Q.wait();
  EXPECT_EQ(handleOf(E1), Resignal);

  // Once the previous executions and the update completed, a new execution
  // has nothing left to wait for.
  sycl::event E3 = Q.ext_oneapi_graph(Exec);
  EXPECT_TRUE(schedulerDependencies(Exec).empty());
  Q.wait();
}

// tests-10-02 U31. Step 5, host-task partition: an update of a graph with a
// host task waits on the host for the original execution's backend event,
// even though the returned event was re-signalled and its later signal is
// complete.
TEST_F(ReusableEventsGraphTest, HostTaskGraphUpdateWaitsForOriginalExecution) {
  std::atomic<int> LastVersion{0};
  sycl::queue Q{Ctx, Dev};
  sycl::queue SignalQueue = inOrderQueue();
  auto versionedGraph = [&](int Version) {
    ModifiableGraph G{Ctx, Dev};
    syclex::node HostNode = G.add([&, Version](sycl::handler &CGH) {
      CGH.host_task([&LastVersion, Version] { LastVersion = Version; });
    });
    G.add(addKernel, {syclex::property::node::depends_on(HostNode)});
    return G;
  };
  ModifiableGraph Graph = versionedGraph(1);
  ModifiableGraph UpdateGraph = versionedGraph(2);
  ExecutableGraph Exec = Graph.finalize({syclex::property::graph::updatable{}});

  std::future<void> Update;
  CompleteAllAtScopeExit CompleteAll;
  setKernelsStayPending(true);
  sycl::event E1 = Q.ext_oneapi_graph(Exec);
  Binding Execution1 = bindingOf(E1);
  ASSERT_TRUE(eventually([] { return CommandBufferEvents.size() == 1; }));
  ur_event_handle_t First = nullptr;
  {
    std::lock_guard<std::mutex> Lock(BackendMutex);
    First = CommandBufferEvents[0];
  }
  EXPECT_EQ(LastVersion.load(), 1);

  syclex::enqueue_signal_event(SignalQueue, E1);
  ur_event_handle_t Resignal = handleOf(E1);
  EXPECT_NE(Resignal, First);
  EXPECT_FALSE(isPending(Resignal));
  EXPECT_EQ(schedulerDependencies(Exec), std::vector<Binding>{Execution1});

  Update = std::async(std::launch::async, [&] { Exec.update(UpdateGraph); });
  EXPECT_TRUE(eventually([&] {
    return std::find(WaitedEvents.begin(), WaitedEvents.end(), First) !=
           WaitedEvents.end();
  }));
  EXPECT_TRUE(stillBlocked(Update));
  {
    std::lock_guard<std::mutex> Lock(BackendMutex);
    EXPECT_TRUE(UpdateCalls.empty());
  }

  complete(First);
  ASSERT_TRUE(finishes(Update));
  EXPECT_NO_THROW(Update.get());
  {
    std::lock_guard<std::mutex> Lock(BackendMutex);
    ASSERT_EQ(UpdateCalls.size(), 1u);
    EXPECT_EQ(UpdateCalls[0].Pending.count(First), 0u);
  }

  setKernelsStayPending(false);
  sycl::event E2 = Q.ext_oneapi_graph(Exec);
  E2.wait();
  EXPECT_EQ(LastVersion.load(), 2);
  Q.wait();
}

// tests-10-02 U31. Step 5, reversed completion order with several outstanding
// executions: the later signals of both returned events complete first, and
// the second execution still waits for the first one's backend event.
TEST_F(ReusableEventsGraphTest,
       ReplaysFollowOriginalExecutionsNotLaterSignals) {
  sycl::queue Q{Ctx, Dev};
  sycl::queue SignalQueue = inOrderQueue();
  ModifiableGraph Graph{Ctx, Dev};
  syclex::node HostNode =
      Graph.add([](sycl::handler &CGH) { CGH.host_task([] {}); });
  Graph.add(addKernel, {syclex::property::node::depends_on(HostNode)});
  ExecutableGraph Exec = Graph.finalize();

  CompleteAllAtScopeExit CompleteAll;
  setKernelsStayPending(true);
  sycl::event E1 = Q.ext_oneapi_graph(Exec);
  Binding Execution1 = bindingOf(E1);
  ASSERT_TRUE(eventually([] { return CommandBufferEvents.size() == 1; }));
  ur_event_handle_t First = nullptr;
  {
    std::lock_guard<std::mutex> Lock(BackendMutex);
    First = CommandBufferEvents[0];
  }
  signalPending(SignalQueue, E1);
  ur_event_handle_t Resignal1 = handleOf(E1);

  sycl::event E2 = Q.ext_oneapi_graph(Exec);
  Binding Execution2 = bindingOf(E2);
  signalPending(SignalQueue, E2);
  ur_event_handle_t Resignal2 = handleOf(E2);
  EXPECT_EQ(schedulerDependencies(Exec),
            (std::vector<Binding>{Execution1, Execution2}));

  complete(Resignal1);
  complete(Resignal2);
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  {
    std::lock_guard<std::mutex> Lock(BackendMutex);
    EXPECT_EQ(CommandBufferEvents.size(), 1u);
  }

  complete(First);
  ASSERT_TRUE(eventually([] { return CommandBufferEvents.size() == 2; }));
  {
    std::lock_guard<std::mutex> Lock(BackendMutex);
    EXPECT_FALSE(contains(CommandBufferWaitLists[1], Resignal1));
    EXPECT_FALSE(contains(CommandBufferWaitLists[1], Resignal2));
  }
  EXPECT_TRUE(waitedFor(First));
  EXPECT_EQ(handleOf(E2), Resignal2);

  completeAll();
  Q.wait();
  SignalQueue.wait();

  sycl::event E3 = Q.ext_oneapi_graph(Exec);
  EXPECT_EQ(schedulerDependencies(Exec), std::vector<Binding>{bindingOf(E3)});
  Q.wait();
}

// A graph with independent terminal partitions: a host task blocked on Gate,
// and a host task followed by a kernel. The kernel's partition is the last
// one, so its event is returned and the gated host task is attached to it.
ExecutableGraph independentTerminals(const sycl::context &Ctx,
                                     const sycl::device &Dev,
                                     std::shared_ptr<HostTaskGate> Gate) {
  ModifiableGraph Graph{Ctx, Dev};
  Graph.add(
      [Gate](sycl::handler &CGH) { CGH.host_task([Gate] { Gate->wait(); }); });
  syclex::node B = Graph.add([](sycl::handler &CGH) { CGH.host_task([] {}); });
  Graph.add(addKernel, {syclex::property::node::depends_on(B)});
  return Graph.finalize();
}

// Executes the graph above, and waits until the returned partition reached the
// backend while the gated host task is still running.
sycl::event executeWithPendingHostTerminal(sycl::queue &Q,
                                           ExecutableGraph &Exec,
                                           HostTaskGate &Gate) {
  sycl::event E = Q.ext_oneapi_graph(Exec);
  auto Impl = sycl::detail::getSyclObjImpl(E);
  EXPECT_FALSE(Impl->isHost());
  EXPECT_EQ(Impl->getPostCompleteEvents().size(), 1u);
  EXPECT_TRUE(eventually([] { return CommandBufferEvents.size() == 1; }));
  EXPECT_TRUE(Gate.waitEntered());
  return E;
}

// tests-10-02 U32. Control: a wait on the returned event, which is not
// re-signalled, covers the independent host-task terminal.
TEST_F(ReusableEventsGraphTest, ReturnedEventWaitCoversIndependentTerminal) {
  sycl::queue Q = inOrderQueue();
  auto Gate = std::make_shared<HostTaskGate>();
  ExecutableGraph Exec = independentTerminals(Ctx, Dev, Gate);
  std::future<void> Waiter;
  OpenAtScopeExit OpenGate{Gate};
  sycl::event E = executeWithPendingHostTerminal(Q, Exec, *Gate);

  Waiter = std::async(std::launch::async, [E]() mutable { E.wait(); });
  EXPECT_TRUE(stillBlocked(Waiter));
  Gate->open();
  EXPECT_TRUE(finishes(Waiter));
  Q.wait();
}

// tests-10-02 U32. A queue wait must cover the independent host-task terminal;
// urQueueFinish alone does not.
// Known defect: review-10-02 #4 (queue waits use the captured binding, whose
// wait ignores the graph's other terminal partitions).
TEST_F(ReusableEventsGraphTest, DISABLED_QueueWaitCoversIndependentTerminal) {
  sycl::queue Q = inOrderQueue();
  auto Gate = std::make_shared<HostTaskGate>();
  ExecutableGraph Exec = independentTerminals(Ctx, Dev, Gate);
  std::future<void> Waiter;
  OpenAtScopeExit OpenGate{Gate};
  sycl::event E = executeWithPendingHostTerminal(Q, Exec, *Gate);

  Waiter = std::async(std::launch::async, [&Q] { Q.wait(); });
  EXPECT_TRUE(stillBlocked(Waiter));
  Gate->open();
  EXPECT_TRUE(finishes(Waiter));
}

// tests-10-02 U32. Steps 4-5: the returned event is re-signalled after a
// consumer captured it, before any wait on it. The consumer and the queue wait
// still cover the old execution's independent terminal, and finish once it
// completes.
// Known defect: review-10-02 #4 (the completion attachments of a graph
// execution are not preserved in the binding the consumer and queue capture).
TEST_F(ReusableEventsGraphTest,
       DISABLED_ResignaledGraphEventKeepsCompletionAttachments) {
  std::atomic<bool> ConsumerRan{false};
  sycl::queue Q = inOrderQueue();
  sycl::queue Consumers = inOrderQueue();
  sycl::queue SignalQueue = inOrderQueue();
  auto Gate = std::make_shared<HostTaskGate>();
  ExecutableGraph Exec = independentTerminals(Ctx, Dev, Gate);
  std::future<void> QueueWaiter;
  std::future<void> ConsumerWaiter;
  OpenAtScopeExit OpenGate{Gate};
  sycl::event E = executeWithPendingHostTerminal(Q, Exec, *Gate);

  Consumers.submit([&](sycl::handler &CGH) {
    CGH.depends_on(E);
    CGH.host_task([&ConsumerRan] { ConsumerRan = true; });
  });
  syclex::enqueue_signal_event(SignalQueue, E);

  QueueWaiter = std::async(std::launch::async, [&Q] { Q.wait(); });
  ConsumerWaiter =
      std::async(std::launch::async, [&Consumers] { Consumers.wait(); });
  EXPECT_TRUE(stillBlocked(QueueWaiter));
  EXPECT_TRUE(stillBlocked(ConsumerWaiter));
  EXPECT_FALSE(ConsumerRan);

  Gate->open();
  EXPECT_TRUE(finishes(QueueWaiter));
  EXPECT_TRUE(finishes(ConsumerWaiter));
  EXPECT_TRUE(ConsumerRan);
  SignalQueue.wait();
}

// tests-10-02 U33. Steps 1-3 and 5: dependencies on recorded events resolve to
// the recorded nodes of a multi-root device/host/device graph. Invalid
// signals while recording are rejected and leave the nodes, the edges and the
// events unchanged.
TEST_F(ReusableEventsGraphTest, RecordedDependenciesResolveToGraphNodes) {
  sycl::queue RecQ{Ctx, Dev};
  sycl::queue Q = inOrderQueue();
  sycl::queue SignalQueue = inOrderQueue();
  sycl::event External = syclex::make_event(Ctx);
  syclex::enqueue_signal_event(SignalQueue, External);
  SignalQueue.wait();

  ModifiableGraph Graph{Ctx, Dev};
  Graph.begin_recording(RecQ);
  sycl::event R1 = RecQ.submit(addKernel);
  sycl::event R2 = RecQ.submit([&](sycl::handler &CGH) {
    CGH.depends_on(R1);
    CGH.host_task([] {});
  });
  sycl::event R3 = RecQ.submit([&](sycl::handler &CGH) {
    CGH.depends_on(R2);
    addKernel(CGH);
  });
  sycl::event R4 = RecQ.submit(addKernel);

  syclex::node N1 = syclex::node::get_node_from_event(R1);
  syclex::node N2 = syclex::node::get_node_from_event(R2);
  syclex::node N3 = syclex::node::get_node_from_event(R3);
  syclex::node N4 = syclex::node::get_node_from_event(R4);
  auto expectTopology = [&] {
    EXPECT_EQ(Graph.get_nodes().size(), 4u);
    EXPECT_EQ(Graph.get_root_nodes().size(), 2u);
    EXPECT_TRUE(N1.get_predecessors().empty());
    EXPECT_TRUE(onlyPredecessor(N2, N1));
    EXPECT_TRUE(onlyPredecessor(N3, N2));
    EXPECT_TRUE(N4.get_predecessors().empty());
  };
  EXPECT_EQ(N1.get_type(), syclex::node_type::kernel);
  EXPECT_EQ(N2.get_type(), syclex::node_type::host_task);
  EXPECT_EQ(N3.get_type(), syclex::node_type::kernel);
  EXPECT_EQ(N4.get_type(), syclex::node_type::kernel);
  expectTopology();

  Binding ExternalBinding = bindingOf(External);
  ur_event_handle_t ExternalHandle = handleOf(External);
  Rejection Result =
      rejectionOf([&] { syclex::enqueue_signal_event(RecQ, External); });
  EXPECT_TRUE(Result.Thrown);
  EXPECT_EQ(Result.Code, sycl::make_error_code(sycl::errc::runtime));
  EXPECT_EQ(bindingOf(External), ExternalBinding);
  EXPECT_EQ(handleOf(External), ExternalHandle);
  expectTopology();

  // A recorded event stands for a graph node, not for a signal.
  Binding RecordedBinding = bindingOf(R1);
  Result = rejectionOf([&] { syclex::enqueue_signal_event(Q, R1); });
  EXPECT_TRUE(Result.Thrown);
  EXPECT_EQ(Result.Code, sycl::make_error_code(sycl::errc::invalid));
  EXPECT_EQ(bindingOf(R1), RecordedBinding);
  EXPECT_TRUE(syclex::node::get_node_from_event(R1) == N1);
  expectTopology();

  Graph.end_recording(RecQ);
  EXPECT_EQ(kernelLaunches(), 0u);
}

// tests-10-02 U33. Steps 2 and 4: a graph with two root partitions and a host
// partition between kernels. The root partitions both get the execution's
// dependencies restored, the second execution waits for the first one's
// terminal partitions, and re-signalling the first execution's returned event
// after the second captured it neither changes the topology nor substitutes
// the later signal.
TEST_F(ReusableEventsGraphTest,
       ResignalAfterDownstreamCaptureKeepsPartitionDependencies) {
  sycl::queue Q{Ctx, Dev};
  sycl::queue SignalQueue = inOrderQueue();
  sycl::event ExtE = syclex::make_event(Ctx);
  ModifiableGraph Graph{Ctx, Dev};
  Graph.add(addKernel);
  syclex::node K1 = Graph.add(addKernel);
  syclex::node H = Graph.add([](sycl::handler &CGH) { CGH.host_task([] {}); },
                             {syclex::property::node::depends_on(K1)});
  Graph.add(addKernel, {syclex::property::node::depends_on(H)});
  ExecutableGraph Exec = Graph.finalize();

  auto Gate = std::make_shared<HostTaskGate>();
  CompleteAllAtScopeExit CompleteAll;
  OpenAtScopeExit OpenGate{Gate};
  signalPending(SignalQueue, ExtE);
  ur_event_handle_t S1 = handleOf(ExtE);

  sycl::event HT = blockQueue(Q, Gate);
  sycl::event E1 = Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on(HT);
    CGH.depends_on(ExtE);
    CGH.ext_oneapi_graph(Exec);
  });
  Binding Orig1 = bindingOf(E1);

  syclex::enqueue_signal_event(SignalQueue, ExtE);
  ur_event_handle_t S2 = handleOf(ExtE);
  sycl::event E2 = Q.ext_oneapi_graph(Exec);
  EXPECT_TRUE(containsBinding(schedulerDependencies(Exec), Orig1));

  syclex::enqueue_signal_event(SignalQueue, E1);
  ur_event_handle_t S3 = handleOf(E1);
  {
    std::vector<Binding> Deps = schedulerDependencies(Exec);
    EXPECT_TRUE(containsBinding(Deps, Orig1));
    EXPECT_FALSE(containsBinding(Deps, bindingOf(E1)));
  }
  EXPECT_EQ(Graph.get_nodes().size(), 4u);

  Gate->open();
  complete(S1);
  ASSERT_TRUE(eventually([] { return CommandBufferEvents.size() == 6; }));
  {
    std::lock_guard<std::mutex> Lock(BackendMutex);
    auto listsContaining = [](ur_event_handle_t Handle) {
      return std::count_if(CommandBufferWaitLists.begin(),
                           CommandBufferWaitLists.end(),
                           [&](const auto &L) { return contains(L, Handle); });
    };
    // The two root partitions of the first execution.
    EXPECT_EQ(listsContaining(S1), 2);
    EXPECT_EQ(listsContaining(S2), 0);
    EXPECT_EQ(listsContaining(S3), 0);
    // The two root partitions of the second execution.
    EXPECT_EQ(listsContaining(Orig1->getHandle()), 2);
  }
  Q.wait();
  SignalQueue.wait();
}

// tests-10-02 U33. Steps 6-7: waits on events from outside the graph while a
// queue is recording. At the head an ordinary event reaches the graph branch
// and is rejected by the edge lookup, while an exported or imported IPC event
// is rejected before it as one which is not enqueued in the backend; an event
// recorded into the same graph is rejected as a host event, so the plan's
// passing same-graph control holds only for handler::depends_on. None of them
// changes the graph or the event.
// Investigation (tests-10-02 U33): pins behaviour at 8337da70; oracle not
// agreed.
TEST_F(ReusableEventsGraphTest, OutsideGraphWaitWhileRecordingPinned) {
  sycl::queue RecQ{Ctx, Dev};
  sycl::queue SignalQueue = inOrderQueue();
  sycl::event ExtE = syclex::make_event(Ctx);
  syclex::enqueue_signal_event(SignalQueue, ExtE);
  sycl::event Exported =
      syclex::make_event(Ctx, syclex::properties{syclex::enable_ipc{true}});
  syclex::enqueue_signal_event(SignalQueue, Exported);
  std::byte HandleBytes[8] = {};
  syclex::ipc::handle_data_t HandleData{HandleBytes,
                                        HandleBytes + sizeof(HandleBytes)};
  sycl::event Imported = syclex::ipc::event::open(HandleData, Ctx);
  SignalQueue.wait();

  ModifiableGraph Graph{Ctx, Dev};
  Graph.begin_recording(RecQ);
  sycl::event R = RecQ.submit(addKernel);
  syclex::node RNode = syclex::node::get_node_from_event(R);

  auto expectRejected = [&](sycl::event &E, const std::string &Message) {
    Binding Before = bindingOf(E);
    ur_event_handle_t HandleBefore = handleOf(E);
    Rejection Result =
        rejectionOf([&] { syclex::enqueue_wait_event(RecQ, E); });
    EXPECT_TRUE(Result.Thrown);
    EXPECT_EQ(Result.Code, sycl::make_error_code(sycl::errc::invalid));
    EXPECT_NE(Result.What.find(Message), std::string::npos) << Result.What;
    EXPECT_EQ(Graph.get_nodes().size(), 1u);
    EXPECT_EQ(bindingOf(E), Before);
    EXPECT_EQ(handleOf(E), HandleBefore);
  };
  expectRejected(ExtE, "not correspond to a node");
  expectRejected(Exported, "IPC event");
  expectRejected(Imported, "IPC event");
  expectRejected(R, "Host events");

  // The same-graph dependency through the handler is accepted.
  sycl::event Passing = RecQ.submit([&](sycl::handler &CGH) {
    CGH.depends_on(R);
    addKernel(CGH);
  });
  EXPECT_EQ(Graph.get_nodes().size(), 2u);
  EXPECT_TRUE(
      onlyPredecessor(syclex::node::get_node_from_event(Passing), RNode));

  Graph.end_recording(RecQ);
}

// tests-10-02 U34. Recorded -> nonrecorded boundary: a signal of E enqueued
// while Q was natively recording is captured by a later consumer on Q, and E
// is then re-signalled on Q outside the recording. The consumer keeps the
// recorded signal in its wait list: the same-queue optimisation follows the
// captured signal's recording state, not E's latest signal.
TEST_F(ReusableEventsNativeRecordingTest,
       RecordedSignalCapturedAcrossBoundary) {
  sycl::queue Q = inOrderQueue();
  sycl::event E = syclex::make_event(Ctx);
  ModifiableGraph Graph = nativeGraph();
  auto Gate = std::make_shared<HostTaskGate>();
  OpenAtScopeExit OpenGate{Gate};

  Graph.begin_recording(Q);
  syclex::enqueue_signal_event(Q, E);
  Graph.end_recording(Q);
  Binding Recorded = bindingOf(E);
  ur_event_handle_t S1 = handleOf(E);
  ASSERT_NE(S1, nullptr);
  EXPECT_TRUE(Recorded->MPotentiallyNativeRecorded);
  EXPECT_EQ(Recorded->MWorkerQueue.lock().get(),
            sycl::detail::getSyclObjImpl(Q).get());

  blockQueue(Q, Gate);
  Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on(E);
    addKernel(CGH);
  });
  syclex::enqueue_signal_event(Q, E);
  Binding Later = bindingOf(E);
  EXPECT_NE(Later, Recorded);

  Gate->open();
  Q.wait();
  EXPECT_FALSE(Later->MPotentiallyNativeRecorded);
  EXPECT_EQ(Later->MWorkerQueue.lock().get(),
            sycl::detail::getSyclObjImpl(Q).get());
  std::lock_guard<std::mutex> Lock(BackendMutex);
  ASSERT_EQ(KernelLaunchWaitLists.size(), 1u);
  EXPECT_TRUE(contains(KernelLaunchWaitLists[0], S1));
}

// tests-10-02 U34. Same-side control: a signal of E enqueued on Q outside any
// recording is captured by a consumer on Q, and E is then re-signalled while
// another queue records natively. The consumer is enqueued after the
// recording ended, so the same-queue optimisation still drops the captured
// signal, and the later signal is not substituted for it.
TEST_F(ReusableEventsNativeRecordingTest,
       UnrecordedSignalKeepsSameQueueOptimisation) {
  sycl::queue Q = inOrderQueue();
  sycl::queue RecQ = inOrderQueue();
  sycl::event E = syclex::make_event(Ctx);
  ModifiableGraph Graph = nativeGraph();
  auto Gate = std::make_shared<HostTaskGate>();
  OpenAtScopeExit OpenGate{Gate};

  syclex::enqueue_signal_event(Q, E);
  ur_event_handle_t S1 = handleOf(E);
  EXPECT_FALSE(bindingOf(E)->MPotentiallyNativeRecorded);
  blockQueue(Q, Gate);
  Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on(E);
    addKernel(CGH);
  });

  Graph.begin_recording(RecQ);
  syclex::enqueue_signal_event(RecQ, E);
  ur_event_handle_t S2 = handleOf(E);
  Graph.end_recording(RecQ);
  EXPECT_TRUE(bindingOf(E)->MPotentiallyNativeRecorded);

  Gate->open();
  Q.wait();
  RecQ.wait();
  std::lock_guard<std::mutex> Lock(BackendMutex);
  ASSERT_EQ(KernelLaunchWaitLists.size(), 1u);
  EXPECT_FALSE(contains(KernelLaunchWaitLists[0], S1));
  EXPECT_FALSE(contains(KernelLaunchWaitLists[0], S2));
}

// tests-10-02 U34. Nonrecorded -> recorded boundary: a consumer recorded on Q
// keeps the dependency on a signal enqueued on Q before the recording, while
// a consumer enqueued after the recording ended may drop it again.
TEST_F(ReusableEventsNativeRecordingTest,
       RecordingConsumerKeepsUnrecordedSignal) {
  sycl::queue Q = inOrderQueue();
  sycl::event E = syclex::make_event(Ctx);
  ModifiableGraph Graph = nativeGraph();

  syclex::enqueue_signal_event(Q, E);
  ur_event_handle_t S1 = handleOf(E);

  Graph.begin_recording(Q);
  Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on(E);
    addKernel(CGH);
  });
  std::vector<std::vector<ur_event_handle_t>> Recorded;
  {
    std::lock_guard<std::mutex> Lock(BackendMutex);
    Recorded = KernelLaunchWaitLists;
  }
  Graph.end_recording(Q);
  ASSERT_EQ(Recorded.size(), 1u);
  EXPECT_TRUE(contains(Recorded[0], S1));

  Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on(E);
    addKernel(CGH);
  });
  Q.wait();
  std::lock_guard<std::mutex> Lock(BackendMutex);
  ASSERT_EQ(KernelLaunchWaitLists.size(), 2u);
  EXPECT_FALSE(contains(KernelLaunchWaitLists[1], S1));
}

// tests-10-02 U35. Steps 1-3: handler-based allocations from the default pool
// and from a pool, and frees, depend on E's pending signal; E is re-signalled
// on another queue before the frees reach the backend. Every wait list has the
// captured signal and only handles the backend created; the frees do not wait
// for the later signal instead.
TEST_F(ReusableEventsGraphTest, AsyncAllocAndFreeUseCapturedSignal) {
  syclex::memory_pool Pool{Ctx, Dev, sycl::usm::alloc::device};
  sycl::queue Q = inOrderQueue();
  sycl::queue SignalQueue = inOrderQueue();
  sycl::queue OtherSignalQueue = inOrderQueue();
  sycl::event E = syclex::make_event(Ctx);
  CompleteAllAtScopeExit CompleteAll;

  signalPending(SignalQueue, E);
  ur_event_handle_t S1 = handleOf(E);
  void *Ptr = nullptr;
  void *PoolPtr = nullptr;
  Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on(E);
    Ptr = syclex::async_malloc(CGH, sycl::usm::alloc::device, 64);
  });
  Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on(E);
    PoolPtr = syclex::async_malloc_from_pool(CGH, 64, Pool);
  });
  {
    std::lock_guard<std::mutex> Lock(BackendMutex);
    ASSERT_EQ(AsyncAllocWaitLists.size(), 2u);
    for (const auto &WaitList : AsyncAllocWaitLists) {
      EXPECT_TRUE(contains(WaitList, S1));
      EXPECT_TRUE(std::all_of(
          WaitList.begin(), WaitList.end(),
          [](ur_event_handle_t H) { return KnownHandles.count(H) != 0; }));
    }
  }
  ASSERT_NE(Ptr, nullptr);
  ASSERT_NE(PoolPtr, nullptr);

  auto Gate = std::make_shared<HostTaskGate>();
  OpenAtScopeExit OpenGate{Gate};
  blockQueue(Q, Gate);
  Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on(E);
    syclex::async_free(CGH, Ptr);
  });
  Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on(E);
    syclex::async_free(CGH, PoolPtr);
  });
  syclex::enqueue_signal_event(OtherSignalQueue, E);
  ur_event_handle_t S2 = handleOf(E);
  EXPECT_NE(S2, S1);
  EXPECT_FALSE(isPending(S2));

  Gate->open();
  ASSERT_TRUE(eventually([] { return AsyncFreeWaitLists.size() == 2; }));
  {
    std::lock_guard<std::mutex> Lock(BackendMutex);
    for (const auto &WaitList : AsyncFreeWaitLists) {
      EXPECT_TRUE(contains(WaitList, S1));
      EXPECT_FALSE(contains(WaitList, S2));
    }
  }
  EXPECT_TRUE(isPending(S1));

  complete(S1);
  Q.wait();
  // Step 5, a host-deferred dependency, is not covered: the handler-based
  // allocation reads the backend events of its dependencies when it is called,
  // and a signal still held in the runtime has none yet; the allocation
  // contract does not specify that case.
}

// tests-10-02 U35. Step 4: allocations, from the default pool and from a pool,
// and frees recorded into a graph depend on the recorded nodes of the events
// they depend on. A dependency on an event from outside the graph is rejected
// by the node lookup and leaves the graph unchanged.
TEST_F(ReusableEventsGraphTest, GraphRecordingAsyncAllocUsesGraphIdentity) {
  syclex::memory_pool Pool{Ctx, Dev, sycl::usm::alloc::device};
  sycl::queue RecQ{Ctx, Dev};
  sycl::queue SignalQueue = inOrderQueue();
  sycl::event ExtE = syclex::make_event(Ctx);
  syclex::enqueue_signal_event(SignalQueue, ExtE);
  SignalQueue.wait();
  constexpr size_t Size = 1 << 16;

  ModifiableGraph Graph{Ctx, Dev};
  Graph.begin_recording(RecQ);
  void *Ptr = nullptr;
  void *PoolPtr = nullptr;
  sycl::event K = RecQ.submit(addKernel);
  sycl::event Malloc = RecQ.submit([&](sycl::handler &CGH) {
    CGH.depends_on(K);
    Ptr = syclex::async_malloc(CGH, sycl::usm::alloc::device, Size);
  });
  sycl::event PoolMalloc = RecQ.submit([&](sycl::handler &CGH) {
    CGH.depends_on(Malloc);
    PoolPtr = syclex::async_malloc_from_pool(CGH, Size, Pool);
  });
  sycl::event Use = RecQ.submit([&](sycl::handler &CGH) {
    CGH.depends_on(PoolMalloc);
    addKernel(CGH);
  });
  sycl::event Free = RecQ.submit([&](sycl::handler &CGH) {
    CGH.depends_on(Use);
    syclex::async_free(CGH, Ptr);
  });
  sycl::event PoolFree = RecQ.submit([&](sycl::handler &CGH) {
    CGH.depends_on(Free);
    syclex::async_free(CGH, PoolPtr);
  });
  EXPECT_NE(Ptr, nullptr);
  EXPECT_NE(PoolPtr, nullptr);
  EXPECT_NE(Ptr, PoolPtr);

  syclex::node KNode = syclex::node::get_node_from_event(K);
  syclex::node MallocNode = syclex::node::get_node_from_event(Malloc);
  syclex::node PoolMallocNode = syclex::node::get_node_from_event(PoolMalloc);
  syclex::node UseNode = syclex::node::get_node_from_event(Use);
  syclex::node FreeNode = syclex::node::get_node_from_event(Free);
  syclex::node PoolFreeNode = syclex::node::get_node_from_event(PoolFree);
  EXPECT_EQ(MallocNode.get_type(), syclex::node_type::async_malloc);
  EXPECT_EQ(PoolMallocNode.get_type(), syclex::node_type::async_malloc);
  EXPECT_EQ(FreeNode.get_type(), syclex::node_type::async_free);
  EXPECT_EQ(PoolFreeNode.get_type(), syclex::node_type::async_free);
  EXPECT_TRUE(onlyPredecessor(MallocNode, KNode));
  EXPECT_TRUE(onlyPredecessor(PoolMallocNode, MallocNode));
  EXPECT_TRUE(onlyPredecessor(UseNode, PoolMallocNode));
  EXPECT_TRUE(onlyPredecessor(FreeNode, UseNode));
  EXPECT_TRUE(onlyPredecessor(PoolFreeNode, FreeNode));
  EXPECT_EQ(Graph.get_nodes().size(), 6u);

  auto expectRejected = [&](auto Allocate) {
    Rejection Result = rejectionOf([&] {
      RecQ.submit([&](sycl::handler &CGH) {
        CGH.depends_on(ExtE);
        Allocate(CGH);
      });
    });
    EXPECT_TRUE(Result.Thrown);
    EXPECT_EQ(Result.Code, sycl::make_error_code(sycl::errc::invalid));
    // Rejected by the handler's dependency check (event_deps.hpp).
    EXPECT_NE(
        Result.What.find("cannot depend on events from outside the graph"),
        std::string::npos)
        << Result.What;
    EXPECT_EQ(Graph.get_nodes().size(), 6u);
  };
  expectRejected([&](sycl::handler &CGH) {
    syclex::async_malloc(CGH, sycl::usm::alloc::device, Size);
  });
  expectRejected([&](sycl::handler &CGH) {
    syclex::async_malloc_from_pool(CGH, Size, Pool);
  });

  Graph.end_recording(RecQ);
  std::lock_guard<std::mutex> Lock(BackendMutex);
  EXPECT_TRUE(AsyncAllocWaitLists.empty());
  EXPECT_TRUE(AsyncFreeWaitLists.empty());
}

// The queue helpers of the asynchronous allocations order an allocation on an
// in-order queue after the queue's last command: a pending kernel whose
// returned event is then re-signalled on another queue. The allocation waits
// for the kernel, not for the later signal.
template <typename AllocateT>
void expectHelperWaitsForLastCommand(sycl::queue &Q, sycl::queue &SignalQueue,
                                     AllocateT Allocate) {
  CompleteAllAtScopeExit CompleteAll;
  auto Gate = std::make_shared<HostTaskGate>();
  OpenAtScopeExit OpenGate{Gate};
  setKernelsStayPending(true);
  blockQueue(Q, Gate);
  sycl::event KernelEvent = Q.submit(addKernel);
  Gate->open();
  ASSERT_TRUE(eventually([] { return KernelEvents.size() == 1; }));
  ur_event_handle_t KernelHandle = nullptr;
  {
    std::lock_guard<std::mutex> Lock(BackendMutex);
    KernelHandle = KernelEvents[0];
  }
  signalPending(SignalQueue, KernelEvent);
  ur_event_handle_t SignalHandle = handleOf(KernelEvent);

  Allocate(Q);
  {
    std::lock_guard<std::mutex> Lock(BackendMutex);
    ASSERT_EQ(AsyncAllocWaitLists.size(), 1u);
    EXPECT_TRUE(contains(AsyncAllocWaitLists[0], KernelHandle));
    EXPECT_FALSE(contains(AsyncAllocWaitLists[0], SignalHandle));
  }
  completeAll();
  Q.wait();
  SignalQueue.wait();
}

// tests-10-02 U35. Step 3, queue helper of the default-pool allocation.
// Known defect: review-10-02 #7 (the helper orders the allocation through
// ext_oneapi_get_last_event, which returns the event, whose current signal is
// the later one on another queue).
TEST_F(ReusableEventsGraphTest,
       DISABLED_AsyncAllocQueueHelperUsesLastCapturedSignal) {
  sycl::queue Q = inOrderQueue();
  sycl::queue SignalQueue = inOrderQueue();
  expectHelperWaitsForLastCommand(Q, SignalQueue, [](sycl::queue &Q) {
    syclex::async_malloc(Q, sycl::usm::alloc::device, 64);
  });
}

// tests-10-02 U35. Step 3, queue helper of the pool allocation.
// Known defect: review-10-02 #7 (the helper orders the allocation through
// ext_oneapi_get_last_event, which returns the event, whose current signal is
// the later one on another queue).
TEST_F(ReusableEventsGraphTest,
       DISABLED_AsyncPoolAllocQueueHelperUsesLastCapturedSignal) {
  syclex::memory_pool Pool{Ctx, Dev, sycl::usm::alloc::device};
  sycl::queue Q = inOrderQueue();
  sycl::queue SignalQueue = inOrderQueue();
  expectHelperWaitsForLastCommand(Q, SignalQueue, [&Pool](sycl::queue &Q) {
    syclex::async_malloc_from_pool(Q, 64, Pool);
  });
}

} // anonymous namespace
