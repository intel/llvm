//==-- ReusableEventsIpc.cpp --- IPC events and invalid signals -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The backend event of an IPC event, exported with enable_ipc or imported with
// ipc::event::open, is shared with another process. Every signal of the event
// uses it, so a new binding of the event shares the handle of the previous
// one, and a signal or wait of the event cannot be held in the scheduler: the
// runtime rejects those (queue_impl.cpp, submit_barrier_direct_impl). The
// tests below check that a rejection, and the other invalid operations on a
// reusable event, leave the previous signal as it was, and that the shared
// handle is owned correctly across bindings.
//
// The scenario numbers refer to the reusable events test plan
// (tests-10-02.md).
//
//===----------------------------------------------------------------------===//

#include "ReusableEventsFixture.hpp"

#include <atomic>
#include <optional>
#include <string>
#include <tuple>

namespace {

using namespace reusable_events_test;

// The handle data the mock exports for every IPC event.
std::byte ExportedHandleData[8] = {std::byte{1}, std::byte{2}, std::byte{3},
                                   std::byte{4}, std::byte{5}, std::byte{6},
                                   std::byte{7}, std::byte{8}};

std::atomic<int> IpcGets{0};
std::atomic<int> IpcPuts{0};

ur_result_t redefinedUrIPCGetEventHandleExp(void *pParams) {
  auto params = *static_cast<ur_ipc_get_event_handle_exp_params_t *>(pParams);
  ++IpcGets;
  if (*params.pppIPCEventHandleData)
    **params.pppIPCEventHandleData = ExportedHandleData;
  if (*params.ppIPCEventHandleDataSizeRet)
    **params.ppIPCEventHandleDataSizeRet = sizeof(ExportedHandleData);
  return UR_RESULT_SUCCESS;
}

ur_result_t redefinedUrIPCPutEventHandleExp(void *pParams) {
  auto params = *static_cast<ur_ipc_put_event_handle_exp_params_t *>(pParams);
  ++IpcPuts;
  EXPECT_EQ(*params.ppIPCEventHandleData,
            static_cast<void *>(ExportedHandleData));
  return UR_RESULT_SUCCESS;
}

// Exports the handle data of IPC events through the mock as well.
class ReusableEventsIpcTest : public ReusableEventsTest {
protected:
  void SetUp() override {
    ReusableEventsTest::SetUp();
    IpcGets = 0;
    IpcPuts = 0;
    mock::getCallbacks().set_replace_callback("urIPCGetEventHandleExp",
                                              &redefinedUrIPCGetEventHandleExp);
    mock::getCallbacks().set_replace_callback("urIPCPutEventHandleExp",
                                              &redefinedUrIPCPutEventHandleExp);
  }
};

// The same, without native reusable-event, IPC or per-event profiling support.
class ReusableEventsIpcNoSupportTest : public ReusableEventsIpcTest {
protected:
  bool nativeSupport() const override { return false; }
};

enum class EventKind { Ordinary, Exported, Imported };

std::string kindName(EventKind Kind) {
  switch (Kind) {
  case EventKind::Ordinary:
    return "Ordinary";
  case EventKind::Exported:
    return "Exported";
  case EventKind::Imported:
    return "Imported";
  }
  return "Unknown";
}

// An event of the given kind which has a backend event: an ordinary or an
// exported event signaled once on SignalQueue, through the scheduler bypass,
// or an event imported from another process.
sycl::event makeSignaledEvent(EventKind Kind, const sycl::context &Ctx,
                              sycl::queue &SignalQueue) {
  if (Kind == EventKind::Imported) {
    std::byte HandleBytes[8] = {};
    syclex::ipc::handle_data_t HandleData{HandleBytes,
                                          HandleBytes + sizeof(HandleBytes)};
    return syclex::ipc::event::open(HandleData, Ctx);
  }
  sycl::event E = Kind == EventKind::Exported
                      ? syclex::make_event(
                            Ctx, syclex::properties{syclex::enable_ipc{true}})
                      : syclex::make_event(Ctx);
  syclex::enqueue_signal_event(SignalQueue, E);
  SignalQueue.wait();
  return E;
}

template <typename OperationT>
void expectError(sycl::errc Code, OperationT &&Operation, const char *What) {
  try {
    Operation();
    ADD_FAILURE() << What << " did not throw";
  } catch (const sycl::exception &Ex) {
    EXPECT_EQ(Ex.code(), sycl::make_error_code(Code)) << What;
  }
}

// What a rejected operation must leave alone: the event's binding and backend
// event, and the backend calls made so far.
struct SignalSnapshot {
  explicit SignalSnapshot(const sycl::event &E)
      : Binding{bindingOf(E)}, Handle{handleOf(E)}, Command{Binding->MCommand} {
    std::lock_guard<std::mutex> Lock(BackendMutex);
    Created = CreatedEvents.size();
    Retains = RetainCounts[Handle];
    Releases = ReleaseCounts[Handle];
    Barriers = BarrierWaitLists.size();
    Launches = KernelLaunchWaitLists.size();
  }

  void expectIntact(const sycl::event &E, const char *What) const {
    EXPECT_EQ(bindingOf(E), Binding) << What;
    EXPECT_EQ(handleOf(E), Handle) << What;
    EXPECT_EQ(Binding->getHandle(), Handle) << What;
    EXPECT_EQ(Binding->MCommand, Command) << What;
    std::lock_guard<std::mutex> Lock(BackendMutex);
    EXPECT_EQ(CreatedEvents.size(), Created) << What;
    EXPECT_EQ(RetainCounts[Handle], Retains) << What;
    EXPECT_EQ(ReleaseCounts[Handle], Releases) << What;
    EXPECT_EQ(BarrierWaitLists.size(), Barriers) << What;
    EXPECT_EQ(KernelLaunchWaitLists.size(), Launches) << What;
  }

  std::shared_ptr<sycl::detail::event_binding> Binding;
  ur_event_handle_t Handle;
  sycl::detail::Command *Command;
  size_t Created = 0;
  int Retains = 0;
  int Releases = 0;
  size_t Barriers = 0;
  size_t Launches = 0;
};

//===----------------------------------------------------------------------===//
// tests-10-02 U36: deferred signals and waits of IPC events are rejected.
//===----------------------------------------------------------------------===//

class ReusableEventsIpcKindTest
    : public ReusableEventsIpcTest,
      public ::testing::WithParamInterface<EventKind> {};

INSTANTIATE_TEST_SUITE_P(ReusableEventsIpc, ReusableEventsIpcKindTest,
                         ::testing::Values(EventKind::Exported,
                                           EventKind::Imported),
                         [](const ::testing::TestParamInfo<EventKind> &Info) {
                           return kindName(Info.param);
                         });

// tests-10-02 U36. A signal of an IPC event behind a kernel held in the
// scheduler throws, before the event is moved on to a new binding: the
// binding a consumer captured, the shared handle and its ownership, and an
// ordinary event next to it are untouched. Once the queue drains, the event is
// signaled through the scheduler bypass with the same shared handle, and an
// ordinary event can still be signaled behind the held kernel.
TEST_P(ReusableEventsIpcKindTest, DeferredSignalRejectedWithoutMutation) {
  sycl::queue SignalQueue{Ctx, Dev};
  sycl::queue Q = inOrderQueue();
  sycl::event E = makeSignaledEvent(GetParam(), Ctx, SignalQueue);
  const ur_event_handle_t Shared = handleOf(E);
  ASSERT_NE(Shared, nullptr);
  sycl::event Plain = makeSignaledEvent(EventKind::Ordinary, Ctx, SignalQueue);

  auto Gate = std::make_shared<HostTaskGate>();
  OpenAtScopeExit OpenGate{Gate};
  blockQueue(Q, Gate);
  // Held behind the host task; captures the current binding of E. The queue
  // now ends with a command which is not in the backend.
  sycl::event Held = Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on(E);
    CGH.single_task<BindingTestKernel>([]() {});
  });

  const SignalSnapshot Before{E};
  const SignalSnapshot PlainBefore{Plain};
  expectError(
      sycl::errc::invalid, [&] { syclex::enqueue_signal_event(Q, E); },
      "a deferred signal of an IPC event");
  Before.expectIntact(E, "after the rejected signal");
  PlainBefore.expectIntact(Plain, "the ordinary event");
  EXPECT_EQ(handleOf(Held), nullptr);

  // An ordinary event is still signaled behind the held kernel.
  EXPECT_NO_THROW(syclex::enqueue_signal_event(Q, Plain));
  EXPECT_EQ(handleOf(Plain), nullptr);

  Gate->open();
  Q.wait();
  EXPECT_NE(handleOf(Plain), nullptr);
  EXPECT_TRUE(isComplete(Plain));
  {
    std::lock_guard<std::mutex> Lock(BackendMutex);
    ASSERT_EQ(KernelLaunchWaitLists.size(), Before.Launches + 1);
    EXPECT_EQ(KernelLaunchWaitLists[Before.Launches],
              std::vector<ur_event_handle_t>{Shared});
  }

  // Nothing is held any more: the signal goes through the scheduler bypass,
  // with the shared backend event.
  EXPECT_NO_THROW(syclex::enqueue_signal_event(Q, E));
  Q.wait();
  EXPECT_EQ(handleOf(E), Shared);
  {
    std::lock_guard<std::mutex> Lock(BackendMutex);
    ASSERT_FALSE(BarrierOutEvents.empty());
    EXPECT_EQ(BarrierOutEvents.back(), Shared);
  }
  EXPECT_EQ(releases(Shared), Before.Releases);
}

// tests-10-02 U36. A wait for an IPC event throws whenever it would have to be
// held in the scheduler: behind a kernel held in the queue, alone or with an
// ordinary event; on an idle queue, with a dependency which is not in the
// backend yet, or with an event of another context; and on a queue of another
// context. None of these changes the event, the binding a consumer captured,
// or the ordinary events in the list. The same waits without the IPC event are
// held as usual, and the waits on the queues of the event's context succeed
// once nothing is held any more.
TEST_P(ReusableEventsIpcKindTest, DeferredWaitRejectedWithoutMutation) {
  sycl::queue SignalQueue{Ctx, Dev};
  sycl::queue Q = inOrderQueue();
  sycl::queue Idle = inOrderQueue();
  sycl::context OtherCtx{Dev};
  sycl::queue OtherQ = inOrderQueue(OtherCtx);
  sycl::event E = makeSignaledEvent(GetParam(), Ctx, SignalQueue);
  const ur_event_handle_t Shared = handleOf(E);
  ASSERT_NE(Shared, nullptr);
  sycl::event Plain = makeSignaledEvent(EventKind::Ordinary, Ctx, SignalQueue);
  sycl::event OtherPlain = syclex::make_event(OtherCtx);
  syclex::enqueue_signal_event(OtherQ, OtherPlain);
  OtherQ.wait();

  auto Gate = std::make_shared<HostTaskGate>();
  OpenAtScopeExit OpenGate{Gate};
  blockQueue(Q, Gate);
  // Held behind the host task, so it has no backend event yet; captures the
  // current binding of E.
  sycl::event Held = Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on(E);
    CGH.single_task<BindingTestKernel>([]() {});
  });

  const SignalSnapshot Before{E};
  const SignalSnapshot PlainBefore{Plain};
  expectError(
      sycl::errc::invalid, [&] { syclex::enqueue_wait_event(Q, E); },
      "a wait behind a held kernel");
  expectError(
      sycl::errc::invalid,
      [&] {
        syclex::enqueue_wait_events(Q, {Plain, E});
      },
      "a mixed wait behind a held kernel");
  expectError(
      sycl::errc::invalid,
      [&] {
        syclex::enqueue_wait_events(Idle, {Held, E});
      },
      "a wait with a dependency held in the scheduler");
  expectError(
      sycl::errc::invalid,
      [&] {
        syclex::enqueue_wait_events(Idle, {E, Plain, OtherPlain});
      },
      "a wait with an event of another context");
  expectError(
      sycl::errc::invalid, [&] { syclex::enqueue_wait_event(OtherQ, E); },
      "a wait on a queue of another context");
  Before.expectIntact(E, "after the rejected waits");
  PlainBefore.expectIntact(Plain, "the ordinary event");
  EXPECT_EQ(handleOf(Held), nullptr);
  EXPECT_FALSE(isComplete(Held));

  // Without the IPC event, the same waits are held or bridged as usual.
  EXPECT_NO_THROW(syclex::enqueue_wait_events(Q, {Plain}));
  EXPECT_NO_THROW(syclex::enqueue_wait_events(Idle, {Held, Plain}));
  EXPECT_NO_THROW(syclex::enqueue_wait_event(OtherQ, Plain));

  Gate->open();
  Q.wait();
  Idle.wait();
  OtherQ.wait();
  EXPECT_TRUE(isComplete(Held));
  EXPECT_EQ(handleOf(E), Shared);

  // Nothing is held any more. A wait on a queue of another context would
  // always need the scheduler, so it is not retried.
  size_t BarriersBefore = 0;
  {
    std::lock_guard<std::mutex> Lock(BackendMutex);
    BarriersBefore = BarrierWaitLists.size();
  }
  EXPECT_NO_THROW(syclex::enqueue_wait_event(Q, E));
  EXPECT_NO_THROW(syclex::enqueue_wait_events(Idle, {Plain, E, Held}));
  Q.wait();
  Idle.wait();
  {
    std::lock_guard<std::mutex> Lock(BackendMutex);
    size_t WaitsForShared = 0;
    for (size_t I = BarriersBefore; I < BarrierWaitLists.size(); ++I) {
      WaitsForShared += std::count(BarrierWaitLists[I].begin(),
                                   BarrierWaitLists[I].end(), Shared);
    }
    EXPECT_EQ(WaitsForShared, 2u);
  }
  EXPECT_EQ(handleOf(E), Shared);
  EXPECT_EQ(releases(Shared), Before.Releases);
}

//===----------------------------------------------------------------------===//
// tests-10-02 U37: an immediate re-signal shares the handle across bindings.
//===----------------------------------------------------------------------===//

class ReusableEventsIpcResignalTest
    : public ReusableEventsIpcTest,
      public ::testing::WithParamInterface<std::tuple<EventKind, bool>> {};

INSTANTIATE_TEST_SUITE_P(
    ReusableEventsIpc, ReusableEventsIpcResignalTest,
    ::testing::Combine(::testing::Values(EventKind::Exported,
                                         EventKind::Imported),
                       ::testing::Bool()),
    [](const ::testing::TestParamInfo<std::tuple<EventKind, bool>> &Info) {
      return kindName(std::get<0>(Info.param)) +
             (std::get<1>(Info.param) ? "OlderFirst" : "NewerFirst");
    });

// tests-10-02 U37. While something still refers to the current binding of an
// IPC event (here the test, standing for a consumer), an immediate re-signal
// moves the event on to a new binding, which retains the shared backend event
// instead of creating one. Whichever binding is dropped first, the other one
// keeps the handle, and the event keeps working; the handle is released once
// both are gone, and the exported handle data is returned when the event is
// destroyed (ipc::event::put does not return it).
TEST_P(ReusableEventsIpcResignalTest,
       ImmediateResignalSharesHandleAcrossBindings) {
  const EventKind Kind = std::get<0>(GetParam());
  const bool OlderFirst = std::get<1>(GetParam());
  sycl::queue SignalQueue{Ctx, Dev};
  sycl::queue Q = inOrderQueue();
  ur_event_handle_t Shared = nullptr;
  std::shared_ptr<sycl::detail::event_binding> Older;
  std::weak_ptr<sycl::detail::event_binding> Newer;
  int Releases = 0;
  {
    sycl::event E = makeSignaledEvent(Kind, Ctx, SignalQueue);
    Shared = handleOf(E);
    ASSERT_NE(Shared, nullptr);
    std::optional<syclex::ipc::handle> Exported;
    if (Kind == EventKind::Exported) {
      Exported.emplace(syclex::ipc::event::get(E));
      EXPECT_EQ(IpcGets, 1);
    }

    Older = bindingOf(E);
    size_t Created = 0;
    int Retains = 0;
    {
      std::lock_guard<std::mutex> Lock(BackendMutex);
      Created = CreatedEvents.size();
      Retains = RetainCounts[Shared];
      Releases = ReleaseCounts[Shared];
    }

    syclex::enqueue_signal_event(SignalQueue, E);
    SignalQueue.wait();
    std::shared_ptr<sycl::detail::event_binding> Current = bindingOf(E);
    EXPECT_NE(Current, Older);
    EXPECT_EQ(Current->getHandle(), Shared);
    EXPECT_EQ(Older->getHandle(), Shared);
    EXPECT_EQ(handleOf(E), Shared);
    {
      std::lock_guard<std::mutex> Lock(BackendMutex);
      EXPECT_EQ(CreatedEvents.size(), Created);
      // One reference for each binding.
      EXPECT_EQ(RetainCounts[Shared], Retains + 1);
      EXPECT_EQ(ReleaseCounts[Shared], Releases);
      ASSERT_FALSE(BarrierOutEvents.empty());
      EXPECT_EQ(BarrierOutEvents.back(), Shared);
    }
    Newer = Current;
    Current.reset();

    if (Exported) {
      syclex::ipc::event::put(*Exported, Ctx);
      EXPECT_EQ(IpcPuts, 0);
    }

    if (OlderFirst) {
      Older.reset();
      EXPECT_EQ(releases(Shared), Releases + 1);
      EXPECT_FALSE(ownershipBalanced(Shared));
      // The event goes on with the shared backend event.
      EXPECT_FALSE(Newer.expired());
      EXPECT_EQ(handleOf(E), Shared);
      EXPECT_NO_THROW(syclex::enqueue_signal_event(SignalQueue, E));
      EXPECT_NO_THROW(syclex::enqueue_wait_event(Q, E));
      SignalQueue.wait();
      Q.wait();
      EXPECT_EQ(handleOf(E), Shared);
      EXPECT_EQ(releases(Shared), Releases + 1);
    }
  }
  if (Kind == EventKind::Exported) {
    EXPECT_TRUE(eventually([] { return IpcPuts == 1; }));
  }

  if (!OlderFirst) {
    // The event, and the binding it had, are gone; the older binding still
    // owns the shared backend event.
    EXPECT_TRUE(eventually([&] { return Newer.expired(); }));
    ASSERT_NE(Older, nullptr);
    EXPECT_EQ(Older->getHandle(), Shared);
    EXPECT_EQ(releases(Shared), Releases + 1);
    EXPECT_FALSE(ownershipBalanced(Shared));
    Older.reset();
  }

  EXPECT_TRUE(eventually(
      [&] { return Newer.expired() && ownershipBalancedLocked(Shared); }));
  EXPECT_EQ(IpcGets, Kind == EventKind::Exported ? 1 : 0);
  EXPECT_EQ(IpcPuts, Kind == EventKind::Exported ? 1 : 0);
}

//===----------------------------------------------------------------------===//
// tests-10-02 U38: invalid operations leave the previous signal alone.
//===----------------------------------------------------------------------===//

// A valid signal of an ordinary event, and a kernel held behind a host task
// which captured it. finish lets the kernel through, checks it waited for that
// signal, and signals and waits for the event again.
struct PriorSignal {
  PriorSignal(const sycl::context &Ctx, const sycl::device &Dev)
      : SignalQueue{Ctx, Dev}, Q{Ctx, Dev, sycl::property::queue::in_order{}},
        E{makeSignaledEvent(EventKind::Ordinary, Ctx, SignalQueue)},
        Handle{handleOf(E)} {
    blockQueue(Q, Gate);
    Consumer = Q.submit([&](sycl::handler &CGH) {
      CGH.depends_on(E);
      CGH.single_task<BindingTestKernel>([]() {});
    });
    Snapshot.emplace(E);
  }

  void expectIntact(const char *What) const {
    Snapshot->expectIntact(E, What);
    EXPECT_TRUE(isComplete(E)) << What;
    EXPECT_EQ(handleOf(Consumer), nullptr) << What;
  }

  void finish() {
    Gate->open();
    Q.wait();
    {
      std::lock_guard<std::mutex> Lock(BackendMutex);
      ASSERT_EQ(KernelLaunchWaitLists.size(), Snapshot->Launches + 1);
      EXPECT_EQ(KernelLaunchWaitLists[Snapshot->Launches],
                std::vector<ur_event_handle_t>{Handle});
    }
    EXPECT_NO_THROW(syclex::enqueue_signal_event(SignalQueue, E));
    SignalQueue.wait();
    EXPECT_NE(handleOf(E), nullptr);
    EXPECT_TRUE(isComplete(E));
    EXPECT_NO_THROW(syclex::enqueue_wait_event(Q, E));
    Q.wait();
  }

  sycl::queue SignalQueue;
  sycl::queue Q;
  std::shared_ptr<HostTaskGate> Gate = std::make_shared<HostTaskGate>();
  OpenAtScopeExit OpenGate{Gate};
  sycl::event E;
  ur_event_handle_t Handle;
  sycl::event Consumer;
  std::optional<SignalSnapshot> Snapshot;
};

// tests-10-02 U38. A signal on a queue of another context.
TEST_F(ReusableEventsIpcTest, WrongContextSignalKeepsPriorSignal) {
  PriorSignal P{Ctx, Dev};
  sycl::context OtherCtx{Dev};
  sycl::queue OtherQ = inOrderQueue(OtherCtx);
  expectError(
      sycl::errc::invalid, [&] { syclex::enqueue_signal_event(OtherQ, P.E); },
      "a signal in another context");
  P.expectIntact("after the signal in another context");
  P.finish();
}

// tests-10-02 U38. A signal of an interop event. The interop event keeps its
// backend event and can still be waited for, and it releases the backend
// event when it is destroyed. The queue which waited for it keeps it until the
// queue is destroyed, so the release is checked after that.
TEST_F(ReusableEventsIpcTest, InteropSignalKeepsPriorSignal) {
  ur_event_handle_t InteropHandle = nullptr;
  {
    std::lock_guard<std::mutex> Lock(BackendMutex);
    InteropHandle = newFakeEvent();
  }
  {
    PriorSignal P{Ctx, Dev};
    {
      sycl::event Interop = sycl::detail::createSyclObjFromImpl<sycl::event>(
          sycl::detail::event_impl::create_from_handle(InteropHandle, Ctx));
      ASSERT_TRUE(sycl::detail::getSyclObjImpl(Interop)->isInterop());
      expectError(
          sycl::errc::runtime,
          [&] { syclex::enqueue_signal_event(P.SignalQueue, Interop); },
          "a signal of an interop event");
      EXPECT_EQ(handleOf(Interop), InteropHandle);
      P.expectIntact("after the signal of an interop event");
      EXPECT_NO_THROW(syclex::enqueue_wait_event(P.SignalQueue, Interop));
      P.SignalQueue.wait();
    }
    P.finish();
  }
  EXPECT_TRUE(
      eventually([&] { return ownershipBalancedLocked(InteropHandle); }));
}

// tests-10-02 U38. A signal on a queue recording a graph. The queue signals
// the event once it stops recording.
TEST_F(ReusableEventsIpcTest, GraphRecordingSignalKeepsPriorSignal) {
  PriorSignal P{Ctx, Dev};
  sycl::queue RecordingQ{Ctx, Dev};
  syclex::command_graph<syclex::graph_state::modifiable> Graph{Ctx, Dev};
  Graph.begin_recording(RecordingQ);
  expectError(
      sycl::errc::runtime,
      [&] { syclex::enqueue_signal_event(RecordingQ, P.E); },
      "a signal on a recording queue");
  Graph.end_recording(RecordingQ);
  P.expectIntact("after the signal on a recording queue");
  P.finish();

  EXPECT_NO_THROW(syclex::enqueue_signal_event(RecordingQ, P.E));
  RecordingQ.wait();
  EXPECT_TRUE(isComplete(P.E));
}

// tests-10-02 U38. A command depending on the event and on a discarded event.
// The runtime no longer returns discarded events from the public API, so the
// event of a completed kernel is marked discarded the way the scheduler marks
// one nobody asked for. The command is not submitted, and the dependency on
// the event it registered before the discarded one goes away with it.
TEST_F(ReusableEventsIpcTest, DiscardedDependencyKeepsPriorSignal) {
  PriorSignal P{Ctx, Dev};
  sycl::queue Idle{Ctx, Dev};
  sycl::event Discarded = Idle.submit(
      [&](sycl::handler &CGH) { CGH.single_task<BindingTestKernel>([]() {}); });
  Idle.wait();
  sycl::detail::getSyclObjImpl(Discarded)->setStateDiscarded();
  {
    std::lock_guard<std::mutex> Lock(BackendMutex);
    P.Snapshot->Launches = KernelLaunchWaitLists.size();
  }

  expectError(
      sycl::errc::invalid,
      [&] {
        P.Q.submit([&](sycl::handler &CGH) {
          CGH.depends_on(P.E);
          CGH.depends_on(Discarded);
          CGH.single_task<BindingTestKernel>([]() {});
        });
      },
      "a dependency on a discarded event");
  P.expectIntact("after the dependency on a discarded event");
  P.finish();
}

// tests-10-02 U38. enable_profiling together with enable_ipc. Each of them
// alone is fine.
TEST_F(ReusableEventsIpcTest, ProfilingWithIpcKeepsPriorSignal) {
  PriorSignal P{Ctx, Dev};
  expectError(
      sycl::errc::invalid,
      [&] {
        (void)syclex::make_event(
            Ctx, syclex::properties{syclex::enable_ipc{true},
                                    syclex::enable_profiling{true}});
      },
      "enable_profiling with enable_ipc");
  P.expectIntact("after enable_profiling with enable_ipc");
  EXPECT_NO_THROW((void)syclex::make_event(
      Ctx, syclex::properties{syclex::enable_profiling{true}}));
  EXPECT_NO_THROW((void)syclex::make_event(
      Ctx, syclex::properties{syclex::enable_ipc{true}}));
  P.finish();
}

// tests-10-02 U38. enable_profiling and enable_ipc on a context without
// per-event profiling and IPC event support. A plain event is still made and
// signaled.
TEST_F(ReusableEventsIpcNoSupportTest, UnsupportedPropertiesKeepPriorSignal) {
  PriorSignal P{Ctx, Dev};
  expectError(
      sycl::errc::feature_not_supported,
      [&] {
        (void)syclex::make_event(
            Ctx, syclex::properties{syclex::enable_profiling{true}});
      },
      "enable_profiling without per-event profiling support");
  expectError(
      sycl::errc::feature_not_supported,
      [&] {
        (void)syclex::make_event(Ctx,
                                 syclex::properties{syclex::enable_ipc{true}});
      },
      "enable_ipc without IPC event support");
  P.expectIntact("after the unsupported properties");
  P.finish();

  sycl::event Plain = syclex::make_event(Ctx);
  EXPECT_NO_THROW(syclex::enqueue_signal_event(P.SignalQueue, Plain));
  P.SignalQueue.wait();
  EXPECT_TRUE(isComplete(Plain));
}

// tests-10-02 U38. A signal of an exported IPC event on a profiling queue. The
// event keeps its binding and shared backend event, and is signaled on a
// queue without profiling afterwards. Imported events are not covered: the
// check reads isIPCEnabled, which is false for them.
TEST_F(ReusableEventsIpcTest, ExportedSignalOnProfilingQueueKeepsPriorSignal) {
  sycl::queue SignalQueue{Ctx, Dev};
  sycl::queue ProfilingQ{Ctx, Dev, sycl::property::queue::enable_profiling{}};
  sycl::event E = makeSignaledEvent(EventKind::Exported, Ctx, SignalQueue);
  const ur_event_handle_t Shared = handleOf(E);
  ASSERT_NE(Shared, nullptr);

  const SignalSnapshot Before{E};
  expectError(
      sycl::errc::invalid, [&] { syclex::enqueue_signal_event(ProfilingQ, E); },
      "a signal of an exported event on a profiling queue");
  Before.expectIntact(E, "after the signal on a profiling queue");

  EXPECT_NO_THROW(syclex::enqueue_signal_event(SignalQueue, E));
  SignalQueue.wait();
  EXPECT_EQ(handleOf(E), Shared);
  EXPECT_EQ(releases(Shared), Before.Releases);
  std::lock_guard<std::mutex> Lock(BackendMutex);
  EXPECT_EQ(CreatedEvents.size(), Before.Created);
  ASSERT_FALSE(BarrierOutEvents.empty());
  EXPECT_EQ(BarrierOutEvents.back(), Shared);
}

//===----------------------------------------------------------------------===//
// tests-10-02 U43: ordinary commands depending on an IPC event.
//===----------------------------------------------------------------------===//

enum class ConsumerKind { Kernel, Barrier, HostTask };

std::string consumerName(ConsumerKind Kind) {
  switch (Kind) {
  case ConsumerKind::Kernel:
    return "Kernel";
  case ConsumerKind::Barrier:
    return "Barrier";
  case ConsumerKind::HostTask:
    return "HostTask";
  }
  return "Unknown";
}

sycl::event submitConsumer(ConsumerKind Kind, sycl::queue &Q, sycl::event &E) {
  switch (Kind) {
  case ConsumerKind::Kernel:
    return Q.submit([&](sycl::handler &CGH) {
      CGH.depends_on(E);
      CGH.single_task<BindingTestKernel>([]() {});
    });
  case ConsumerKind::Barrier:
    return Q.ext_oneapi_submit_barrier({E});
  case ConsumerKind::HostTask:
    return Q.submit([&](sycl::handler &CGH) {
      CGH.depends_on(E);
      CGH.host_task([] {});
    });
  }
  return {};
}

class ReusableEventsIpcConsumerTest : public ReusableEventsIpcTest,
                                      public ::testing::WithParamInterface<
                                          std::tuple<EventKind, ConsumerKind>> {
};

INSTANTIATE_TEST_SUITE_P(
    ReusableEventsIpc, ReusableEventsIpcConsumerTest,
    ::testing::Combine(::testing::Values(EventKind::Ordinary,
                                         EventKind::Exported,
                                         EventKind::Imported),
                       ::testing::Values(ConsumerKind::Kernel,
                                         ConsumerKind::Barrier,
                                         ConsumerKind::HostTask)),
    [](const ::testing::TestParamInfo<std::tuple<EventKind, ConsumerKind>>
           &Info) {
      return kindName(std::get<0>(Info.param)) +
             consumerName(std::get<1>(Info.param));
    });

// clang-format off
// Investigation (tests-10-02 U43): pins behaviour at 8337da70; oracle not agreed.
// clang-format on
// tests-10-02 U43. A kernel, an event-returning barrier or a host task held
// behind a host task depends on an IPC event, which is then signaled again
// through the scheduler bypass, with a backend event still pending. The
// consumer is accepted - only the signal and the public waits are rejected -
// and the binding it captured shares the backend event with the new one, so
// once it reaches the backend it waits for the backend event of the new
// signal: the host task does not finish before that completes. An ordinary
// event is the control: its consumer waits for the signal it captured.
TEST_P(ReusableEventsIpcConsumerTest, HeldConsumerIsAccepted) {
  const EventKind Kind = std::get<0>(GetParam());
  const ConsumerKind Consumer = std::get<1>(GetParam());
  const bool Ipc = Kind != EventKind::Ordinary;
  ur_event_handle_t First = nullptr;
  ur_event_handle_t Second = nullptr;
  {
    sycl::queue SignalQueue{Ctx, Dev};
    sycl::queue Q = inOrderQueue();
    auto Gate = std::make_shared<HostTaskGate>();
    OpenAtScopeExit OpenGate{Gate};
    CompleteAllAtScopeExit CompleteAll;
    sycl::event E = makeSignaledEvent(Kind, Ctx, SignalQueue);
    First = handleOf(E);
    ASSERT_NE(First, nullptr);

    blockQueue(Q, Gate);
    sycl::event Consumed;
    EXPECT_NO_THROW(Consumed = submitConsumer(Consumer, Q, E));
    const SignalSnapshot Captured{E};

    setBarriersStayPending(true);
    EXPECT_NO_THROW(syclex::enqueue_signal_event(SignalQueue, E));
    setBarriersStayPending(false);
    Second = handleOf(E);
    ASSERT_NE(Second, nullptr);
    EXPECT_NE(bindingOf(E), Captured.Binding);
    EXPECT_EQ(Captured.Binding->getHandle(), First);
    EXPECT_TRUE(isPending(Second));
    {
      std::lock_guard<std::mutex> Lock(BackendMutex);
      if (Ipc) {
        // The new binding shares the backend event, with a reference of its
        // own.
        EXPECT_EQ(Second, First);
        EXPECT_EQ(CreatedEvents.size(), Captured.Created);
        EXPECT_EQ(RetainCounts[First], Captured.Retains + 1);
      } else {
        EXPECT_NE(Second, First);
        EXPECT_EQ(CreatedEvents.size(), Captured.Created + 1);
      }
      EXPECT_EQ(ReleaseCounts[First], Captured.Releases);
    }
    EXPECT_EQ(handleOf(Consumed), nullptr);

    size_t WaitsBefore = 0;
    {
      std::lock_guard<std::mutex> Lock(BackendMutex);
      WaitsBefore = WaitedEvents.size();
    }
    Gate->open();
    switch (Consumer) {
    case ConsumerKind::Kernel:
    case ConsumerKind::Barrier: {
      // The backend command waits for the captured binding's backend event:
      // for an IPC event that is the backend event of the new signal.
      Q.wait();
      std::lock_guard<std::mutex> Lock(BackendMutex);
      const std::vector<ur_event_handle_t> Expected{First};
      if (Consumer == ConsumerKind::Kernel) {
        ASSERT_EQ(KernelLaunchWaitLists.size(), Captured.Launches + 1);
        EXPECT_EQ(KernelLaunchWaitLists.back(), Expected);
      } else {
        size_t Matches = 0;
        for (size_t I = Captured.Barriers; I < BarrierWaitLists.size(); ++I) {
          if (BarrierWaitLists[I] == Expected)
            ++Matches;
        }
        EXPECT_EQ(Matches, 1u);
      }
      break;
    }
    case ConsumerKind::HostTask: {
      // The host task waits for the captured binding's backend event on the
      // host.
      EXPECT_TRUE(eventually([&] {
        return std::find(WaitedEvents.begin() + WaitsBefore, WaitedEvents.end(),
                         First) != WaitedEvents.end();
      }));
      if (Ipc) {
        // That is the pending backend event of the new signal.
        EXPECT_FALSE(isComplete(Consumed));
      } else {
        Consumed.wait();
        EXPECT_TRUE(isPending(Second));
      }
      break;
    }
    }
    complete(Second);
    Q.wait();
    SignalQueue.wait();
    EXPECT_TRUE(isComplete(Consumed));
    EXPECT_TRUE(isComplete(E));
    EXPECT_EQ(handleOf(E), Second);
  }
  EXPECT_TRUE(eventually([&] {
    return ownershipBalancedLocked(First) && ownershipBalancedLocked(Second);
  }));
}

// clang-format off
// Investigation (tests-10-02 U43): pins behaviour at 8337da70; oracle not agreed.
// clang-format on
// tests-10-02 U43. The immediate control: on a queue with nothing held, the
// same consumers of an IPC event, or of an ordinary one, wait for its backend
// event.
TEST_P(ReusableEventsIpcConsumerTest, ImmediateConsumerIsAccepted) {
  const EventKind Kind = std::get<0>(GetParam());
  const ConsumerKind Consumer = std::get<1>(GetParam());
  ur_event_handle_t Handle = nullptr;
  {
    sycl::queue SignalQueue{Ctx, Dev};
    sycl::queue Q = inOrderQueue();
    sycl::event E = makeSignaledEvent(Kind, Ctx, SignalQueue);
    Handle = handleOf(E);
    ASSERT_NE(Handle, nullptr);
    size_t WaitsBefore = 0;
    size_t BarriersBefore = 0;
    size_t LaunchesBefore = 0;
    {
      std::lock_guard<std::mutex> Lock(BackendMutex);
      WaitsBefore = WaitedEvents.size();
      BarriersBefore = BarrierWaitLists.size();
      LaunchesBefore = KernelLaunchWaitLists.size();
    }

    sycl::event Consumed;
    EXPECT_NO_THROW(Consumed = submitConsumer(Consumer, Q, E));
    Q.wait();
    EXPECT_TRUE(isComplete(Consumed));
    EXPECT_EQ(handleOf(E), Handle);

    std::lock_guard<std::mutex> Lock(BackendMutex);
    const std::vector<ur_event_handle_t> Expected{Handle};
    switch (Consumer) {
    case ConsumerKind::Kernel:
      ASSERT_EQ(KernelLaunchWaitLists.size(), LaunchesBefore + 1);
      EXPECT_EQ(KernelLaunchWaitLists.back(), Expected);
      break;
    case ConsumerKind::Barrier: {
      size_t Matches = 0;
      for (size_t I = BarriersBefore; I < BarrierWaitLists.size(); ++I) {
        if (BarrierWaitLists[I] == Expected)
          ++Matches;
      }
      EXPECT_EQ(Matches, 1u);
      break;
    }
    case ConsumerKind::HostTask:
      EXPECT_NE(std::find(WaitedEvents.begin() + WaitsBefore,
                          WaitedEvents.end(), Handle),
                WaitedEvents.end());
      break;
    }
  }
  EXPECT_TRUE(eventually([&] { return ownershipBalancedLocked(Handle); }));
}

} // anonymous namespace
