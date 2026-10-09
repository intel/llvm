// REQUIRES: level_zero_v2_adapter

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Investigation (tests-10-02 E19): pins behaviour at 8337da70; oracle not
// agreed.
//
// tests-10-02 E19, ordinary (not native) recording variants: the reusable
// event APIs on a queue recording an ordinary graph, and with events recorded
// into such a graph. The expected results below are the behaviour at the
// head, not an agreed oracle; the recording policy is pending U33.
//
// Pinned at 8337da70, for an in-order and an out-of-order recording queue:
//   * enqueue_signal_event(RecordingQueue, E) throws errc::runtime (the
//     recording check in enqueue_signal_event), for a fresh event, a
//     completed event and an exported IPC event alike: the recording check
//     precedes the IPC checks.
//   * enqueue_wait_event(s)(RecordingQueue, E) for an ordinary event from
//     outside the graph throws errc::invalid: graph_impl::getCGEdges finds no
//     node for the event.
//   * Waiting on the recording queue for an exported or imported IPC event, or
//     a vector which contains one, throws errc::invalid from the IPC check in
//     submit_barrier_direct_impl, before the graph branch.
//   * An event returned by a recorded submission is a host event, so waiting
//     for it (on the recording queue or elsewhere) and signaling it throw
//     errc::invalid from CheckEventForWait/CheckEventForSignal.
// After every rejected call the graph's node count and the event's status are
// unchanged; after end_recording the graph executes correctly and the events
// are usable through an immediate signal.

#include "../graph_common.hpp"

#include <sycl/ext/oneapi/experimental/ipc_event.hpp>
#include <sycl/ext/oneapi/experimental/reusable_events.hpp>
#include <sycl/properties/all_properties.hpp>

#include <optional>
#include <string>

namespace ipc = sycl::ext::oneapi::experimental::ipc;

namespace {

constexpr size_t N = 256;

int Failures = 0;

#define CHECK(Cond)                                                            \
  do {                                                                         \
    if (!(Cond)) {                                                             \
      std::cerr << __FILE__ << ":" << __LINE__ << ": check failed: " #Cond     \
                << std::endl;                                                  \
      ++Failures;                                                              \
    }                                                                          \
  } while (0)

info::event_command_status status(const event &E) {
  return E.get_info<info::event::command_execution_status>();
}

// Runs a call that must be rejected with Code and checks that neither the
// graph nor the event changed.
template <typename Func>
void expectRejected(Func &&Call, const std::string &What, errc Code,
                    exp_ext::command_graph<exp_ext::graph_state::modifiable> &G,
                    const event &E) {
  const size_t NodesBefore = G.get_nodes().size();
  const auto StatusBefore = status(E);
  if (!expectException(Call, What.c_str(), Code))
    ++Failures;
  if (G.get_nodes().size() != NodesBefore) {
    std::cerr << What << ": the graph's node count changed" << std::endl;
    ++Failures;
  }
  if (status(E) != StatusBefore) {
    std::cerr << What << ": the event's status changed" << std::endl;
    ++Failures;
  }
}

void runTest(bool InOrder) {
  std::cout << "Recording queue: " << (InOrder ? "in-order" : "out-of-order")
            << std::endl;

  device Dev;
  context Ctx{Dev};
  property_list Props;
  if (InOrder)
    Props = property_list{property::queue::in_order{}};
  queue RecQ{Ctx, Dev, Props};
  queue OtherQ{Ctx, Dev, {property::queue::in_order{}}};
  const bool HasIPC = Dev.has(aspect::ext_oneapi_ipc_event);

  int *Data = malloc_device<int>(N, Dev, Ctx);

  // Events from outside the graph.
  event Fresh = exp_ext::make_event(Ctx);
  event Completed = exp_ext::make_event(Ctx);
  exp_ext::enqueue_signal_event(OtherQ, Completed);
  Completed.wait();
  event KernelEvent = OtherQ.single_task([=]() { Data[0] = -1; });
  KernelEvent.wait();

  event Exported;
  event Imported;
  std::optional<ipc::handle> Handle;
  if (HasIPC) {
    Exported = exp_ext::make_event(
        Ctx, exp_ext::properties{exp_ext::enable_ipc{true}});
    exp_ext::enqueue_signal_event(OtherQ, Exported);
    Exported.wait();
    Handle.emplace(ipc::event::get(Exported));
    ipc::handle_data_t Bytes = Handle->data();
    Imported = ipc::event::open(Bytes, Ctx);
  } else {
    std::cout << "  no aspect::ext_oneapi_ipc_event, IPC cases skipped"
              << std::endl;
  }

  exp_ext::command_graph G{Ctx, Dev};
  G.begin_recording(RecQ);

  event Recorded = RecQ.submit([&](handler &CGH) {
    CGH.parallel_for(range<1>{N},
                     [=](id<1> Idx) { Data[Idx] = static_cast<int>(Idx[0]); });
  });

  // Signals on the recording queue.
  for (event *E : {&Fresh, &Completed}) {
    expectRejected([&]() { exp_ext::enqueue_signal_event(RecQ, *E); },
                   "signal on the recording queue", errc::runtime, G, *E);
  }
  if (HasIPC)
    expectRejected([&]() { exp_ext::enqueue_signal_event(RecQ, Exported); },
                   "signal of an exported IPC event on the recording queue",
                   errc::runtime, G, Exported);

  // Waits on the recording queue for events from outside the graph.
  for (event *E : {&Fresh, &Completed, &KernelEvent}) {
    expectRejected([&]() { exp_ext::enqueue_wait_event(RecQ, *E); },
                   "wait for an external event", errc::invalid, G, *E);
    expectRejected([&]() { exp_ext::enqueue_wait_events(RecQ, {*E}); },
                   "wait for an external event (vector)", errc::invalid, G, *E);
  }
  if (HasIPC) {
    for (event *E : {&Exported, &Imported}) {
      expectRejected([&]() { exp_ext::enqueue_wait_event(RecQ, *E); },
                     "wait for an IPC event", errc::invalid, G, *E);
      expectRejected(
          [&]() {
            exp_ext::enqueue_wait_events(RecQ, {Completed, *E});
          },
          "wait for a mixed vector with an IPC event", errc::invalid, G, *E);
    }
  }

  // Events returned by recorded submissions are host events.
  expectRejected([&]() { exp_ext::enqueue_wait_event(RecQ, Recorded); },
                 "wait for a recorded event on the recording queue",
                 errc::invalid, G, Completed);
  expectRejected(
      [&]() {
        exp_ext::enqueue_wait_events(RecQ, {Recorded, Completed});
      },
      "wait for a mixed vector with a recorded event", errc::invalid, G,
      Completed);
  expectRejected([&]() { exp_ext::enqueue_wait_event(OtherQ, Recorded); },
                 "wait for a recorded event outside the graph", errc::invalid,
                 G, Completed);
  expectRejected([&]() { exp_ext::enqueue_signal_event(OtherQ, Recorded); },
                 "signal of a recorded event", errc::invalid, G, Completed);

  // The graph is still usable for recording.
  RecQ.submit([&](handler &CGH) {
    if (!InOrder)
      CGH.depends_on(Recorded);
    CGH.parallel_for(range<1>{N}, [=](id<1> Idx) { Data[Idx] += 5; });
  });
  G.end_recording();
  CHECK(G.get_nodes().size() == 2);

  auto Exec = G.finalize();
  RecQ.ext_oneapi_graph(Exec).wait();
  std::vector<int> Host(N);
  RecQ.memcpy(Host.data(), Data, N * sizeof(int)).wait();
  for (size_t I = 0; I < N; ++I) {
    if (!check_value(I, static_cast<int>(I) + 5, Host[I], "Data")) {
      ++Failures;
      break;
    }
  }

  // The events remain usable through an immediate signal.
  for (event *E : {&Fresh, &Completed}) {
    exp_ext::enqueue_signal_event(RecQ, *E);
    E->wait();
    CHECK(status(*E) == info::event_command_status::complete);
  }
  if (HasIPC) {
    exp_ext::enqueue_signal_event(OtherQ, Exported);
    Exported.wait();
    Imported.wait();
    CHECK(status(Imported) == info::event_command_status::complete);
    ipc::event::put(*Handle, Ctx);
  }

  RecQ.wait_and_throw();
  OtherQ.wait_and_throw();
  free(Data, Ctx);
}

} // namespace

int main() {
  try {
    runTest(/*InOrder=*/true);
    runTest(/*InOrder=*/false);
  } catch (const sycl::exception &Ex) {
    std::cerr << "Unexpected exception: " << Ex.what() << std::endl;
    return 1;
  }
  if (Failures) {
    std::cerr << Failures << " check(s) failed" << std::endl;
    return 1;
  }
  return 0;
}
