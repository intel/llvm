// Tests that a restricted host task submitted through the
// handler path can be recorded into a SYCL Graph and
// participate in event-based dependencies.

#include "../../graph_common.hpp"

#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/properties/all_properties.hpp>

namespace syclex = sycl::ext::oneapi::experimental;

constexpr size_t N = 1024;

int main() {
  device Dev;
  context Ctx{Dev};

  queue Queue1{Ctx, Dev, {property::queue::in_order{}}};
  queue Queue2{Ctx, Dev, {property::queue::in_order{}}};
  queue Queue3{Ctx, Dev, {property::queue::in_order{}}};

#ifdef GRAPH_E2E_NATIVE_RECORDING
  exp_ext::command_graph Graph{
      Ctx, Dev, {exp_ext::property::graph::enable_native_recording{}}};
#else
  exp_ext::command_graph Graph{Ctx, Dev};
#endif

  uint32_t *Data = malloc_shared<uint32_t>(N, Queue1);
  std::fill(Data, Data + N, 0);

  QueueStateVerifier verifier(Queue1, Queue2, Queue3);
  verifier.verify(EXECUTING, EXECUTING, EXECUTING);

  Graph.begin_recording(Queue1);
  verifier.verify(RECORDING, EXECUTING, EXECUTING);

  syclex::submit_with_event(Queue1, [&](handler &CGH) {
    syclex::host_task(CGH, [=] {
      for (size_t i = 0; i < N; i++)
        Data[i] += 5;
    });
  });

  Queue1.submit([&](handler &CGH) {
    CGH.parallel_for(range<1>{N}, [=](id<1> idx) {
      Data[idx] += static_cast<uint32_t>(idx[0]) + 1;
    });
  });

  // Fork: host task signals ForkEvent, consumed by Q2 and Q3.
  auto ForkEvent = syclex::submit_with_event(Queue1, [&](handler &CGH) {
    syclex::host_task(CGH, [=] {
      for (size_t i = 0; i < N; i++)
        Data[i] += 100;
    });
  });

  // Q2 and Q3 run concurrently, so they operate on disjoint halves.
  // Q2 fork: kernel on the first half.
  auto Q2Event = syclex::submit_with_event(Queue2, [&](handler &CGH) {
    CGH.depends_on(ForkEvent);
    CGH.parallel_for(range<1>{N / 2}, [=](id<1> idx) { Data[idx] += 10; });
  });
  verifier.verify(RECORDING, RECORDING, EXECUTING);

  // Q3 fork: host task on the second half.
  auto Q3Event = syclex::submit_with_event(Queue3, [&](handler &CGH) {
    CGH.depends_on(ForkEvent);
    syclex::host_task(CGH, [=] {
      for (size_t i = N / 2; i < N; i++)
        Data[i] += 10;
    });
  });
  verifier.verify(RECORDING, RECORDING, RECORDING);

  // Join: host task waits on events from two distinct queues
  Queue1.submit([&](handler &CGH) {
    CGH.depends_on({Q2Event, Q3Event});
    syclex::host_task(CGH, [=] {
      for (size_t i = 0; i < N; i++)
        Data[i] += 1;
    });
  });

  Graph.end_recording();

  auto ExecutableGraph = Graph.finalize();

  // Each replay adds 5 + (i + 1) + 100 + 10 + 1 = (i + 117) to every element.
  Queue1.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecutableGraph); });
  Queue1.wait();

  for (size_t i = 0; i < N; i++) {
    uint32_t Expected = static_cast<uint32_t>(i) + 117;
    assert(check_value(i, Expected, Data[i], "Data"));
  }

  Queue1.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecutableGraph); });
  Queue1.wait();

  for (size_t i = 0; i < N; i++) {
    uint32_t Expected = 2 * (static_cast<uint32_t>(i) + 117);
    assert(check_value(i, Expected, Data[i], "Data"));
  }

  free(Data, Queue1);
  return 0;
}
