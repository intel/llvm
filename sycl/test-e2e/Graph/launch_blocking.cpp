// Verifies the interaction of SYCL_LAUNCH_BLOCKING with command graphs:
// recording must not block (nothing executes and waiting on a recording queue
// is illegal), while executing a finalized graph must be synchronous.
//
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: env SYCL_LAUNCH_BLOCKING=1 %{run} %t.out
// RUN: env SYCL_LAUNCH_BLOCKING=2 %{run} %t.out

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <sycl/properties/queue_properties.hpp>
#include <sycl/usm.hpp>
#include <vector>

namespace exp_ext = sycl::ext::oneapi::experimental;

// Sized so that an asynchronous graph execution is still in flight right after
// the submission returns.
constexpr int Iterations = 20000000;
constexpr size_t N = 1024;

static const bool Blocking = []() {
  const char *Val = std::getenv("SYCL_LAUNCH_BLOCKING");
  return Val && std::atoi(Val) != 0;
}();

static void checkIdle(sycl::queue &Q, const char *What) {
  const bool Idle = Q.ext_oneapi_empty();
  if (Blocking && !Idle)
    std::cerr << "queue not idle after " << What << std::endl;
  assert((!Blocking || Idle) && "graph submission was not synchronous");
  Q.wait();
}

// Spins long enough to be observable and bumps the counter by exactly one.
static void spin(int *Out, size_t Idx) {
  int Acc = 0;
  for (int I = 0; I < Iterations / static_cast<int>(N); ++I)
    Acc += I % 7;
  if (Acc >= 0)
    Out[Idx] += 1;
}

// Queue recording mode: commands submitted between begin_recording and
// end_recording are captured, not executed, so they must not block.
static void runRecordReplay(sycl::queue &Q) {
  int *Out = sycl::malloc_device<int>(N, Q);
  Q.fill(Out, 0, N).wait();

  exp_ext::command_graph Graph{Q.get_context(), Q.get_device()};

  Graph.begin_recording(Q);
  Q.parallel_for(sycl::range<1>{N}, [=](sycl::id<1> Idx) { spin(Out, Idx); });
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for(sycl::range<1>{N},
                     [=](sycl::id<1> Idx) { spin(Out, Idx); });
  });
  Graph.end_recording();

  auto ExecGraph = Graph.finalize();

  // Each execution must complete before the submission returns, and replaying
  // the same executable graph repeatedly must keep working.
  for (int Run = 0; Run < 3; ++Run) {
    Q.ext_oneapi_graph(ExecGraph);
    checkIdle(Q, "ext_oneapi_graph (recorded)");
  }

  // The same executable graph submitted through a command group instead.
  Q.submit([&](sycl::handler &CGH) { CGH.ext_oneapi_graph(ExecGraph); });
  checkIdle(Q, "submit + ext_oneapi_graph");

  std::vector<int> Host(N, -1);
  Q.memcpy(Host.data(), Out, N * sizeof(int)).wait();
  for (int V : Host)
    assert(V == 8 && "recorded graph produced a wrong result");

  sycl::free(Out, Q);
  std::cout << "record/replay: OK" << std::endl;
}

// Explicit graph building never touches a queue until the graph is executed.
static void runExplicit(sycl::queue &Q) {
  int *Out = sycl::malloc_device<int>(N, Q);
  Q.fill(Out, 0, N).wait();

  exp_ext::command_graph Graph{Q.get_context(), Q.get_device()};

  auto NodeA = Graph.add([&](sycl::handler &CGH) {
    CGH.parallel_for(sycl::range<1>{N},
                     [=](sycl::id<1> Idx) { spin(Out, Idx); });
  });
  Graph.add(
      [&](sycl::handler &CGH) {
        CGH.parallel_for(sycl::range<1>{N},
                         [=](sycl::id<1> Idx) { spin(Out, Idx); });
      },
      exp_ext::property::node::depends_on{NodeA});

  auto ExecGraph = Graph.finalize();

  Q.ext_oneapi_graph(ExecGraph);
  checkIdle(Q, "ext_oneapi_graph (explicit)");

  std::vector<int> Host(N, -1);
  Q.memcpy(Host.data(), Out, N * sizeof(int)).wait();
  for (int V : Host)
    assert(V == 2 && "explicit graph produced a wrong result");

  sycl::free(Out, Q);
  std::cout << "explicit: OK" << std::endl;
}

// A graph containing a host task takes a different submission path, since host
// task dependencies cannot be expressed natively.
static void runWithHostTask(sycl::queue &Q) {
  int *Out = sycl::malloc_device<int>(N, Q);
  Q.fill(Out, 0, N).wait();

  exp_ext::command_graph Graph{Q.get_context(), Q.get_device()};

  Graph.begin_recording(Q);
  Q.parallel_for(sycl::range<1>{N}, [=](sycl::id<1> Idx) { spin(Out, Idx); });
  Q.submit([&](sycl::handler &CGH) {
    CGH.host_task([]() {
      volatile int Acc = 0;
      for (int I = 0; I < Iterations; ++I)
        Acc += I % 7;
    });
  });
  Graph.end_recording();

  auto ExecGraph = Graph.finalize();

  Q.ext_oneapi_graph(ExecGraph);
  checkIdle(Q, "ext_oneapi_graph (with host task)");

  std::vector<int> Host(N, -1);
  Q.memcpy(Host.data(), Out, N * sizeof(int)).wait();
  for (int V : Host)
    assert(V == 1 && "graph with host task produced a wrong result");

  sycl::free(Out, Q);
  std::cout << "host task in graph: OK" << std::endl;
}

int main() {
  sycl::queue Q{sycl::property::queue::in_order{}};
  runRecordReplay(Q);
  runExplicit(Q);
  runWithHostTask(Q);
  return 0;
}
