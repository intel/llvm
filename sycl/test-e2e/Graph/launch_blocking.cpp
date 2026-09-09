// SYCL_LAUNCH_BLOCKING with command graphs: recording must not block, while
// executing a finalized graph is a regular submission. The unit tests check
// that directly; here we check that graphs still work in blocking mode. Each
// node accumulates into the same buffer, so no node may be dropped or run
// twice.
//
// REQUIRES: aspect-ext_oneapi_limited_graph, aspect-usm_shared_allocations
//
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: env SYCL_LAUNCH_BLOCKING=1 %{run} %t.out

#include <cassert>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <sycl/properties/queue_properties.hpp>
#include <sycl/usm.hpp>

namespace exp_ext = sycl::ext::oneapi::experimental;

constexpr size_t N = 1024;

static void check(sycl::queue &Q, const int *Out, int NodesRun) {
  Q.wait();
  assert(Out[0] == NodesRun && Out[N - 1] == NodesRun);
}

// Queue recording mode: commands submitted between begin_recording and
// end_recording are captured, not executed, so they must not block.
static void runRecordReplay(sycl::queue &Q) {
  int *Out = sycl::malloc_shared<int>(N, Q);
  Q.fill(Out, 0, N).wait();

  exp_ext::command_graph Graph{Q.get_context(), Q.get_device()};

  Graph.begin_recording(Q);
  Q.parallel_for(sycl::range<1>{N}, [=](sycl::id<1> Idx) { Out[Idx] += 1; });
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for(sycl::range<1>{N},
                     [=](sycl::id<1> Idx) { Out[Idx] += 1; });
  });
  Graph.end_recording();

  auto ExecGraph = Graph.finalize();

  // Replaying the same executable graph repeatedly must keep working.
  int NodesRun = 0;
  for (int Run = 0; Run < 3; ++Run) {
    Q.ext_oneapi_graph(ExecGraph);
    NodesRun += 2;
    check(Q, Out, NodesRun);
  }

  // The same executable graph submitted through a command group instead.
  Q.submit([&](sycl::handler &CGH) { CGH.ext_oneapi_graph(ExecGraph); });
  NodesRun += 2;
  check(Q, Out, NodesRun);

  sycl::free(Out, Q);
}

// Explicit graph building never touches a queue until the graph is executed.
static void runExplicit(sycl::queue &Q) {
  int *Out = sycl::malloc_shared<int>(N, Q);
  Q.fill(Out, 0, N).wait();

  exp_ext::command_graph Graph{Q.get_context(), Q.get_device()};

  auto NodeA = Graph.add([&](sycl::handler &CGH) {
    CGH.parallel_for(sycl::range<1>{N},
                     [=](sycl::id<1> Idx) { Out[Idx] += 1; });
  });
  Graph.add(
      [&](sycl::handler &CGH) {
        CGH.parallel_for(sycl::range<1>{N},
                         [=](sycl::id<1> Idx) { Out[Idx] += 1; });
      },
      exp_ext::property::node::depends_on{NodeA});

  auto ExecGraph = Graph.finalize();

  Q.ext_oneapi_graph(ExecGraph);
  check(Q, Out, 2);

  sycl::free(Out, Q);
}

// A graph containing a host task takes a different submission path, since host
// task dependencies cannot be expressed natively.
static void runWithHostTask(sycl::queue &Q) {
  int *Out = sycl::malloc_shared<int>(N, Q);
  Q.fill(Out, 0, N).wait();

  bool HostTaskDone = false;

  exp_ext::command_graph Graph{Q.get_context(), Q.get_device()};

  Graph.begin_recording(Q);
  Q.parallel_for(sycl::range<1>{N}, [=](sycl::id<1> Idx) { Out[Idx] += 1; });
  Q.submit([&](sycl::handler &CGH) {
    CGH.host_task([&HostTaskDone]() { HostTaskDone = true; });
  });
  Graph.end_recording();

  auto ExecGraph = Graph.finalize();

  Q.ext_oneapi_graph(ExecGraph);
  check(Q, Out, 1);
  assert(HostTaskDone && "host task in the graph did not run");

  sycl::free(Out, Q);
}

int main() {
  sycl::queue Q{sycl::property::queue::in_order{}};

  runRecordReplay(Q);
  runExplicit(Q);
  runWithHostTask(Q);

  return 0;
}
