// Verifies the interaction of SYCL_LAUNCH_BLOCKING with command graphs:
// recording must not block (nothing executes and waiting on a recording queue
// is illegal), while executing a finalized graph must be synchronous, i.e. its
// results are visible to the host by the time the submission returns.
//
// REQUIRES: aspect-ext_oneapi_limited_graph, aspect-usm_shared_allocations
//
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: env SYCL_LAUNCH_BLOCKING=1 %{run} %t.out blocking
// RUN: env SYCL_LAUNCH_BLOCKING=2 %{run} %t.out blocking

#include <cassert>
#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <sycl/properties/queue_properties.hpp>
#include <sycl/usm.hpp>

namespace exp_ext = sycl::ext::oneapi::experimental;

constexpr size_t N = 1024;

// Iteration count of the spin loop below. Long enough that an asynchronous
// graph execution is still in flight when the submission returns.
constexpr int SpinCount = 20000;

// Set by the RUN lines that enable blocking mode. Without blocking a submission
// may legitimately have completed already, so the checks only apply when this
// is set.
static bool Blocking = false;

// The spin loop, also run on the host to get the expected result. The loop
// count is read from memory so that the device-side loop cannot be folded away.
static int spin(const int *Limit) {
  int Acc = 0;
  for (int I = 0; I < *Limit; ++I)
    Acc += I % 7;
  return Acc;
}

// Each graph node accumulates the spin result, so the value tells the host how
// many nodes have run.
static void spinAndAccumulate(int *Out, size_t Idx, const int *Limit) {
  Out[Idx] += spin(Limit);
}

static void check(sycl::queue &Q, const int *Out, const int *Limit,
                  int NodesRun, const char *What) {
  if (Blocking) {
    const int Expected = NodesRun * spin(Limit);
    if (Out[0] != Expected)
      std::cerr << "result of " << What << " is not visible yet" << std::endl;
    assert(Out[0] == Expected && "graph submission was not synchronous");
  }
  Q.wait();
}

// Queue recording mode: commands submitted between begin_recording and
// end_recording are captured, not executed, so they must not block.
static void runRecordReplay(sycl::queue &Q, int *Limit) {
  int *Out = sycl::malloc_shared<int>(N, Q);
  Q.fill(Out, 0, N).wait();

  exp_ext::command_graph Graph{Q.get_context(), Q.get_device()};

  Graph.begin_recording(Q);
  Q.parallel_for(sycl::range<1>{N},
                 [=](sycl::id<1> Idx) { spinAndAccumulate(Out, Idx, Limit); });
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for(sycl::range<1>{N}, [=](sycl::id<1> Idx) {
      spinAndAccumulate(Out, Idx, Limit);
    });
  });
  Graph.end_recording();

  auto ExecGraph = Graph.finalize();

  // Each execution must complete before the submission returns, and replaying
  // the same executable graph repeatedly must keep working.
  int NodesRun = 0;
  for (int Run = 0; Run < 3; ++Run) {
    Q.ext_oneapi_graph(ExecGraph);
    NodesRun += 2;
    check(Q, Out, Limit, NodesRun, "ext_oneapi_graph (recorded)");
  }

  // The same executable graph submitted through a command group instead.
  Q.submit([&](sycl::handler &CGH) { CGH.ext_oneapi_graph(ExecGraph); });
  NodesRun += 2;
  check(Q, Out, Limit, NodesRun, "submit + ext_oneapi_graph");

  sycl::free(Out, Q);
  std::cout << "record/replay: OK" << std::endl;
}

// Explicit graph building never touches a queue until the graph is executed.
static void runExplicit(sycl::queue &Q, int *Limit) {
  int *Out = sycl::malloc_shared<int>(N, Q);
  Q.fill(Out, 0, N).wait();

  exp_ext::command_graph Graph{Q.get_context(), Q.get_device()};

  auto NodeA = Graph.add([&](sycl::handler &CGH) {
    CGH.parallel_for(sycl::range<1>{N}, [=](sycl::id<1> Idx) {
      spinAndAccumulate(Out, Idx, Limit);
    });
  });
  Graph.add(
      [&](sycl::handler &CGH) {
        CGH.parallel_for(sycl::range<1>{N}, [=](sycl::id<1> Idx) {
          spinAndAccumulate(Out, Idx, Limit);
        });
      },
      exp_ext::property::node::depends_on{NodeA});

  auto ExecGraph = Graph.finalize();

  Q.ext_oneapi_graph(ExecGraph);
  check(Q, Out, Limit, 2, "ext_oneapi_graph (explicit)");

  sycl::free(Out, Q);
  std::cout << "explicit: OK" << std::endl;
}

// A graph containing a host task takes a different submission path, since host
// task dependencies cannot be expressed natively.
static void runWithHostTask(sycl::queue &Q, int *Limit) {
  int *Out = sycl::malloc_shared<int>(N, Q);
  Q.fill(Out, 0, N).wait();

  exp_ext::command_graph Graph{Q.get_context(), Q.get_device()};

  Graph.begin_recording(Q);
  Q.parallel_for(sycl::range<1>{N},
                 [=](sycl::id<1> Idx) { spinAndAccumulate(Out, Idx, Limit); });
  Q.submit([&](sycl::handler &CGH) {
    CGH.host_task([]() {
      volatile int Acc = 0;
      for (int I = 0; I < SpinCount * 100; ++I)
        Acc += I % 7;
    });
  });
  Graph.end_recording();

  auto ExecGraph = Graph.finalize();

  Q.ext_oneapi_graph(ExecGraph);
  check(Q, Out, Limit, 1, "ext_oneapi_graph (with host task)");

  sycl::free(Out, Q);
  std::cout << "host task in graph: OK" << std::endl;
}

int main(int argc, char *argv[]) {
  Blocking = argc > 1;

  sycl::queue Q{sycl::property::queue::in_order{}};

  int *Limit = sycl::malloc_shared<int>(1, Q);
  *Limit = SpinCount;

  runRecordReplay(Q, Limit);
  runExplicit(Q, Limit);
  runWithHostTask(Q, Limit);

  sycl::free(Limit, Q);
  return 0;
}
