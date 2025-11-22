// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
//

// Tests adding a sub-graph to an out-of-order queue using the handler-less
// path with event dependencies

#include "../graph_common.hpp"

int main() {
  queue Queue;

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};
  exp_ext::command_graph SubGraph1{Queue.get_context(), Queue.get_device()};
  exp_ext::command_graph SubGraph2{Queue.get_context(), Queue.get_device()};

  const size_t N = 10;
  int *X = malloc_device<int>(N, Queue);
  int *Y = malloc_device<int>(N, Queue);

  SubGraph1.begin_recording(Queue);
  {
    auto Event = Queue.submit([&](handler &CGH) {
      CGH.parallel_for(N, [=](id<1> it) { X[it] *= 2; });
    });

    Queue.submit([&](handler &CGH) {
      CGH.depends_on(Event);
      CGH.parallel_for(N, [=](id<1> it) { X[it] += 1; });
    });
  }
  SubGraph1.end_recording(Queue);
  auto ExecSubGraph1 = SubGraph1.finalize();

  SubGraph2.begin_recording(Queue);
  {
    auto Event = Queue.submit([&](handler &CGH) {
      CGH.parallel_for(N, [=](id<1> it) { Y[it] += X[it]; });
    });

    Queue.submit([&](handler &CGH) {
      CGH.depends_on(Event);
      CGH.parallel_for(N, [=](id<1> it) { Y[it] *= 3; });
    });
  }
  SubGraph2.end_recording(Queue);
  auto ExecSubGraph2 = SubGraph2.finalize();

  Graph.begin_recording(Queue);
  auto Event1 = Queue.submit(
      [&](handler &CGH) { CGH.parallel_for(N, [=](id<1> it) { X[it] = 1; }); });
  auto Event2 = Queue.ext_oneapi_graph(ExecSubGraph1, Event1);
  auto Event3 = Queue.submit(
      [&](handler &CGH) { CGH.parallel_for(N, [=](id<1> it) { Y[it] = 1; }); });
  auto Event4 = Queue.ext_oneapi_graph(ExecSubGraph2, {Event2, Event3});

  Queue.submit([&](handler &CGH) {
    CGH.depends_on(Event4);
    CGH.parallel_for(range<1>{N}, [=](id<1> it) { X[it] += 3; });
  });
  Graph.end_recording();
  auto ExecGraph = Graph.finalize();

  Queue.ext_oneapi_graph(ExecGraph).wait();

  int OutputX, OutputY;
  Queue.memcpy(&OutputX, X, sizeof(int)).wait();
  Queue.memcpy(&OutputY, Y, sizeof(int)).wait();

  assert(OutputX == 6);
  assert(OutputY == 12);

  sycl::free(X, Queue);
  sycl::free(Y, Queue);

  return 0;
}
