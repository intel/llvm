// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{ %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
//
// Intended - USM fill command not supported for OpenCL
// UNSUPPORTED: opencl
//
// Tests adding a USM memset queue shortcut operation as a graph node.

#include "../graph_common.hpp"

int main() {

  queue Queue;

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  const size_t N = 10;
  unsigned char *Arr = malloc_device<unsigned char>(N, Queue);

  int Value = 77;
  Graph.begin_recording(Queue);
  auto Init = Queue.memset(Arr, Value, N);
  Queue.submit([&](handler &CGH) {
    CGH.depends_on(Init);
    CGH.single_task<class double_dest>([=]() {
      for (int i = 0; i < Size; i++)
        Arr[i] = 2 * Arr[i];
    });
  });

  Graph.end_recording();

  auto ExecGraph = Graph.finalize();

  Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecGraph); }).wait();

  std::vector<unsigned char> Output(N);
  Queue.memcpy(Output.data(), Arr, N).wait();
  for (int i = 0; i < N; i++)
    assert(Output[i] == (Value * 2));

  sycl::free(Arr, Queue);

  return 0;
}
