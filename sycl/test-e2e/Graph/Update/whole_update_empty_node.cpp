// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
// Extra run to check for immediate-command-list in Level Zero
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Tests that whole graph update works when a graph contain an empty node.

#include "../graph_common.hpp"

// Creates a graph with an empty node separating initialization and computation
// kernel nodes
template <class T>
void CreateGraph(
    exp_ext::command_graph<exp_ext::graph_state::modifiable> &Graph,
    size_t Size, T *Input1, T *Input2, T *Output) {
  Graph.add([&](handler &CGH) {
    CGH.single_task([=]() {
      for (int i = 0; i < Size; i++) {
        Input1[i] += i;
      }
    });
  });

  Graph.add([&](handler &CGH) {
    CGH.single_task([=]() {
      for (int i = 0; i < Size; i++) {
        Input2[i] += i;
      }
    });
  });

  auto EmptyNodeA =
      Graph.add({exp_ext::property::node::depends_on_all_leaves()});

  Graph.add(
      [&](handler &CGH) {
        CGH.single_task([=]() {
          for (int i = 0; i < Size; i++) {
            Output[i] = Input1[i] * Input2[i];
          }
        });
      },
      {exp_ext::property::node::depends_on(EmptyNodeA)});
}

int main() {
  queue Queue{};

  using T = int;

  // USM allocations for GraphA
  T *InputA1 = malloc_device<T>(Size, Queue);
  T *InputA2 = malloc_device<T>(Size, Queue);
  T *OutputA = malloc_device<T>(Size, Queue);

  // Initialize USM allocations
  T Pattern1 = 0xA;
  T Pattern2 = 0x42;
  T PatternZero = 0;

  Queue.fill(InputA1, Pattern1, Size);
  Queue.fill(InputA2, Pattern2, Size);
  Queue.fill(OutputA, PatternZero, Size);
  Queue.wait();

  // Construct GraphA
  exp_ext::command_graph GraphA{Queue};
  CreateGraph(GraphA, Size, InputA1, InputA2, OutputA);

  // Finalize, run, and validate GraphA
  auto GraphExecA = GraphA.finalize(exp_ext::property::graph::updatable{});
  Queue.ext_oneapi_graph(GraphExecA).wait();

  std::vector<T> HostOutput(Size);
  Queue.copy(OutputA, HostOutput.data(), Size).wait();

  for (int i = 0; i < Size; i++) {
    T Ref = (Pattern1 + i) * (Pattern2 + i);
    assert(check_value(i, Ref, HostOutput[i], "OutputA"));
  }

  // Create GraphB which will be used to update GraphA
  exp_ext::command_graph GraphB{Queue};

  // USM allocations for GraphB
  T *InputB1 = malloc_device<T>(Size, Queue);
  T *InputB2 = malloc_device<T>(Size, Queue);
  T *OutputB = malloc_device<T>(Size, Queue);

  // Initialize GraphB
  Pattern1 = -42;
  Pattern2 = 0xF;

  Queue.fill(InputB1, Pattern1, Size);
  Queue.fill(InputB2, Pattern2, Size);
  Queue.fill(OutputB, PatternZero, Size);
  Queue.wait();

  // Construct GraphB
  CreateGraph(GraphB, Size, InputB1, InputB2, OutputB);

  // Update executable GraphA with GraphB, run, and validate
  GraphExecA.update(GraphB);
  Queue.ext_oneapi_graph(GraphExecA).wait();

  Queue.copy(OutputB, HostOutput.data(), Size).wait();

  for (int i = 0; i < Size; i++) {
    T Ref = (Pattern1 + i) * (Pattern2 + i);
    assert(check_value(i, Ref, HostOutput[i], "OutputB"));
  }

  free(InputA1, Queue);
  free(InputA2, Queue);
  free(OutputA, Queue);

  free(InputB1, Queue);
  free(InputB2, Queue);
  free(OutputB, Queue);
  return 0;
}
