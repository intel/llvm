// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
//

// Tests updating a graph node from using a sycl::range to a sycl::nd_range

#include "../graph_common.hpp"

int main() {
  queue Queue{};

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  int *PtrA = malloc_device<int>(Size, Queue);

  std::vector<int> HostDataA(Size);

  Queue.memset(PtrA, 0, Size * sizeof(int)).wait();

  range<1> Range{Size};

  auto KernelNode = Graph.add([&](handler &cgh) {
    cgh.parallel_for(Range, [=](item<1> Item) {
      size_t GlobalID = Item.get_id();
      PtrA[GlobalID] += GlobalID;
    });
  });

  auto ExecGraph = Graph.finalize(exp_ext::property::graph::updatable{});

  // first half of PtrA should be filled with values
  Queue.ext_oneapi_graph(ExecGraph).wait();

  Queue.copy(PtrA, HostDataA.data(), Size).wait();
  for (size_t i = 0; i < Size; i++) {
    assert(HostDataA[i] == i);
  }

  // Update NDRange to target first half only
  size_t UpdateSize = Size / 2;
  nd_range<1> NDRange{range{UpdateSize}, range{32}};
  KernelNode.update_nd_range(NDRange);
  ExecGraph.update(KernelNode);
  Queue.ext_oneapi_graph(ExecGraph).wait();

  Queue.copy(PtrA, HostDataA.data(), Size).wait();
  for (size_t i = 0; i < Size; i++) {
    assert(HostDataA[i] == (i >= UpdateSize ? i : i * 2));
  }
  return 0;
}
