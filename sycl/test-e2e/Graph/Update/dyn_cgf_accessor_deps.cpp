// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
// Extra run to check for immediate-command-list in Level Zero
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Tests adding a dynamic command-group node to a graph using buffer
// accessors for the node edges.

#include "../graph_common.hpp"

int main() {
  queue Queue{};
  int *Ptr = (int *)sycl::malloc_device<int>(Size, Queue);
  buffer<int> Buf{sycl::range<1>(Size)};
  Buf.set_write_back(false);

  exp_ext::command_graph Graph{
      Queue.get_context(),
      Queue.get_device(),
      {exp_ext::property::graph::assume_buffer_outlives_graph{}}};

  auto RootNode = Graph.add([&](handler &CGH) {
    auto Acc = Buf.get_access<access::mode::write>(CGH);
    CGH.parallel_for(Size, [=](item<1> Item) { Acc[Item.get_id()] = 1; });
  });

  int PatternA = 42;
  auto CGFA = [&](handler &CGH) {
    auto Acc = Buf.get_access<access::mode::read_write>(CGH);
    CGH.parallel_for(Size,
                     [=](item<1> Item) { Acc[Item.get_id()] += PatternA; });
  };

  int PatternB = 0xA;
  auto CGFB = [&](handler &CGH) {
    auto Acc = Buf.get_access<access::mode::read_write>(CGH);
    CGH.parallel_for(Size,
                     [=](item<1> Item) { Acc[Item.get_id()] += PatternB; });
  };

  auto DynamicCG = exp_ext::dynamic_command_group(Graph, {CGFA, CGFB});
  auto DynamicCGNode = Graph.add(DynamicCG);

  auto LeafNode = Graph.add([&](handler &CGH) {
    auto Acc = Buf.get_access<access::mode::read>(CGH);
    CGH.parallel_for(
        Size, [=](item<1> Item) { Ptr[Item.get_id()] = Acc[Item.get_id()]; });
  });
  auto ExecGraph = Graph.finalize(exp_ext::property::graph::updatable{});

  Queue.ext_oneapi_graph(ExecGraph).wait();

  std::vector<int> HostData(Size, 0);
  Queue.copy(Ptr, HostData.data(), Size).wait();

  int Ref = PatternA + 1;
  for (size_t i = 0; i < Size; i++) {
    assert(HostData[i] == Ref);
  }

  DynamicCG.set_active_index(1);
  ExecGraph.update(DynamicCGNode);

  Queue.ext_oneapi_graph(ExecGraph).wait();
  Queue.copy(Ptr, HostData.data(), Size).wait();
  Ref = PatternB + 1;
  for (size_t i = 0; i < Size; i++) {
    assert(HostData[i] == Ref);
  }

  sycl::free(Ptr, Queue);

  return 0;
}
