// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
// Extra run to check for immediate-command-list in Level Zero
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// XFAIL: level_zero
// XFAIL-TRACKER: OFNAAO-307

// Tests adding a dynamic command-group node to a graph using buffer
// accessors for the node edges, but where different command-groups
// use different buffers that create identical edges.

#include "../graph_common.hpp"

int main() {
  queue Queue{};
  int *Ptr = (int *)sycl::malloc_device<int>(Size, Queue);
  buffer<int> BufA{sycl::range<1>(Size)};
  buffer<int> BufB{sycl::range<1>(Size)};
  BufA.set_write_back(false);
  BufB.set_write_back(false);

  exp_ext::command_graph Graph{
      Queue.get_context(),
      Queue.get_device(),
      {exp_ext::property::graph::assume_buffer_outlives_graph{}}};

  int InitA = 4;
  int InitB = -4;
  auto RootNode = Graph.add([&](handler &CGH) {
    auto AccA = BufA.get_access<access::mode::write>(CGH);
    auto AccB = BufB.get_access<access::mode::write>(CGH);
    CGH.parallel_for(Size, [=](item<1> Item) {
      AccA[Item.get_id()] = InitA;
      AccB[Item.get_id()] = InitB;
    });
  });

  int PatternA = 42;
  auto CGFA = [&](handler &CGH) {
    auto AccA = BufA.get_access<access::mode::read_write>(CGH);
    CGH.parallel_for(Size,
                     [=](item<1> Item) { AccA[Item.get_id()] += PatternA; });
  };

  int PatternB = 0xA;
  auto CGFB = [&](handler &CGH) {
    auto AccB = BufB.get_access<access::mode::read_write>(CGH);
    CGH.parallel_for(Size,
                     [=](item<1> Item) { AccB[Item.get_id()] += PatternB; });
  };

  auto DynamicCG = exp_ext::dynamic_command_group(Graph, {CGFA, CGFB});
  auto DynamicCGNode = Graph.add(DynamicCG);

  auto LeafNode = Graph.add([&](handler &CGH) {
    auto AccA = BufA.get_access<access::mode::read>(CGH);
    auto AccB = BufB.get_access<access::mode::read>(CGH);
    CGH.parallel_for(Size, [=](item<1> Item) {
      Ptr[Item.get_id()] = AccA[Item.get_id()] + AccB[Item.get_id()];
    });
  });
  auto ExecGraph = Graph.finalize(exp_ext::property::graph::updatable{});

  Queue.ext_oneapi_graph(ExecGraph).wait();

  std::vector<int> HostData(Size, 0);
  Queue.copy(Ptr, HostData.data(), Size).wait();
  for (size_t i = 0; i < Size; i++) {
    assert(HostData[i] == (InitA + InitB + PatternA));
  }

  DynamicCG.set_active_cgf(1);
  ExecGraph.update(DynamicCGNode);

  Queue.ext_oneapi_graph(ExecGraph).wait();
  Queue.copy(Ptr, HostData.data(), Size).wait();
  int Ref = InitA + InitB + PatternB;
  for (size_t i = 0; i < Size; i++) {
    assert(HostData[i] == Ref);
  }

  sycl::free(Ptr, Queue);

  return 0;
}
