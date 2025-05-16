// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
// Extra run to check for immediate-command-list in Level Zero
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Tests using a dynamic_local_accessor with multiple nodes.

#include "../graph_common.hpp"
#include <sycl/group_barrier.hpp>

int main() {
  queue Queue{};

  constexpr int LocalSize{16};

  using T = int;

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  int *PtrA = malloc_device<T>(Size * Size, Queue);
  std::vector<T> HostDataA(Size * Size);

  exp_ext::dynamic_local_accessor<T, 2> DynLocalMemA{
      Graph, range<2>{LocalSize, LocalSize}};
  exp_ext::dynamic_local_accessor<T, 1> DynLocalMemC{Graph,
                                                     range<1>{LocalSize}};

  Queue.memset(PtrA, 0, Size * Size * sizeof(T)).wait();

  nd_range<2> NDRange2D{range<2>{Size, Size}, range<2>{LocalSize, LocalSize}};

  auto KernelNodeA = Graph.add([&](handler &cgh) {
    cgh.parallel_for(NDRange2D, [=](nd_item<2> Item) {
      size_t GlobalID = Item.get_global_linear_id();
      auto LocalRange = Item.get_local_range(0);
      const auto i = Item.get_local_id()[0];
      const auto j = Item.get_local_id()[1];

      auto LocalMemA = DynLocalMemA.get();

      LocalMemA[i][j] = LocalRange;
      group_barrier(Item.get_group());

      for (size_t k{0}; k < LocalRange; ++k) {
        for (size_t z{0}; z < LocalRange; ++z) {
          PtrA[GlobalID] += (T)(LocalMemA[k][z]);
        }
      }
    });
  });

  auto KernelNodeB = Graph.add(
      [&](handler &cgh) {
        cgh.parallel_for(NDRange2D, [=](nd_item<2> Item) {
          size_t GlobalID = Item.get_global_linear_id();
          auto LocalRange = Item.get_local_range(0);
          const auto i = Item.get_local_id()[0];
          const auto j = Item.get_local_id()[1];

          auto LocalMemA = DynLocalMemA.get();

          LocalMemA[i][j] = LocalRange;
          group_barrier(Item.get_group());

          // Substracting what was added in NodeA gives 0.
          for (size_t k{0}; k < LocalRange; ++k) {
            for (size_t z{0}; z < LocalRange; ++z) {
              PtrA[GlobalID] -= (T)(LocalMemA[k][z]);
            }
          }
        });
      },
      exp_ext::property::node::depends_on{KernelNodeA});

  nd_range<1> NDRange{Size * Size, LocalSize};
  auto KernelNodeC = Graph.add(
      [&](handler &cgh) {
        cgh.parallel_for(NDRange, [=](nd_item<1> Item) {
          size_t GlobalID = Item.get_global_id();
          auto LocalRange = Item.get_local_range(0);

          auto LocalMemC = DynLocalMemC.get();

          LocalMemC[Item.get_local_id()] = LocalRange;
          group_barrier(Item.get_group());

          for (size_t i{0}; i < LocalRange; ++i) {
            PtrA[GlobalID] += (T)(LocalMemC[i]);
          }
        });
      },
      exp_ext::property::node::depends_on{KernelNodeB});

  auto ExecGraph = Graph.finalize(exp_ext::property::graph::updatable{});

  Queue.ext_oneapi_graph(ExecGraph).wait();

  Queue.copy(PtrA, HostDataA.data(), Size * Size).wait();
  for (size_t i = 0; i < Size * Size; i++) {
    assert(HostDataA[i] == LocalSize * LocalSize);
  }

  Queue.memset(PtrA, 0, Size * Size * sizeof(T)).wait();

  constexpr size_t NewLocalSize{32};

  DynLocalMemA.update(range<2>{NewLocalSize, NewLocalSize});
  DynLocalMemC.update(range<1>{NewLocalSize});

  KernelNodeA.update_nd_range(
      nd_range<2>{range<2>{Size, Size}, range<2>{NewLocalSize, NewLocalSize}});
  KernelNodeB.update_nd_range(
      nd_range<2>{range<2>{Size, Size}, range<2>{NewLocalSize, NewLocalSize}});
  KernelNodeC.update_nd_range(nd_range<1>{Size * Size, NewLocalSize});

  ExecGraph.update(KernelNodeA);
  ExecGraph.update(KernelNodeB);
  ExecGraph.update(KernelNodeC);

  Queue.ext_oneapi_graph(ExecGraph).wait();

  Queue.copy(PtrA, HostDataA.data(), Size * Size).wait();
  for (size_t i = 0; i < Size * Size; i++) {
    assert(HostDataA[i] == NewLocalSize * NewLocalSize);
  }

  free(PtrA, Queue);
  return 0;
}