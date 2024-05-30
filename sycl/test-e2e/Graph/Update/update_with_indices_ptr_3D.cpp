// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
// Extra run to check for immediate-command-list in Level Zero
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
//

// Tests updating a 3D ND-Range graph kernel node using index-based explicit
// update

#include "../graph_common.hpp"

int main() {
  queue Queue{};

  const range<3> GlobalWorkSize(1, 2, 2);
  const range<3> LocalWorkSize(1, 2, 2);
  const size_t N = GlobalWorkSize[0] * GlobalWorkSize[1] * GlobalWorkSize[2];

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  int *PtrA = malloc_device<int>(N, Queue);
  int *PtrB = malloc_device<int>(N, Queue);

  std::vector<int> HostDataA(N);
  std::vector<int> HostDataB(N);

  Queue.memset(PtrA, 0, N * sizeof(int)).wait();
  Queue.memset(PtrB, 0, N * sizeof(int)).wait();

  exp_ext::dynamic_parameter DynParam(Graph, PtrA);

  nd_range<3> NDRange{GlobalWorkSize, LocalWorkSize};
  auto NodeA = Graph.add([&](handler &cgh) {
    cgh.set_arg(0, DynParam);
    // TODO: Use the free function kernel extension instead of regular kernels
    // when available.
    cgh.parallel_for(NDRange, [=](nd_item<3> Item) {
      size_t GlobalID = Item.get_global_linear_id();
      PtrA[GlobalID] = GlobalID;
    });
  });

  range<3> Range{GlobalWorkSize};
  auto NodeB = Graph.add(
      [&](handler &cgh) {
        cgh.set_arg(0, DynParam);
        // TODO: Use the free function kernel extension instead of regular
        // kernels when available.
        cgh.parallel_for(Range, [=](item<3> Item) {
          size_t GlobalID = Item.get_linear_id();
          PtrA[GlobalID] *= 2;
        });
      },
      exp_ext::property::node::depends_on{NodeA});

  auto ExecGraph = Graph.finalize(exp_ext::property::graph::updatable{});

  // PtrA should be filled with values
  Queue.ext_oneapi_graph(ExecGraph).wait();

  Queue.copy(PtrA, HostDataA.data(), N).wait();
  Queue.copy(PtrB, HostDataB.data(), N).wait();
  for (size_t i = 0; i < N; i++) {
    assert(HostDataA[i] == (i * 2));
    assert(HostDataB[i] == 0);
  }

  // Swap PtrB to be the input/output
  DynParam.update(PtrB);
  ExecGraph.update({NodeA, NodeB});
  Queue.ext_oneapi_graph(ExecGraph).wait();

  Queue.copy(PtrA, HostDataA.data(), N).wait();
  Queue.copy(PtrB, HostDataB.data(), N).wait();
  for (size_t i = 0; i < N; i++) {
    const size_t Ref = i * 2;
    assert(HostDataA[i] == Ref);
    assert(HostDataB[i] == Ref);
  }
  return 0;
}
