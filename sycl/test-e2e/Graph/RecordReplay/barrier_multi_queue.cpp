// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
// Extra run to check for immediate-command-list in Level Zero
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
//

#include "../graph_common.hpp"

#include <sycl/properties/all_properties.hpp>

int main() {
  queue Queue1{{sycl::property::queue::in_order()}};
  queue Queue2{Queue1.get_context(),
               Queue1.get_device(),
               {sycl::property::queue::in_order()}};

  int *PtrA = malloc_device<int>(Size, Queue1);
  int *PtrB = malloc_device<int>(Size, Queue1);

  exp_ext::command_graph Graph{Queue1};
  Graph.begin_recording({Queue1, Queue2});

  auto EventA = Queue1.submit([&](handler &CGH) {
    CGH.parallel_for(range<1>{Size}, [=](id<1> it) { PtrA[it] = it; });
  });

  Queue2.ext_oneapi_submit_barrier({EventA});

  auto EventB = Queue2.copy(PtrA, PtrB, Size);
  Graph.end_recording();

  auto ExecGraph = Graph.finalize();
  Queue1.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecGraph); });

  std::array<int, Size> Output;
  Queue1.memcpy(Output.data(), PtrB, sizeof(int) * Size).wait();

  for (int i = 0; i < Size; i++) {
    assert(Output[i] == i);
  }

  free(PtrA, Queue1);
  free(PtrB, Queue1);
  return 0;
}
