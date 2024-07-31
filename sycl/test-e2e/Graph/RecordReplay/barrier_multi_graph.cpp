// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
// Extra run to check for immediate-command-list in Level Zero
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
//

#include "../graph_common.hpp"

int main() {
  queue Queue{};

  int *PtrA = malloc_device<int>(Size, Queue);
  int *PtrB = malloc_device<int>(Size, Queue);

  exp_ext::command_graph GraphA{Queue};
  exp_ext::command_graph GraphB{Queue};

  GraphA.begin_recording(Queue);
  auto EventA = Queue.submit([&](handler &CGH) {
    CGH.parallel_for(range<1>{Size}, [=](id<1> it) { PtrA[it] = it; });
  });
  Queue.ext_oneapi_submit_barrier({EventA});
  Queue.copy(PtrA, PtrB, Size);
  GraphA.end_recording();

  GraphB.begin_recording(Queue);
  auto EventB = Queue.submit([&](handler &CGH) {
    CGH.parallel_for(range<1>{Size}, [=](id<1> it) { PtrA[it] = it * 2; });
  });
  Queue.ext_oneapi_submit_barrier();
  Queue.copy(PtrA, PtrB, Size);
  GraphB.end_recording();

  auto ExecGraphA = GraphA.finalize();
  auto ExecGraphB = GraphB.finalize();

  Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecGraphA); }).wait();

  std::array<int, Size> Output;
  Queue.memcpy(Output.data(), PtrB, sizeof(int) * Size).wait();

  for (int i = 0; i < Size; i++) {
    assert(Output[i] == i);
  }

  Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecGraphB); }).wait();
  Queue.memcpy(Output.data(), PtrB, sizeof(int) * Size).wait();

  for (int i = 0; i < Size; i++) {
    assert(Output[i] == 2 * i);
  }

  free(PtrA, Queue);
  free(PtrB, Queue);
  return 0;
}
