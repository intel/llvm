// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
// Extra run to check for immediate-command-list in Level Zero
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// REQUIRES: aspect-usm_shared_allocations

// Tests that whole graph update works when using dynamic parameters.

#include "../graph_common.hpp"

int main() {
  queue Queue{};

  using T = int;

  std::vector<T> InputDataHost1(Size);
  std::vector<T> InputDataHost2(Size);
  std::vector<T> OutputDataHost1(Size);

  std::iota(InputDataHost1.begin(), InputDataHost1.end(), 1);
  std::iota(InputDataHost2.begin(), InputDataHost2.end(), 10);
  std::iota(OutputDataHost1.begin(), OutputDataHost1.end(), 100);

  T *InputDataDevice1 = malloc_device<T>(Size, Queue);
  T *InputDataDevice2 = malloc_device<T>(Size, Queue);
  T *OutputDataDevice1 = malloc_device<T>(Size, Queue);

  Queue.copy(InputDataHost1.data(), InputDataDevice1, Size);
  Queue.copy(InputDataHost2.data(), InputDataDevice2, Size);
  Queue.copy(OutputDataHost1.data(), OutputDataDevice1, Size);
  Queue.wait();

  exp_ext::command_graph GraphA{Queue.get_context(), Queue.get_device()};

  exp_ext::dynamic_parameter InputParam(GraphA, InputDataDevice1);
  GraphA.add([&](handler &CGH) {
    CGH.set_arg(1, InputParam);
    CGH.single_task([=]() {
      for (size_t i = 0; i < Size; i++) {
        OutputDataDevice1[i] = InputDataDevice1[i];
      }
    });
  });

  auto GraphExecA = GraphA.finalize();
  Queue.ext_oneapi_graph(GraphExecA).wait();

  Queue.copy(OutputDataDevice1, OutputDataHost1.data(), Size);
  Queue.wait_and_throw();

  for (size_t i = 0; i < Size; i++) {
    assert(check_value(i, InputDataHost1[i], OutputDataHost1[i],
                       "OutputDataHost1"));
  }

  InputParam.update(InputDataDevice2);
  exp_ext::command_graph GraphB{Queue.get_context(), Queue.get_device()};

  GraphB.add([&](handler &CGH) {
    CGH.single_task([=]() {
      for (size_t i = 0; i < Size; i++) {
        OutputDataDevice1[i] = InputDataDevice1[i];
      }
    });
  });

  auto GraphExecB = GraphB.finalize(exp_ext::property::graph::updatable{});
  GraphExecB.update(GraphA);
  Queue.ext_oneapi_graph(GraphExecB).wait();

  Queue.copy(OutputDataDevice1, OutputDataHost1.data(), Size);
  Queue.wait_and_throw();

  free(InputDataDevice1, Queue);
  free(InputDataDevice2, Queue);
  free(OutputDataDevice1, Queue);

  for (size_t i = 0; i < Size; i++) {
    assert(check_value(i, InputDataHost2[i], OutputDataHost1[i],
                       "OutputDataHost1"));
  }

  return 0;
}
