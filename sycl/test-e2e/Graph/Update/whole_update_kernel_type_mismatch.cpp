// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
// Extra run to check for immediate-command-list in Level Zero
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Test that an error is thrown when the types of kernels do not match in Whole
// Graph Update

#include "../graph_common.hpp"

void testFunctors(queue Queue, int *Data) {
  exp_ext::command_graph Graph{Queue};
  exp_ext::command_graph UpdateGraph{Queue};
  struct KernelFunctorA {
    KernelFunctorA(int *Data) : Data(Data) {}

    void operator()() const { Data[0] = 42; }

    int *Data;
  };

  struct KernelFunctorB {
    KernelFunctorB(int *Data) : Data(Data) {}
    void operator()() const { Data[0] = 42; }

    int *Data;
  };

  Graph.add([&](handler &CGH) { CGH.single_task(KernelFunctorA{Data}); });

  UpdateGraph.add([&](handler &CGH) { CGH.single_task(KernelFunctorB{Data}); });

  auto GraphExec = Graph.finalize(exp_ext::property::graph::updatable{});

  // Check it's an error if kernel types don't match
  std::error_code ErrorCode = make_error_code(sycl::errc::success);
  try {
    GraphExec.update(UpdateGraph);
  } catch (const sycl::exception &e) {
    ErrorCode = e.code();
  }
  assert(ErrorCode == sycl::errc::invalid);
}

void testUnNamedLambdas(queue Queue, int *Data) {
  exp_ext::command_graph Graph{Queue};
  exp_ext::command_graph UpdateGraph{Queue};

  Graph.add([&](handler &CGH) { CGH.single_task([=]() { Data[0] = 42; }); });

  UpdateGraph.add(
      [&](handler &CGH) { CGH.single_task([=]() { Data[0] = 42; }); });

  auto GraphExec = Graph.finalize(exp_ext::property::graph::updatable{});

  // Check it's an error if kernel types don't match
  std::error_code ErrorCode = make_error_code(sycl::errc::success);
  try {
    GraphExec.update(UpdateGraph);
  } catch (const sycl::exception &e) {
    ErrorCode = e.code();
  }
  assert(ErrorCode == sycl::errc::invalid);
}
void testNamedLambdas(queue Queue, int *Data) {
  exp_ext::command_graph Graph{Queue};
  exp_ext::command_graph UpdateGraph{Queue};

  auto LambdaA = [=]() { Data[0] = 42; };

  Graph.add([&](handler &CGH) { CGH.single_task<class TestLambdaA>(LambdaA); });

  auto LambdaB = [=]() { Data[0] = 42; };

  UpdateGraph.add(
      [&](handler &CGH) { CGH.single_task<class TestLambdaB>(LambdaB); });

  auto GraphExec = Graph.finalize(exp_ext::property::graph::updatable{});

  // Check it's an error if kernel types don't match
  std::error_code ErrorCode = make_error_code(sycl::errc::success);
  try {
    GraphExec.update(UpdateGraph);
  } catch (const sycl::exception &e) {
    ErrorCode = e.code();
  }
  assert(ErrorCode == sycl::errc::invalid);
}

int main() {
  queue Queue{};
  int *Data = malloc_device<int>(1, Queue);

  testNamedLambdas(Queue, Data);
  testUnNamedLambdas(Queue, Data);
  testFunctors(Queue, Data);

  sycl::free(Data, Queue);

  return 0;
}
