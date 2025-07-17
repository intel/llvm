// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
// Extra run to check for immediate-command-list in Level Zero
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Tests that attempting to finalize and create more than one executable graph
// containing allocations is an error.

#include "../graph_common.hpp"
#include <sycl/ext/oneapi/experimental/async_alloc/async_alloc.hpp>

using T = int;

int main() {
  queue Queue{};

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  std::vector<T> OutputData(Size);
  std::vector<T> ReferenceData(Size);

  std::iota(ReferenceData.begin(), ReferenceData.end(), 0);

  // Add alloc and free commands
  T *AsyncPtr = nullptr;
  // Add alloc node
  auto AllocNode = Graph.add([&](handler &CGH) {
    AsyncPtr = static_cast<T *>(
        exp_ext::async_malloc(CGH, usm::alloc::device, Size * sizeof(T)));
  });

  auto KernelNode = Graph.add(
      [&](handler &CGH) {
        CGH.parallel_for(range<1>(Size), [=](item<1> ID) {
          size_t LinID = ID.get_linear_id();
          AsyncPtr[LinID] = static_cast<T>(LinID);
        });
      },
      {exp_ext::property::node::depends_on{AllocNode}});

  // Copy data out for verification
  auto CopyNode = Graph.add(
      [&](handler &CGH) {
        CGH.memcpy(OutputData.data(), AsyncPtr, Size * sizeof(T));
      },
      {exp_ext::property::node::depends_on{KernelNode}});

  // Free memory, node depends on only the associated allocation node
  Graph.add([&](handler &CGH) { exp_ext::async_free(CGH, AsyncPtr); },
            {exp_ext::property::node::depends_on{CopyNode}});

  // Constrain scope of GraphExec
  {
    auto GraphExec = Graph.finalize();
    // Graphs support CRS so copying here does not create a separate instance
    // and should be allowed.
    auto GraphExec2 = GraphExec;

    // Check that the graph executes correctly

    Queue.ext_oneapi_graph(GraphExec).wait_and_throw();

    for (size_t i = 0; i < Size; i++) {
      assert(check_value(i, ReferenceData[i], OutputData[i], "OutputData"));
    }

    std::error_code ExceptionCode = make_error_code(sycl::errc::success);
    try {
      // GraphExec is still alive so this should be an error
      Graph.finalize();
    } catch (const exception &e) {
      ExceptionCode = e.code();
    }

    assert(ExceptionCode == sycl::errc::invalid);
  }

  // GraphExec and GraphExec2 are now destroyed, so we should be able to
  // finalize again
  auto GraphExec = Graph.finalize();

  // Check that the graph executes correctly again

  Queue.ext_oneapi_graph(GraphExec).wait_and_throw();

  for (size_t i = 0; i < Size; i++) {
    assert(check_value(i, ReferenceData[i], OutputData[i], "OutputData"));
  }

  return 0;
}
