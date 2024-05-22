// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Tests that introducing a cycle to the graph will throw when
// property::graph::no_cycle_check is not passed to the graph constructor and
// will not throw when it is.

#include "../graph_common.hpp"

void CreateGraphWithCyclesTest(bool DisableCycleChecks) {

  // If we are testing without cycle checks we need to do multiple iterations so
  // we can test multiple types of cycle, since introducing a cycle with no
  // checks may put the graph into an undefined state.
  const size_t Iterations = DisableCycleChecks ? 2 : 1;

  queue Queue;

  property_list Props;

  if (DisableCycleChecks) {
    Props = {ext::oneapi::experimental::property::graph::no_cycle_check{}};
  }

  for (size_t i = 0; i < Iterations; i++) {
    ext::oneapi::experimental::command_graph Graph{Queue.get_context(),
                                                   Queue.get_device(), Props};

    auto NodeA = Graph.add([&](sycl::handler &CGH) {
      CGH.single_task<class testKernelA>([=]() {});
    });
    auto NodeB = Graph.add([&](sycl::handler &CGH) {
      CGH.single_task<class testKernelB>([=]() {});
    });
    auto NodeC = Graph.add([&](sycl::handler &CGH) {
      CGH.single_task<class testKernelC>([=]() {});
    });

    // Make normal edges
    std::error_code ErrorCode = sycl::make_error_code(sycl::errc::success);
    try {
      Graph.make_edge(NodeA, NodeB);
      Graph.make_edge(NodeB, NodeC);
    } catch (const sycl::exception &e) {
      ErrorCode = e.code();
    }

    assert(ErrorCode == sycl::errc::success);

    // Introduce cycles to the graph. If we are performing cycle checks we can
    // test both cycles, if they are disabled we need to test one per iteration.
    if (i == 0 || !DisableCycleChecks) {
      ErrorCode = sycl::make_error_code(sycl::errc::success);
      try {
        Graph.make_edge(NodeC, NodeA);
      } catch (const sycl::exception &e) {
        ErrorCode = e.code();
      }

      assert(ErrorCode ==
             (DisableCycleChecks ? sycl::errc::success : sycl::errc::invalid));
    }

    if (i == 1 || !DisableCycleChecks) {
      ErrorCode = sycl::make_error_code(sycl::errc::success);
      try {
        Graph.make_edge(NodeC, NodeB);
      } catch (const sycl::exception &e) {
        ErrorCode = e.code();
      }

      assert(ErrorCode ==
             (DisableCycleChecks ? sycl::errc::success : sycl::errc::invalid));
    }
  }
}

int main() {
  // Test with cycle checks
  CreateGraphWithCyclesTest(false);
  // Test without cycle checks
  CreateGraphWithCyclesTest(true);

  return 0;
}
