// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
//
// UNSUPPORTED: level_zero && windows && gpu-intel-gen12
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/20696
//
// REQUIRES: aspect-usm_host_allocations

// Tests injected barrier between an in-order operation in no event mode and a
// graph consisting of a single host_task. Test attempts to produce a race
// condition if the barrier is not correctly injected.

#include "../graph_common.hpp"

#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/properties/all_properties.hpp>

int main() {
  constexpr int KernelValue = 1;
  constexpr int HostTaskValue = 7;
  sycl::queue Q{sycl::property::queue::in_order{}};

  int *HostUSM = sycl::malloc_host<int>(1, Q);

  // Record graph with a single host_task that overwrites the value.
  sycl::ext::oneapi::experimental::command_graph Graph{Q.get_context(),
                                                       Q.get_device()};
  Graph.begin_recording(Q);
  Q.submit([&](sycl::handler &H) {
    H.host_task([=]() { *HostUSM = HostTaskValue; });
  });
  Graph.end_recording(Q);
  auto ExecGraph = Graph.finalize();

  exp_ext::single_task(Q, [=]() {
    // Empirically determined to trigger race condition when
    // barrier is removed.
    int SpinIters = 500;
    for (volatile int i = 0; i < SpinIters; i += 1) {
      *HostUSM = KernelValue;
    }
  });

  // Due to in-order queue, implicit dependency on prior event. Scheduler should
  // inject barrier.
  exp_ext::execute_graph(Q, ExecGraph);
  Q.wait_and_throw();
  int ActualValue = *HostUSM;
  assert(check_value(0, HostTaskValue, ActualValue, "HostUSM"));
  sycl::free(HostUSM, Q);
  return 0;
}
