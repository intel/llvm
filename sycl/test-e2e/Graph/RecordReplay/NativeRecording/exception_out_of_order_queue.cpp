// REQUIRES: level_zero_v2_adapter && arch-intel_gpu_bmg_g21

// RUN: %{build} -o %t.out
// RUN: env SYCL_GRAPH_ENABLE_NATIVE_RECORDING=1 %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_GRAPH_ENABLE_NATIVE_RECORDING=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

#include "../../graph_common.hpp"

#include <sycl/properties/all_properties.hpp>

int main() {
  device Dev;
  context Ctx{Dev};
  queue OutOfOrderQueue{Ctx, Dev};
  exp_ext::command_graph Graph{Ctx, Dev};

  if (!expectException([&]() { Graph.begin_recording(OutOfOrderQueue); },
                         "begin_recording with out-of-order queue")) {
    std::cerr << "Out-of-order queue should throw exception"
              << std::endl;
    return 1;
  }


  return 0;
}
