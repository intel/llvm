// REQUIRES: level_zero_v2_adapter && arch-intel_gpu_bmg_g21

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

#include "../../graph_common.hpp"

#include <sycl/properties/all_properties.hpp>

int main() {
  queue Queue{property::queue::in_order{}};

  exp_ext::command_graph Graph{
      Queue.get_context(),
      Queue.get_device(),
      {exp_ext::property::graph::enable_native_recording{}}};

  // get_nodes() should throw errc::invalid for native recording graphs
  if (!expectException([&]() { Graph.get_nodes(); },
                       "get_nodes() with enable_native_recording",
                       sycl::errc::invalid)) {
    return 1;
  }

  // get_root_nodes() should throw errc::invalid for native recording graphs
  if (!expectException([&]() { Graph.get_root_nodes(); },
                       "get_root_nodes() with enable_native_recording",
                       sycl::errc::invalid)) {
    return 1;
  }

  return 0;
}
