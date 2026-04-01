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
  {
    std::error_code ExceptionCode;
    try {
      auto Nodes = Graph.get_nodes();
      std::cerr << "ERROR: Expected exception was not thrown for get_nodes()"
                << std::endl;
      return 1;
    } catch (const sycl::exception &e) {
      ExceptionCode = e.code();
    }
    assert(ExceptionCode == sycl::errc::invalid);
  }

  // get_root_nodes() should throw errc::invalid for native recording graphs
  {
    std::error_code ExceptionCode;
    try {
      auto Nodes = Graph.get_root_nodes();
      std::cerr
          << "ERROR: Expected exception was not thrown for get_root_nodes()"
          << std::endl;
      return 1;
    } catch (const sycl::exception &e) {
      ExceptionCode = e.code();
    }
    assert(ExceptionCode == sycl::errc::invalid);
  }

  return 0;
}
