// RUN: %{build} -o %t.out
// RUN: env SYCL_GRAPH_ENABLE_NATIVE_RECORDING=1 %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_GRAPH_ENABLE_NATIVE_RECORDING=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Test that native recording properly fails when used with non-immediate command lists

#include "../graph_common.hpp"

#include <sycl/properties/all_properties.hpp>

int main() {
  // Create a regular queue without immediate command list property
  queue Queue{{property::queue::in_order{}}};

  // Create a graph - native recording is enabled via SYCL_GRAPH_ENABLE_NATIVE_RECORDING
  // environment variable
  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  // This should throw an exception because native recording requires immediate command lists
  // (unless the global environment variable SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 is set)
  bool exceptionThrown = false;
  try {
    Graph.begin_recording(Queue);
  } catch (const sycl::exception& e) {
    std::string errorMsg = e.what();
    if (errorMsg.find("Native recording requires queues with immediate command lists") != std::string::npos) {
      exceptionThrown = true;
      std::cout << "Expected exception thrown: " << errorMsg << std::endl;
    }
  }

  if (!exceptionThrown) {
    std::cerr << "ERROR: Expected exception was not thrown!" << std::endl;
    return 1;
  }

  // Test with explicit no_immediate_command_list property
  queue QueueBatched{{
      property::queue::in_order{},
      ext::intel::property::queue::no_immediate_command_list{}
  }};

  bool exceptionThrown2 = false;
  try {
    Graph.begin_recording(QueueBatched);
  } catch (const sycl::exception& e) {
    std::string errorMsg = e.what();
    if (errorMsg.find("Native recording requires queues with immediate command lists") != std::string::npos) {
      exceptionThrown2 = true;
      std::cout << "Expected exception thrown for batched queue: " << errorMsg << std::endl;
    }
  }

  if (!exceptionThrown2) {
    std::cerr << "ERROR: Expected exception was not thrown for batched queue!" << std::endl;
    return 1;
  }

  std::cout << "All tests passed - native recording correctly validates queue command list mode" << std::endl;
  return 0;
}