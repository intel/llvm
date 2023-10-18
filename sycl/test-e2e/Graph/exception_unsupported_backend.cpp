// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Tests the ability to finalize a empty command graph
// The test checks that invalid exception is thrown
// when trying to create a graph with an unsupported backend.

#include "graph_common.hpp"

int GetUnsupportedBackend(const sycl::device &Dev) {
  // Return 1 if the device backend is unsupported or 0 else.
  // 0 does not prevent another device to be picked as a second choice
  return Dev.get_info<
             ext::oneapi::experimental::info::device::graph_support>() ==
         ext::oneapi::experimental::graph_support_level::unsupported;
}

int main() {
  sycl::device Dev{GetUnsupportedBackend};
  queue Queue{Dev};

  if (Dev.get_info<ext::oneapi::experimental::info::device::graph_support>() !=
      ext::oneapi::experimental::graph_support_level::unsupported)
    return 0;

  std::error_code ExceptionCode = make_error_code(sycl::errc::success);
  try {
    exp_ext::command_graph Graph{Queue.get_context(), Dev};
  } catch (exception &Exception) {
    ExceptionCode = Exception.code();
  }
  assert(ExceptionCode == sycl::errc::invalid);

  return 0;
}
