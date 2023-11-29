// RUN: %{build} -o %t.out
// RUN: %{run-unfiltered-devices} %t.out
//

// This test checks that an expection is thrown when we try to
// record a graph whose device differs from the queue device.
// We ensure that the exception code matches the expected code.

#include "../graph_common.hpp"

int GetLZeroBackend(const sycl::device &Dev) {
  // Return 1 if the device backend is "Level_zero" or 0 else.
  // 0 does not prevent another device to be picked as a second choice
  return Dev.get_backend() == backend::ext_oneapi_level_zero;
}

int GetOtherBackend(const sycl::device &Dev) {
  // Return 1 if the device backend is not "Level_zero" or 0 else.
  // 0 does not prevent another device to be picked as a second choice
  return Dev.get_backend() != backend::ext_oneapi_level_zero;
}

int main() {
  sycl::device Dev0{GetLZeroBackend};
  sycl::device Dev1{GetOtherBackend};

  if (Dev0 == Dev1) {
    // Skip if we don't have two different devices
    std::cout << "Test skipped: the devices are the same" << std::endl;
    return 0;
  }

  queue Queue{Dev1};

  if (!are_graphs_supported(Queue)) {
    return 0;
  }

  exp_ext::command_graph Graph{Queue.get_context(), Dev0};

  std::error_code ExceptionCode = make_error_code(sycl::errc::success);
  try {
    Graph.begin_recording(Queue);
  } catch (exception &Exception) {
    ExceptionCode = Exception.code();
  }
  assert(ExceptionCode == sycl::errc::invalid);

  return 0;
}
