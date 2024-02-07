// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for immediate-command-list in Level Zero
// RUN: %if level_zero && linux %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 %{run} %t.out %}
//

// This test checks that an expection is thrown when we try to
// record a graph whose context differs from the queue context.
// We ensure that the exception code matches the expected code.

#include "../graph_common.hpp"

int main() {
  queue Queue{{sycl::ext::intel::property::queue::no_immediate_command_list{}}};

  if (!are_graphs_supported(Queue)) {
    return 0;
  }

  context InOrderContext;

  exp_ext::command_graph Graph{InOrderContext, Queue.get_device()};

  std::error_code ExceptionCode = make_error_code(sycl::errc::success);
  try {
    Graph.begin_recording(Queue);
  } catch (exception &Exception) {
    ExceptionCode = Exception.code();
  }
  assert(ExceptionCode == sycl::errc::invalid);

  return 0;
}
