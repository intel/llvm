// Tests that waiting on an event signaled during graph recording
// throws.

#include "../graph_common.hpp"

#include <sycl/properties/all_properties.hpp>

int main() {
  queue Queue{property::queue::in_order{}};

#ifdef GRAPH_E2E_NATIVE_RECORDING
  exp_ext::command_graph Graph{
      Queue.get_context(),
      Queue.get_device(),
      {exp_ext::property::graph::enable_native_recording{}}};
#else
  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};
#endif

  Graph.begin_recording(Queue);

  auto GraphEvent = Queue.single_task([]() {});

  // TODO: The SYCL graph spec mandates errc::invalid but L0 graph's generic
  // error codes make it difficult to determine the root cause. Once L0 graph
  // expands their error codes we can add our handling and return the correct
  // SYCL code along with a descriptive message to the user.
#ifdef GRAPH_E2E_NATIVE_RECORDING
  if (!expectException([&]() { GraphEvent.wait(); },
                       "event wait during graph recording", errc::runtime)) {
    return 1;
  }
#else
  if (!expectException([&]() { GraphEvent.wait(); },
                       "event wait during graph recording", errc::invalid)) {
    return 1;
  }
#endif

  Graph.end_recording();

  return 0;
}
