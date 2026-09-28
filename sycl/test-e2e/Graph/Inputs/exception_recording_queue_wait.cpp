// Tests that waiting on a Queue in recording mode throws.

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

#ifdef GRAPH_E2E_NATIVE_RECORDING
  if (!expectException([&]() { Queue.wait(); },
                       "queue wait during graph recording", errc::runtime)) {
    return 1;
  }
#else
  if (!expectException([&]() { Queue.wait(); },
                       "queue wait during graph recording", errc::invalid)) {
    return 1;
  }
#endif
  Graph.end_recording();

  return 0;
}
