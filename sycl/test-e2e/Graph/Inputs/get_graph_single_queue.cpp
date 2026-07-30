// Test ext_oneapi_get_graph() with single queue

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
  assert(expectException([&]() { Queue.ext_oneapi_get_graph(); },
                         "Queue.ext_oneapi_get_graph() before begin_recording",
                         sycl::errc::invalid) &&
         "Expected exception on Queue before begin_recording");

  Graph.begin_recording(Queue);

  auto RetrievedGraph1 = Queue.ext_oneapi_get_graph();
  assert(RetrievedGraph1 == Graph);

  Graph.end_recording(Queue);

  assert(expectException([&]() { Queue.ext_oneapi_get_graph(); },
                         "Queue.ext_oneapi_get_graph() after end_recording",
                         sycl::errc::invalid) &&
         "Expected exception on Queue1 after end_recording");

  return 0;
}
