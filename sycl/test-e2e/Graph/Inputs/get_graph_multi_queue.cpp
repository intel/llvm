// Test ext_oneapi_get_graph() with multiple queues (fork-join pattern)

#include "../graph_common.hpp"
#include <sycl/properties/all_properties.hpp>

int main() {
  device Dev;
  context Ctx{Dev};

  queue Queue1{Ctx, Dev, {property::queue::in_order{}}};
  queue Queue2{Ctx, Dev, {property::queue::in_order{}}};

#ifdef GRAPH_E2E_NATIVE_RECORDING
  exp_ext::command_graph Graph{
      Ctx, Dev, {exp_ext::property::graph::enable_native_recording{}}};
#else
  exp_ext::command_graph Graph{Ctx, Dev};
#endif

  assert(expectException([&]() { Queue1.ext_oneapi_get_graph(); },
                         "Queue1.ext_oneapi_get_graph() before begin_recording",
                         sycl::errc::invalid) &&
         "Expected exception on Queue1 before begin_recording");
  assert(expectException([&]() { Queue2.ext_oneapi_get_graph(); },
                         "Queue2.ext_oneapi_get_graph() before begin_recording",
                         sycl::errc::invalid) &&
         "Expected exception on Queue2 before begin_recording");

  Graph.begin_recording(Queue1);

  auto RetrievedGraph1 = Queue1.ext_oneapi_get_graph();
  assert(RetrievedGraph1 == Graph);

  auto ForkEvent = Queue1.ext_oneapi_submit_barrier();
  auto JoinEvent = Queue2.ext_oneapi_submit_barrier({ForkEvent});

  auto RetrievedGraph2 = Queue2.ext_oneapi_get_graph();
  assert(RetrievedGraph2 == Graph);

  Queue1.ext_oneapi_submit_barrier({JoinEvent});

  Graph.end_recording();

  assert(expectException([&]() { Queue1.ext_oneapi_get_graph(); },
                         "Queue1.ext_oneapi_get_graph() after end_recording",
                         sycl::errc::invalid) &&
         "Expected exception on Queue1 after end_recording");
  assert(expectException([&]() { Queue2.ext_oneapi_get_graph(); },
                         "Queue2.ext_oneapi_get_graph() after end_recording",
                         sycl::errc::invalid) &&
         "Expected exception on Queue2 after end_recording");

  return 0;
}
