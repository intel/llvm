// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <condition_variable>

#include <sycl/detail/core.hpp>

#include <sycl/properties/all_properties.hpp>
#include <sycl/usm.hpp>

namespace syclex = sycl::ext::oneapi::experimental;

bool checkBarrierEvent(sycl::backend backend, sycl::event LastEvent,
                       sycl::event BarrierEvent, bool noEventMode) {
  // In noEventMode or when using opencl backend,
  // barrier will always return last event
  if (!noEventMode) {
    return BarrierEvent == LastEvent;
  } else {
    return BarrierEvent != LastEvent;
  }
}

int main() {
  sycl::queue Q({sycl::property::queue::in_order{}});
  int *Res = sycl::malloc_host<int>(1, Q);
  {
    // Test case 1 - regular case.
    std::cout << "Test 1" << std::endl;
    *Res = 1;
    auto Event1 = Q.submit([&](sycl::handler &cgh) {
      cgh.single_task<class kernel1>([=]() { *Res += 9; });
    });
    auto BarrierEvent1 = Q.ext_oneapi_submit_barrier();
    assert(checkBarrierEvent(Q.get_backend(), Event1, BarrierEvent1, true));
    auto Event2 = Q.submit([&](sycl::handler &cgh) {
      cgh.single_task<class kernel2>([=]() { *Res *= 2; });
    });

    auto BarrierEvent2 = Q.ext_oneapi_submit_barrier();
    assert(checkBarrierEvent(Q.get_backend(), Event1, BarrierEvent1, true));
    BarrierEvent2.wait();

    // Check that kernel events are completed after waiting for barrier event.
    assert(Event1.get_info<sycl::info::event::command_execution_status>() ==
           sycl::info::event_command_status::complete);
    assert(Event2.get_info<sycl::info::event::command_execution_status>() ==
           sycl::info::event_command_status::complete);
    assert(*Res == 20);
  }
  {
    // Test cast 2 - test case with the host task.
    std::cout << "Test 2" << std::endl;
    *Res = 0;

    auto Event1 = Q.submit(
        [&](sycl::handler &CGH) { CGH.host_task([&] { *Res += 1; }); });
    auto BarrierEvent1 = Q.ext_oneapi_submit_barrier();
    assert(checkBarrierEvent(Q.get_backend(), Event1, BarrierEvent1,
                             false /* host tasks used */));
    auto Event2 = Q.submit([&](sycl::handler &CGH) { CGH.fill(Res, 10, 1); });

    Q.wait();
    assert(*Res == 10);
  }

  {
    // Test cast 3 - empty queue.
    std::cout << "Test 3" << std::endl;
    sycl::queue EmptyQ({sycl::property::queue::in_order{}});
    auto BarrierEvent = EmptyQ.ext_oneapi_submit_barrier();
    assert(
        BarrierEvent.get_info<sycl::info::event::command_execution_status>() ==
        sycl::info::event_command_status::complete);
    BarrierEvent.wait();
  }

  {
    // Test cast 4 - graph.
    sycl::queue GQueue{sycl::property::queue::in_order{}};

    if (GQueue.get_device().has(sycl::aspect::ext_oneapi_graph)) {
      std::cout << "Test 4" << std::endl;
      syclex::command_graph Graph{GQueue.get_context(), GQueue.get_device()};
      *Res = 1;
      // Add commands to graph
      Graph.begin_recording(GQueue);
      auto BeforeBarrierEvent = GQueue.submit([&](sycl::handler &cgh) {
        cgh.single_task<class kernel3>([=]() { *Res += 9; });
      });
      auto Barrier = GQueue.ext_oneapi_submit_barrier();
      GQueue.submit([&](sycl::handler &cgh) {
        cgh.single_task<class kernel4>([=]() { *Res *= 2; });
      });
      Graph.end_recording(GQueue);
      auto GraphExec = Graph.finalize();

      auto Event = GQueue.submit(
          [&](sycl::handler &CGH) { CGH.ext_oneapi_graph(GraphExec); });
      Event.wait();

      assert(*Res == 20);
    }
  }

  return 0;
}
