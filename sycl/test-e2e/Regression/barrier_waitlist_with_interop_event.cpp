// REQUIRES: level_zero, level_zero_dev_kit
// RUN: %{build} %level_zero_options -o %t.out
// RUN: %{run} %t.out
// UNSUPPORTED: ze_debug

#include <level_zero/ze_api.h>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/usm.hpp>

// Test checks the case when an interoperability event is passed as a dependency
// to the barrier. In such case, waiting for the event produced by barrier must
// guarantee completion of the interoperability event.

using namespace sycl;

int main() {
  sycl::queue Queue;
  if (!Queue.get_device().get_info<info::device::usm_shared_allocations>())
    return 0;

  const size_t N = 1024;
  int *Data = sycl::malloc_shared<int>(N, Queue);
  auto FillEvent = Queue.fill(Data, 0, N);
  auto FillZeEvent = get_native<backend::ext_oneapi_level_zero>(FillEvent);

  backend_input_t<backend::ext_oneapi_level_zero, event> EventInteropInput = {
      FillZeEvent};
  EventInteropInput.Ownership = sycl::ext::oneapi::level_zero::ownership::keep;
  auto EventInterop = make_event<backend::ext_oneapi_level_zero>(
      EventInteropInput, Queue.get_context());

  auto BarrierEvent = Queue.ext_oneapi_submit_barrier({EventInterop});
  BarrierEvent.wait();

  if (EventInterop.get_info<sycl::info::event::command_execution_status>() !=
      sycl::info::event_command_status::complete) {
    Queue.wait();
    sycl::free(Data, Queue);
    return -1;
  }

  // Free the USM memory
  sycl::free(Data, Queue);

  return 0;
}
