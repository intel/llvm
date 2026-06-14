// REQUIRES: aspect-ext_oneapi_device_wait

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>
#include <sycl/properties/all_properties.hpp>

#include <array>
#include <vector>

constexpr size_t NContexts = 2;
constexpr size_t NQueues = 6;

int main() try {
  sycl::device D;
  std::array<sycl::context, NContexts> Contexts{sycl::context{D},
                                                sycl::context{D}};
  std::array<sycl::queue, NQueues> Queues{
      sycl::queue{Contexts[0], D},
      sycl::queue{Contexts[0], D, sycl::property::queue::in_order()},
      sycl::queue{Contexts[0], D},
      sycl::queue{Contexts[1], D, sycl::property::queue::in_order()},
      sycl::queue{Contexts[1], D},
      sycl::queue{Contexts[1], D, sycl::property::queue::in_order()}};

  // Queue registers itself to the per-device queue list when it's created,
  // And unregisters itself when it's destructed. This test verifies that the
  // queue list is updated correctly when the queue is destructed. Otherwise,
  // ext_oneapi_wait call would assert fail.
  {
    sycl::queue QTemp{Contexts[0], D};
  }

  std::vector<sycl::event> Events;
  Events.reserve(NQueues);
  for (sycl::queue &Q : Queues) {
    sycl::event E = Q.single_task([]() {
      volatile int value = 1024 * 1024;
      while (--value)
        ;
    });
    Events.push_back(std::move(E));
  }

  D.ext_oneapi_wait();

  int Failed = 0;
  for (size_t I = 0; I < Events.size(); ++I) {
    sycl::info::event_command_status EventStatus =
        Events[I].get_info<sycl::info::event::command_execution_status>();
    if (EventStatus != sycl::info::event_command_status::complete) {
      std::cout << "Unexpected event status for event at " << I << std::endl;
      ++Failed;
    }
  }
  return Failed;
} catch (sycl::exception &e) {
  std::cout << "Exception thrown: " << e.what() << std::endl;
  throw;
}
