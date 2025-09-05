#pragma once

#include <sycl/builtins.hpp>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/profiling_tag.hpp>

#define CHECK(Counter, Check)                                                  \
  if (!(Check)) {                                                              \
    std::cout << #Check << " Failed!" << std::endl;                            \
    ++Counter;                                                                 \
  }

int run_test_on_queue(sycl::queue &Queue) {
  Queue.single_task([]() {
    volatile float X = 1.0f;
    for (size_t I = 0; I < 200; ++I)
      X = sycl::sin(X);
  });
  sycl::event StartTagE =
      sycl::ext::oneapi::experimental::submit_profiling_tag(Queue);
  sycl::event E1 = Queue.single_task([]() {
    volatile float X = 1.0f;
    for (size_t I = 0; I < 200; ++I)
      X = sycl::sin(X);
  });
  sycl::event E2 = Queue.single_task([]() {
    volatile float X = 1.0f;
    for (size_t I = 0; I < 200; ++I)
      X = sycl::cos(X);
  });
  sycl::event EndTagE =
      sycl::ext::oneapi::experimental::submit_profiling_tag(Queue);

  Queue.wait();

  int Failures = 0;

  uint64_t StartTagSubmit =
      StartTagE
          .get_profiling_info<sycl::info::event_profiling::command_submit>();
  uint64_t StartTagStart =
      StartTagE
          .get_profiling_info<sycl::info::event_profiling::command_start>();
  uint64_t StartTagEnd =
      StartTagE.get_profiling_info<sycl::info::event_profiling::command_end>();
  uint64_t EndTagSubmit =
      EndTagE.get_profiling_info<sycl::info::event_profiling::command_submit>();
  uint64_t EndTagStart =
      EndTagE.get_profiling_info<sycl::info::event_profiling::command_start>();
  uint64_t EndTagEnd =
      EndTagE.get_profiling_info<sycl::info::event_profiling::command_end>();

  std::cout << "StartTagSubmit: " << StartTagSubmit << std::endl;
  std::cout << "StartTagStart: " << StartTagStart << std::endl;
  std::cout << "StartTagEnd: " << StartTagEnd << std::endl;
  std::cout << "EndTagSubmit: " << EndTagSubmit << std::endl;
  std::cout << "EndTagStart: " << EndTagStart << std::endl;
  std::cout << "EndTagEnd: " << EndTagEnd << std::endl;

  CHECK(Failures, StartTagSubmit != 0)
  CHECK(Failures, StartTagStart != 0)
  CHECK(Failures, StartTagEnd != 0)
  CHECK(Failures, EndTagSubmit != 0)
  CHECK(Failures, EndTagStart != 0)
  CHECK(Failures, StartTagSubmit != 0)

  CHECK(Failures, StartTagSubmit <= StartTagEnd)
  CHECK(Failures, StartTagSubmit <= StartTagStart)
  CHECK(Failures, StartTagStart <= StartTagEnd)
  CHECK(Failures, EndTagSubmit <= EndTagEnd)
  CHECK(Failures, EndTagSubmit <= EndTagStart)
  CHECK(Failures, EndTagStart <= EndTagEnd)
  CHECK(Failures, StartTagEnd <= EndTagEnd)

  if (Queue.has_property<sycl::property::queue::enable_profiling>()) {
    uint64_t E1Start =
        E1.get_profiling_info<sycl::info::event_profiling::command_start>();
    uint64_t E1End =
        E1.get_profiling_info<sycl::info::event_profiling::command_end>();
    uint64_t E2Start =
        E2.get_profiling_info<sycl::info::event_profiling::command_start>();
    uint64_t E2End =
        E2.get_profiling_info<sycl::info::event_profiling::command_end>();

    std::cout << "E1Start: " << E1Start << std::endl;
    std::cout << "E1End: " << E1End << std::endl;
    std::cout << "E2Start: " << E2Start << std::endl;
    std::cout << "E2End: " << E2End << std::endl;

    CHECK(Failures, E1Start <= E1End)
    CHECK(Failures, E2Start <= E2End)

    CHECK(Failures, StartTagEnd <= E1Start)
    CHECK(Failures, StartTagEnd <= E2Start)

    CHECK(Failures, E1End <= EndTagEnd)
    CHECK(Failures, E2End <= EndTagEnd)
  }

  return Failures;
}
