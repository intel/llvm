// REQUIRES: aspect-usm_shared_allocations
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// A barrier with a wait list must wait for events which belong to a context
// other than the context of the queue the barrier is submitted to. This is
// checked for both handler::ext_oneapi_barrier and
// queue::ext_oneapi_submit_barrier.

#include <sycl/detail/core.hpp>

#include <sycl/builtins.hpp>
#include <sycl/usm.hpp>

#include <iostream>
#include <vector>

constexpr int Expected = 42;

template <typename SubmitBarrierT>
static int test(const char *Name, SubmitBarrierT SubmitBarrier) {
  sycl::device Dev;
  sycl::context Ctx1{Dev};
  sycl::context Ctx2{Dev};
  sycl::queue Q1{Ctx1, Dev};
  sycl::queue Q2{Ctx2, Dev};

  int *Data = sycl::malloc_shared<int>(1, Q1);
  *Data = 0;

  sycl::event KernelEvent = Q1.single_task([=]() {
    // Do some busywork, so that the kernel is likely still running when the
    // barrier is submitted.
    volatile float Y = 1.0f;
    for (int I = 0; I < 100000; ++I)
      Y = sycl::cos(Y);
    *Data = Expected;
  });

  sycl::event BarrierEvent = SubmitBarrier(Q2, KernelEvent);

  int Observed = -1;
  bool KernelComplete = false;
  Q2.submit([&](sycl::handler &CGH) {
    CGH.depends_on(BarrierEvent);
    CGH.host_task([&]() {
      KernelComplete =
          KernelEvent.get_info<sycl::info::event::command_execution_status>() ==
          sycl::info::event_command_status::complete;
      Observed = *Data;
    });
  });
  Q2.wait_and_throw();
  Q1.wait_and_throw();

  sycl::free(Data, Q1);

  int Error = 0;
  if (!KernelComplete) {
    std::cout << "FAIL: " << Name
              << ": the kernel was not complete after the barrier\n";
    Error = 1;
  }
  if (Observed != Expected) {
    std::cout << "FAIL: " << Name << ": got " << Observed << ", expected "
              << Expected << "\n";
    Error = 1;
  }
  return Error;
}

int main() {
  int Error = 0;

  Error +=
      test("handler::ext_oneapi_barrier", [](sycl::queue &Q, sycl::event E) {
        return Q.submit([&](sycl::handler &CGH) {
          CGH.ext_oneapi_barrier(std::vector<sycl::event>{E});
        });
      });

  Error += test(
      "queue::ext_oneapi_submit_barrier", [](sycl::queue &Q, sycl::event E) {
        return Q.ext_oneapi_submit_barrier(std::vector<sycl::event>{E});
      });

  std::cout << (Error ? "failed\n" : "passed\n");
  return Error;
}
