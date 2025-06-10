// REQUIRES: aspect-usm_shared_allocations
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// This test checks whether the free function command launch reduce is valid.
#define __DPCPP_ENABLE_UNFINISHED_KHR_EXTENSIONS

#include <sycl/detail/core.hpp>
#include <sycl/khr/free_function_commands.hpp>
#include <sycl/reduction.hpp>
#include <sycl/usm.hpp>

#include "helpers.hpp"

int main() {

  int Failed = 0;
  sycl::queue Queue;

  {
    int Result = 0;
    constexpr size_t Dim = 8;
    constexpr size_t N = Dim * Dim * Dim - 1;
    constexpr int ExpectedResult = N * (N + 1) / 2;
    {
      sycl::buffer<int> sumBuf{&Result, 1};
      Queue.submit([&](sycl::handler &Handler) {
        sycl::khr::launch_reduce(
            Handler, sycl::range<3>(Dim, Dim, Dim),
            [=](sycl::item<3> Item, auto &Sum) { Sum += Item.get_linear_id(); },
            sycl::reduction(sumBuf, Handler, sycl::plus<>()));
      });
    }
    Failed += Check(Result, ExpectedResult, "launch_reduce_with_buffer");
  }

  {
    int *Result = sycl::malloc_shared<int>(1, Queue);
    Result[0] = 0;
    constexpr size_t N = 1024;
    constexpr int ExpectedResult = ((N - 1) * N) / 2;

    sycl::event Event =
        sycl::khr::submit_tracked(Queue, [&](sycl::handler &Handler) {
          sycl::khr::launch_reduce(
              Handler, sycl::range<1>(N),
              [=](sycl::id<1> Id, auto &Sum) {
                int NegativeId = -(int)Id;
                Sum += NegativeId;
              },
              sycl::reduction(Result, sycl::plus<>()));
        });

    Queue.submit([&](sycl::handler &Handler) {
      Handler.depends_on(Event);
      sycl::khr::launch_reduce(
          Handler, sycl::range<1>(N),
          [=](sycl::item<1> Item, auto &Sum) { Sum += Item.get_linear_id(); },
          sycl::reduction(Result, sycl::plus<>()));
    });

    Queue.wait();
    Failed += Check(Result[0], 0, "launch_reduce_with_usm");

    sycl::khr::launch_reduce(
        Queue, sycl::range<1>(N), [=](sycl::id<1> Id, auto &Sum) { Sum += Id; },
        sycl::reduction(Result, sycl::plus<>()));

    Queue.wait();
    Failed +=
        Check(Result[0], ExpectedResult, "launch_reduce_shortcut_with_usm");
    sycl::free(Result, Queue);
  }

  return Failed;
}
