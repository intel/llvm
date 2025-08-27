// REQUIRES: aspect-usm_shared_allocations
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// This test checks whether the free function command launch grouped reduce is
// valid.

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
        sycl::khr::launch_grouped_reduce(
            Handler, sycl::range<3>(Dim, Dim, Dim), sycl::range<3>(8, 8, 8),
            [=](sycl::nd_item<3> Item, auto &Sum) {
              Sum += Item.get_local_linear_id();
            },
            sycl::reduction(sumBuf, Handler, sycl::plus<>()));
      });
    }
    Failed +=
        Check(Result, ExpectedResult, "launch_grouped_reduce_with_buffer");
  }

  {
    int *Result = sycl::malloc_shared<int>(1, Queue);
    Result[0] = 0;
    constexpr size_t N = 1024;
    constexpr int ExpectedResult = ((N - 1) * N) / 2;

    sycl::event Event =
        sycl::khr::submit_tracked(Queue, [&](sycl::handler &Handler) {
          sycl::khr::launch_grouped_reduce(
              Handler, sycl::range<3>(8, 8, N / 64), sycl::range<3>(8, 8, 8),
              [=](sycl::nd_item<3> Item, auto &Sum) { Sum += 1; },
              sycl::reduction(Result, sycl::plus<>()));
        });

    sycl::khr::submit(Queue, [&](sycl::handler &Handler) {
      Handler.depends_on(Event);
      sycl::khr::launch_grouped_reduce(
          Handler, sycl::range<2>(16, N / 16), sycl::range<2>(8, 8),
          [=](sycl::nd_item<2> Item, auto &Sum) { Sum += 1; },
          sycl::reduction(Result, sycl::plus<>()));
    });
    Queue.wait();
    Failed += Check(Result[0], N * 2, "launch_grouped_reduce_with_usm");

    Result[0] = 0;
    sycl::khr::launch_grouped_reduce(
        Queue, sycl::range<1>(N), sycl::range<1>(8),
        [=](sycl::nd_item<1> Item, auto &Sum) {
          Sum += Item.get_global_linear_id();
        },
        sycl::reduction(Result, sycl::plus<>()));

    Queue.wait();
    Failed += Check(Result[0], ExpectedResult,
                    "launch_grouped_reduce_shortcut_with_usm");
    sycl::free(Result, Queue);
  }

  return Failed;
}
