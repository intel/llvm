// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// This test checks whether the free function command launch reduce is valid.

#include <sycl/detail/core.hpp>
#include <sycl/ext/khr/free_function_commands.hpp>
#include <sycl/reduction.hpp>
#include <sycl/usm.hpp>

#include "helpers.hpp"

int main() {

  int Failed = 0;
  sycl::queue Queue;

  // launch_reduced with sycl buffer
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
    Failed += Check(Result, ExpectedResult, "launch_reduce_with_sycl_buffer");
  }

  // launch_reduced with USM
  {
    int *Result = sycl::malloc_shared<int>(1, Queue);
    Result[0] = 0;
    constexpr size_t N = 1024;
    constexpr int ExpectedResult = (N - 1);
    {
      Queue.submit([&](sycl::handler &Handler) {
        sycl::khr::launch_reduce(
            Handler, sycl::range<1>(N),
            [=](sycl::id<1> Id, auto &Max) { Max.combine(Id); },
            sycl::reduction(Result, sycl::maximum<>()));
      });
    }

    Failed += Check(Result[0], ExpectedResult, "launch_reduce_with_sycl_usm");
    sycl::free(Result, Queue);
  }

  return Failed;
}
