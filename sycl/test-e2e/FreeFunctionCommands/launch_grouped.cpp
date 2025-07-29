// REQUIRES: aspect-usm_shared_allocations
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// This test checks whether the free function command launch grouped is valid.

#define __DPCPP_ENABLE_UNFINISHED_KHR_EXTENSIONS

#include <sycl/detail/core.hpp>
#include <sycl/khr/free_function_commands.hpp>
#include <sycl/usm.hpp>

#include "helpers.hpp"

int main() {

  int Failed = 0;
  sycl::queue Queue;
  constexpr size_t N = 1024;

  {
    std::array<int, N> Numbers = {0};
    {
      sycl::buffer<int> MemBuffer{Numbers.data(), N};
      Queue.submit([&](sycl::handler &Handler) {
        sycl::accessor MemAcc{MemBuffer, Handler, sycl::write_only};
        sycl::khr::launch_grouped(Handler, sycl::range<3>(8, 8, N / 64),
                                  sycl::range<3>(4, 4, 4),
                                  [=](sycl::nd_item<3> Item) {
                                    size_t Index = Item.get_global_linear_id();
                                    MemAcc[Index] = 301;
                                  });
      });
    }
    for (size_t i = 0; i < N; ++i)
      Failed +=
          Check(Numbers.data(), 301, i, "launch_grouped_with_sycl_buffer");
  }

  {
    int *Numbers = sycl::malloc_shared<int>(N, Queue);

    sycl::event Event =
        sycl::khr::submit_tracked(Queue, [&](sycl::handler &Handler) {
          sycl::khr::launch_grouped(Handler, sycl::range<3>(4, 4, N / 16),
                                    sycl::range<3>(4, 4, 4),
                                    [=](sycl::nd_item<3> Item) {
                                      Numbers[Item.get_global_linear_id()] = 2;
                                    });
        });

    sycl::khr::submit(Queue, [&](sycl::handler &Handler) {
      Handler.depends_on(Event);
      sycl::khr::launch_grouped(Handler, sycl::range<1>(N), sycl::range<1>(8),
                                [=](sycl::nd_item<1> Item) {
                                  size_t Index = Item.get_global_linear_id();
                                  Numbers[Index] = Numbers[Index] + 300;
                                });
    });
    Queue.wait();
    for (size_t i = 0; i < N; ++i)
      Failed += Check(Numbers, 302, i, "launch_grouped_with_usm");

    sycl::khr::launch_grouped(Queue, sycl::range<2>(8, N / 8),
                              sycl::range<2>(8, 8), [=](sycl::nd_item<2> Item) {
                                Numbers[Item.get_global_linear_id()] = 303;
                              });
    Queue.wait();
    for (size_t i = 0; i < N; ++i)
      Failed += Check(Numbers, 303, i, "launch_grouped_shortcut_with_usm");
    sycl::free(Numbers, Queue);
  }

  return Failed;
}
