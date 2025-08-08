// REQUIRES: aspect-usm_shared_allocations
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// This test checks whether the free function command launch is valid.
#define __DPCPP_ENABLE_UNFINISHED_KHR_EXTENSIONS

#include <sycl/detail/core.hpp>
#include <sycl/khr/free_function_commands.hpp>
#include <sycl/usm.hpp>

#include <array>

#include "helpers.hpp"

int main() {

  int Failed = 0;

  sycl::queue Queue;
  constexpr size_t Dim = 8;

  {
    constexpr size_t N = Dim * Dim * Dim;

    std::array<int, N> Numbers = {0};
    {
      sycl::buffer<int, 1> MemBuffer{Numbers.data(), sycl::range<1>{N}};
      sycl::khr::submit(Queue, [&](sycl::handler &Handler) {
        sycl::accessor MemAcc{MemBuffer, Handler, sycl::write_only};
        sycl::khr::launch(Handler, sycl::range<3>(Dim, Dim, Dim),
                          [=](sycl::item<3> Item) {
                            size_t Index = Item.get_linear_id();
                            MemAcc[Index] = 901;
                          });
      });
    }
    for (size_t i = 0; i < N; ++i)
      Failed += Check(Numbers.data(), 901, i, "launch_with_buffer");
  }

  {
    constexpr size_t N = Dim * Dim;
    int *Numbers = sycl::malloc_shared<int>(N, Queue);

    sycl::event Event =
        sycl::khr::submit_tracked(Queue, [&](sycl::handler &Handler) {
          sycl::khr::launch(Handler, sycl::range<1>(N),
                            [=](sycl::item<1> Item) { Numbers[Item] = 2; });
        });

    sycl::khr::submit(Queue, [&](sycl::handler &Handler) {
      Handler.depends_on(Event);
      sycl::khr::launch(Handler, sycl::range<3>(Dim / 4, Dim / 2, Dim),
                        [=](sycl::item<3> Item) {
                          size_t Index = Item.get_linear_id();
                          Numbers[Index] = Numbers[Index] + 900;
                        });
    });

    Queue.wait();
    for (size_t i = 0; i < N; ++i)
      Failed += Check(Numbers, 902, i, "launch_with_usm");

    sycl::khr::launch(Queue, sycl::range<2>(Dim, Dim), [=](sycl::item<2> Item) {
      size_t Index = Item.get_linear_id();
      Numbers[Index] = 903;
    });

    Queue.wait();
    for (size_t i = 0; i < N; ++i)
      Failed += Check(Numbers, 903, i, "launch_shortcut_with_usm");

    sycl::free(Numbers, Queue);
  }

  return Failed;
}
