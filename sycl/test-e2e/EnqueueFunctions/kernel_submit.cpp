// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Tests the enqueue free function kernel submits.

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/usm.hpp>

#include "common.hpp"

#include <iostream>

namespace oneapiext = sycl::ext::oneapi::experimental;

constexpr size_t N = 1024;

int main() {
  sycl::queue Q;
  int Memory[N] = {0};

  int Failed = 0;

  // single_task
  {
    sycl::buffer<int, 1> MemBuf{Memory, sycl::range<1>{N}};
    {
      oneapiext::submit(Q, [&](sycl::handler &CGH) {
        sycl::accessor MemAcc{MemBuf, CGH, sycl::write_only};
        oneapiext::single_task(CGH, [=]() {
          for (size_t I = 0; I < N; ++I)
            MemAcc[I] = 42;
        });
      });
    }
  }
  for (size_t I = 0; I < N; ++I)
    Failed += Check(Memory, 42, I, "single_task");

  // 1D parallel_for
  {
    sycl::buffer<int, 1> MemBuf{Memory, sycl::range<1>{N}};
    {
      oneapiext::submit(Q, [&](sycl::handler &CGH) {
        sycl::accessor MemAcc{MemBuf, CGH, sycl::write_only};
        oneapiext::parallel_for(CGH, sycl::range<1>{N},
                                [=](sycl::item<1> Item) { MemAcc[Item] = 43; });
      });
    }
  }
  for (size_t I = 0; I < N; ++I)
    Failed += Check(Memory, 43, I, "1D parallel_for");

  // 2D parallel_for
  {
    sycl::range<2> Range{8, N / 8};
    sycl::buffer<int, 2> MemBuf{Memory, Range};
    {
      oneapiext::submit(Q, [&](sycl::handler &CGH) {
        sycl::accessor MemAcc{MemBuf, CGH, sycl::write_only};
        oneapiext::parallel_for(CGH, Range,
                                [=](sycl::item<2> Item) { MemAcc[Item] = 44; });
      });
    }
  }
  for (size_t I = 0; I < N; ++I)
    Failed += Check(Memory, 44, I, "2D parallel_for");

  // 3D parallel_for
  {
    sycl::range<3> Range{8, 8, N / 64};
    sycl::buffer<int, 3> MemBuf{Memory, Range};
    {
      oneapiext::submit(Q, [&](sycl::handler &CGH) {
        sycl::accessor MemAcc{MemBuf, CGH, sycl::write_only};
        oneapiext::parallel_for(CGH, Range,
                                [=](sycl::item<3> Item) { MemAcc[Item] = 45; });
      });
    }
  }
  for (size_t I = 0; I < N; ++I)
    Failed += Check(Memory, 45, I, "3D parallel_for");

  // 1D nd_launch
  {
    sycl::buffer<int, 1> MemBuf{Memory, sycl::range<1>{N}};
    {
      oneapiext::submit(Q, [&](sycl::handler &CGH) {
        sycl::accessor MemAcc{MemBuf, CGH, sycl::write_only};
        oneapiext::nd_launch(
            CGH, sycl::nd_range<1>{sycl::range<1>{N}, sycl::range{8}},
            [=](sycl::nd_item<1> Item) { MemAcc[Item.get_global_id()] = 46; });
      });
    }
  }
  for (size_t I = 0; I < N; ++I)
    Failed += Check(Memory, 46, I, "1D nd_launch");

  // 2D nd_launch
  {
    sycl::range<2> Range{8, N / 8};
    sycl::buffer<int, 2> MemBuf{Memory, Range};
    {
      oneapiext::submit(Q, [&](sycl::handler &CGH) {
        sycl::accessor MemAcc{MemBuf, CGH, sycl::write_only};
        oneapiext::nd_launch(
            CGH, sycl::nd_range<2>{Range, sycl::range{8, 8}},
            [=](sycl::nd_item<2> Item) { MemAcc[Item.get_global_id()] = 47; });
      });
    }
  }
  for (size_t I = 0; I < N; ++I)
    Failed += Check(Memory, 47, I, "2D nd_launch");

  // 3D nd_launch
  {
    sycl::range<3> Range{8, 8, N / 64};
    sycl::buffer<int, 3> MemBuf{Memory, Range};
    {
      oneapiext::submit(Q, [&](sycl::handler &CGH) {
        sycl::accessor MemAcc{MemBuf, CGH, sycl::write_only};
        oneapiext::nd_launch(
            CGH, sycl::nd_range<3>{Range, sycl::range{8, 8, 8}},
            [=](sycl::nd_item<3> Item) { MemAcc[Item.get_global_id()] = 48; });
      });
    }
  }
  for (size_t I = 0; I < N; ++I)
    Failed += Check(Memory, 48, I, "3D nd_launch");

  // 1D parallel_for with launch config
  {
    sycl::buffer<int, 1> MemBuf{Memory, sycl::range<1>{N}};
    {
      oneapiext::submit(Q, [&](sycl::handler &CGH) {
        sycl::accessor MemAcc{MemBuf, CGH, sycl::write_only};
        oneapiext::parallel_for(
            CGH, oneapiext::launch_config<sycl::range<1>>{sycl::range<1>{N}},
            [=](sycl::item<1> Item) { MemAcc[Item] = 49; });
      });
    }
  }
  for (size_t I = 0; I < N; ++I)
    Failed += Check(Memory, 49, I, "1D parallel_for with launch config");

  // 2D parallel_for with launch config
  {
    sycl::range<2> Range{8, N / 8};
    sycl::buffer<int, 2> MemBuf{Memory, Range};
    {
      oneapiext::submit(Q, [&](sycl::handler &CGH) {
        sycl::accessor MemAcc{MemBuf, CGH, sycl::write_only};
        oneapiext::parallel_for(CGH,
                                oneapiext::launch_config<sycl::range<2>>{Range},
                                [=](sycl::item<2> Item) { MemAcc[Item] = 50; });
      });
    }
  }
  for (size_t I = 0; I < N; ++I)
    Failed += Check(Memory, 50, I, "2D parallel_for with launch config");

  // 3D parallel_for with launch config
  {
    sycl::range<3> Range{8, 8, N / 64};
    sycl::buffer<int, 3> MemBuf{Memory, Range};
    {
      oneapiext::submit(Q, [&](sycl::handler &CGH) {
        sycl::accessor MemAcc{MemBuf, CGH, sycl::write_only};
        oneapiext::parallel_for(CGH,
                                oneapiext::launch_config<sycl::range<3>>{Range},
                                [=](sycl::item<3> Item) { MemAcc[Item] = 51; });
      });
    }
  }
  for (size_t I = 0; I < N; ++I)
    Failed += Check(Memory, 51, I, "3D parallel_for with launch config");

  // 1D nd_launch with launch config
  {
    sycl::buffer<int, 1> MemBuf{Memory, sycl::range<1>{N}};
    {
      oneapiext::submit(Q, [&](sycl::handler &CGH) {
        sycl::accessor MemAcc{MemBuf, CGH, sycl::write_only};
        oneapiext::nd_launch(
            CGH,
            oneapiext::launch_config<sycl::nd_range<1>>{
                sycl::nd_range<1>{sycl::range<1>{N}, sycl::range{8}}},
            [=](sycl::nd_item<1> Item) { MemAcc[Item.get_global_id()] = 52; });
      });
    }
  }
  for (size_t I = 0; I < N; ++I)
    Failed += Check(Memory, 52, I, "1D nd_launch with launch config");

  // 2D nd_launch with launch config
  {
    sycl::range<2> Range{8, N / 8};
    sycl::buffer<int, 2> MemBuf{Memory, Range};
    {
      oneapiext::submit(Q, [&](sycl::handler &CGH) {
        sycl::accessor MemAcc{MemBuf, CGH, sycl::write_only};
        oneapiext::nd_launch(
            CGH,
            oneapiext::launch_config<sycl::nd_range<2>>{
                sycl::nd_range<2>{Range, sycl::range{8, 8}}},
            [=](sycl::nd_item<2> Item) { MemAcc[Item.get_global_id()] = 53; });
      });
    }
  }
  for (size_t I = 0; I < N; ++I)
    Failed += Check(Memory, 53, I, "2D nd_launch with launch config");

  // 3D nd_launch with launch config
  {
    sycl::range<3> Range{8, 8, N / 64};
    sycl::buffer<int, 3> MemBuf{Memory, Range};
    {
      oneapiext::submit(Q, [&](sycl::handler &CGH) {
        sycl::accessor MemAcc{MemBuf, CGH, sycl::write_only};
        oneapiext::nd_launch(
            CGH,
            oneapiext::launch_config<sycl::nd_range<3>>{
                sycl::nd_range<3>{Range, sycl::range{8, 8, 8}}},
            [=](sycl::nd_item<3> Item) { MemAcc[Item.get_global_id()] = 54; });
      });
    }
  }
  for (size_t I = 0; I < N; ++I)
    Failed += Check(Memory, 54, I, "3D nd_launch with launch config");

  return Failed;
}
