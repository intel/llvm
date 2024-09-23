// REQUIRES: aspect-usm_shared_allocations
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Tests the enqueue free function kernel shortcuts.

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/usm.hpp>

#include "common.hpp"

#include <iostream>

constexpr size_t N = 1024;

int main() {
  sycl::queue Q;
  int *Memory = sycl::malloc_shared<int>(N, Q);

  int Failed = 0;

  // single_task shortcut
  oneapiext::single_task(Q, [=]() {
    for (size_t I = 0; I < N; ++I)
      Memory[I] = 42;
  });
  Q.wait();
  for (size_t I = 0; I < N; ++I)
    Failed += Check(Memory, 42, I, "single_task shortcut");

  // 1D parallel_for shortcut
  oneapiext::parallel_for(Q, sycl::range<1>{N},
                          [=](sycl::item<1> Item) { Memory[Item] = 43; });
  Q.wait();
  for (size_t I = 0; I < N; ++I)
    Failed += Check(Memory, 43, I, "1D parallel_for shortcut");

  // 2D parallel_for shortcut
  oneapiext::parallel_for(Q, sycl::range<2>{8, N / 8}, [=](sycl::item<2> Item) {
    Memory[Item.get_linear_id()] = 44;
  });
  Q.wait();
  for (size_t I = 0; I < N; ++I)
    Failed += Check(Memory, 44, I, "2D parallel_for shortcut");

  // 3D parallel_for shortcut
  oneapiext::parallel_for(
      Q, sycl::range<3>{8, 8, N / 64},
      [=](sycl::item<3> Item) { Memory[Item.get_linear_id()] = 45; });
  Q.wait();
  for (size_t I = 0; I < N; ++I)
    Failed += Check(Memory, 45, I, "3D parallel_for shortcut");

  // 1D nd_launch shortcut
  oneapiext::nd_launch(
      Q, sycl::nd_range<1>{sycl::range<1>{N}, sycl::range{8}},
      [=](sycl::nd_item<1> Item) { Memory[Item.get_global_linear_id()] = 46; });
  Q.wait();
  for (size_t I = 0; I < N; ++I)
    Failed += Check(Memory, 46, I, "1D nd_launch shortcut");

  // 2D nd_launch shortcut
  oneapiext::nd_launch(
      Q, sycl::nd_range<2>{sycl::range<2>{8, N / 8}, sycl::range{8, 8}},
      [=](sycl::nd_item<2> Item) { Memory[Item.get_global_linear_id()] = 47; });
  Q.wait();
  for (size_t I = 0; I < N; ++I)
    Failed += Check(Memory, 47, I, "2D nd_launch shortcut");

  // 3D nd_launch shortcut
  oneapiext::nd_launch(
      Q, sycl::nd_range<3>{sycl::range<3>{8, 8, N / 64}, sycl::range{8, 8, 8}},
      [=](sycl::nd_item<3> Item) { Memory[Item.get_global_linear_id()] = 48; });
  Q.wait();
  for (size_t I = 0; I < N; ++I)
    Failed += Check(Memory, 48, I, "3D nd_launch shortcut");

  // 1D parallel_for shortcut with launch config
  oneapiext::parallel_for(
      Q, oneapiext::launch_config<sycl::range<1>>{sycl::range<1>{N}},
      [=](sycl::item<1> Item) { Memory[Item] = 49; });
  Q.wait();
  for (size_t I = 0; I < N; ++I)
    Failed +=
        Check(Memory, 49, I, "1D parallel_for shortcut with launch config");

  // 2D parallel_for shortcut with launch config
  oneapiext::parallel_for(
      Q, oneapiext::launch_config<sycl::range<2>>{sycl::range<2>{8, N / 8}},
      [=](sycl::item<2> Item) { Memory[Item.get_linear_id()] = 50; });
  Q.wait();
  for (size_t I = 0; I < N; ++I)
    Failed +=
        Check(Memory, 50, I, "2D parallel_for shortcut with launch config");

  // 3D parallel_for shortcut with launch config
  oneapiext::parallel_for(
      Q, oneapiext::launch_config<sycl::range<3>>{sycl::range<3>{8, 8, N / 64}},
      [=](sycl::item<3> Item) { Memory[Item.get_linear_id()] = 51; });
  Q.wait();
  for (size_t I = 0; I < N; ++I)
    Failed +=
        Check(Memory, 51, I, "3D parallel_for shortcut with launch config");

  // 1D nd_launch shortcut with launch config
  oneapiext::nd_launch(
      Q,
      oneapiext::launch_config<sycl::nd_range<1>>{
          sycl::nd_range<1>{sycl::range<1>{N}, sycl::range{8}}},
      [=](sycl::nd_item<1> Item) { Memory[Item.get_global_linear_id()] = 52; });
  Q.wait();
  for (size_t I = 0; I < N; ++I)
    Failed += Check(Memory, 52, I, "1D nd_launch shortcut with launch config");

  // 2D nd_launch shortcut with launch config
  oneapiext::nd_launch(
      Q,
      oneapiext::launch_config<sycl::nd_range<2>>{
          sycl::nd_range<2>{sycl::range<2>{8, N / 8}, sycl::range{8, 8}}},
      [=](sycl::nd_item<2> Item) { Memory[Item.get_global_linear_id()] = 53; });
  Q.wait();
  for (size_t I = 0; I < N; ++I)
    Failed += Check(Memory, 53, I, "2D nd_launch shortcut with launch config");

  // 3D nd_launch shortcut with launch config
  oneapiext::nd_launch(
      Q,
      oneapiext::launch_config<sycl::nd_range<3>>{sycl::nd_range<3>{
          sycl::range<3>{8, 8, N / 64}, sycl::range{8, 8, 8}}},
      [=](sycl::nd_item<3> Item) { Memory[Item.get_global_linear_id()] = 54; });
  Q.wait();
  for (size_t I = 0; I < N; ++I)
    Failed += Check(Memory, 54, I, "3D nd_launch shortcut with launch config");

  sycl::free(Memory, Q);
  return Failed;
}
