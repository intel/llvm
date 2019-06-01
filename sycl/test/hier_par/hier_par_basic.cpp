//==- hier_par_basic.cpp --- hierarchical parallelism API test -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %clang -std=c++11 -fsycl %s -o %t.out -lOpenCL -lstdc++
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// This test checks hierarchical parallelism invocation APIs, but without any
// data or code with side-effects between the work group and work item scopes.

#include <CL/sycl.hpp>
#include <iostream>
#include <memory>

using namespace cl::sycl;

template <typename GoldFnTy>
bool verify(int testcase, int range_length, int *ptr, GoldFnTy get_gold) {
  int err_cnt = 0;

  for (int i = 0; i < range_length; i++) {
    int gold = get_gold(i);

    if (ptr[i] != gold) {
      if (++err_cnt < 20) {
        std::cout << testcase << " - ERROR at " << i << ": " << ptr[i]
                  << " != " << gold << "(expected)\n";
      }
    }
  }
  if (err_cnt > 0)
    std::cout << "-- Failure rate: " << err_cnt << "/" << range_length << "("
              << err_cnt / (float)range_length * 100.f << "%)\n";
  return err_cnt == 0;
}

int main() {
  constexpr int N_WG = 7;
  constexpr int WG_SIZE_PHYSICAL = 3;
  constexpr int WG_SIZE_GREATER_THAN_PHYSICAL = 5;
  constexpr int WG_SIZE_LESS_THAN_PHYSICAL = 2;
  constexpr int N_ITER = 2;

  constexpr size_t range_length = N_WG * WG_SIZE_PHYSICAL;
  std::unique_ptr<int> data(new int[range_length]);
  int *ptr = data.get();
  bool passed = true;

  try {
    queue myQueue;

    std::cout << "Running on "
              << myQueue.get_device().get_info<cl::sycl::info::device::name>()
              << "\n";
    {
      // Testcase1
      // - handler::parallel_for_work_group w/o local size specification +
      //   group::parallel_for_work_item w/o flexible range
      // - h_item::get_global_id
      // The global size is not known, so kernel code needs to adapt
      // dynamically, hence the complex loop bound and index computation.
      std::memset(ptr, 0, range_length * sizeof(ptr[0]));
      buffer<int, 1> buf(ptr, range<1>(range_length));
      myQueue.submit([&](handler &cgh) {
        auto dev_ptr = buf.get_access<access::mode::read_write>(cgh);
        // number of 'buf' elements per work group:
        size_t wg_chunk = (range_length + N_WG - 1) / N_WG;

        cgh.parallel_for_work_group<class hpar_simple>(
            range<1>(N_WG), [=](group<1> g) {
              size_t wg_offset = wg_chunk * g.get_id(0);
              size_t wg_size = g.get_local_range(0);

              for (int cnt = 0; cnt < N_ITER; cnt++) {
                g.parallel_for_work_item([&](h_item<1> i) {
                  // number of buf elements per work item:
                  size_t wi_chunk = (wg_chunk + wg_size - 1) / wg_size;
                  auto id = i.get_physical_local_id().get(0);
                  if (id >= wg_chunk)
                    return;
                  size_t wi_offset = wg_offset + id * wi_chunk;
                  size_t ub = cl::sycl::min(wi_offset + wi_chunk, range_length);

                  for (size_t ind = wi_offset; ind < ub; ind++)
                    dev_ptr[ind]++;
                });
              }
            });
      });
      auto ptr1 = buf.get_access<access::mode::read>().get_pointer();
      passed &=
          verify(1, range_length, ptr1, [&](int i) -> int { return N_ITER; });
    }

    {
      // Testcase2
      // - handler::parallel_for_work_group with local size specification +
      //   group::parallel_for_work_item with flexible range
      // - h_item::get_global_id
      std::memset(ptr, 0, range_length * sizeof(ptr[0]));
      buffer<int, 1> buf(ptr, range<1>(range_length));
      myQueue.submit([&](handler &cgh) {
        auto dev_ptr = buf.get_access<access::mode::read_write>(cgh);

        cgh.parallel_for_work_group<class hpar_flex>(
            range<1>(N_WG), range<1>(WG_SIZE_PHYSICAL), [=](group<1> g) {
              for (int cnt = 0; cnt < N_ITER; cnt++) {
                g.parallel_for_work_item(
                    range<1>(WG_SIZE_GREATER_THAN_PHYSICAL),
                    [&](h_item<1> i) { dev_ptr[i.get_global_id(0)]++; });
                g.parallel_for_work_item(
                    range<1>(WG_SIZE_LESS_THAN_PHYSICAL),
                    [&](h_item<1> i) { dev_ptr[i.get_global_id(0)]++; });
              }
            });
      });
      auto ptr1 = buf.get_access<access::mode::read>().get_pointer();
      passed &= verify(2, range_length, ptr1, [&](int i) -> int {
        // consider increments by the first PFWI:
        int gold = (WG_SIZE_GREATER_THAN_PHYSICAL - 1) / WG_SIZE_PHYSICAL;
        if (i % WG_SIZE_PHYSICAL <
            WG_SIZE_GREATER_THAN_PHYSICAL % WG_SIZE_PHYSICAL) {
          gold++;
        }
        // consider increments by the second PFWI:
        if (i % WG_SIZE_PHYSICAL < WG_SIZE_LESS_THAN_PHYSICAL) {
          gold++;
        }
        gold *= N_ITER;
        return gold;
      });
    }

    {
      // Testcase3
      // - handler::parallel_for_work_group with local size specification +
      //   group::parallel_for_work_item with flexible range
      // - h_item::get_logical_local_id,get_physical_local_id
      std::memset(ptr, 0, range_length * sizeof(ptr[0]));
      buffer<int, 1> buf(ptr, range<1>(range_length));
      myQueue.submit([&](handler &cgh) {
        auto dev_ptr = buf.get_access<access::mode::read_write>(cgh);

        cgh.parallel_for_work_group<class hpar_hitem>(
            range<1>(N_WG), range<1>(WG_SIZE_PHYSICAL), [=](group<1> g) {
              for (int cnt = 0; cnt < N_ITER; cnt++) {
                g.parallel_for_work_item(
                    range<1>(WG_SIZE_GREATER_THAN_PHYSICAL), [&](h_item<1> i) {
                      int n =
                          i.get_logical_local_id() == i.get_physical_local_id()
                              ? 0
                              : 1;
                      dev_ptr[i.get_global_id(0)] += n;
                    });
              }
            });
      });
      auto ptr1 = buf.get_access<access::mode::read>().get_pointer();
      passed &= verify(3, range_length, ptr1, [&](int i) -> int {
        int gold = 0;
        if (i % WG_SIZE_PHYSICAL <
            WG_SIZE_GREATER_THAN_PHYSICAL % WG_SIZE_PHYSICAL) {
          gold++;
        }
        gold *= N_ITER;
        return gold;
      });
    }
  } catch (cl::sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    return 2;
  }

  if (passed) {
    std::cout << "Passed\n";
    return 0;
  }
  std::cout << "FAILED\n";
  return 1;
}
