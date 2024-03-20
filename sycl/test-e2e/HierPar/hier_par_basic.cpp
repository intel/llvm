//==- hier_par_basic.cpp --- hierarchical parallelism API test -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// This test checks hierarchical parallelism invocation APIs, but without any
// data or code with side-effects between the work group and work item scopes.

#include <iostream>
#include <memory>
#include <sycl/detail/core.hpp>

using namespace sycl;

template <typename GoldFnTy>
bool verify(int testcase, int range_length, const int *ptr, GoldFnTy get_gold) {
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

struct MyStruct {
  int x;
  int y;
};

using AccTy =
    accessor<int, 1, access::mode::read_write, sycl::access::target::device>;

struct PFWIFunctor {
  PFWIFunctor(size_t wg_chunk, size_t wg_size, size_t wg_offset,
              size_t range_length, int v, AccTy &dev_ptr)
      : wg_chunk(wg_chunk), wg_size(wg_size), wg_offset(wg_offset),
        range_length(range_length), v(v), dev_ptr(dev_ptr) {}

  void operator()(h_item<1> i) const {
    // number of buf elements per work item:
    size_t wi_chunk = (wg_chunk + wg_size - 1) / wg_size;
    auto id = i.get_physical_local_id().get(0);
    if (id >= wg_chunk)
      return;
    size_t wi_offset = wg_offset + id * wi_chunk;
    size_t ub = std::min(wi_offset + wi_chunk, range_length);

    for (size_t ind = wi_offset; ind < ub; ind++)
      dev_ptr[ind] += v;
  }

  size_t wg_chunk;
  size_t wg_size;
  size_t wg_offset;
  size_t range_length;
  int v;
  AccTy &dev_ptr;
};

struct PFWGFunctor {
  PFWGFunctor(size_t wg_chunk, size_t range_length, int addend, int n_iter,
              AccTy &dev_ptr)
      : wg_chunk(wg_chunk), range_length(range_length), dev_ptr(dev_ptr),
        addend(addend), n_iter(n_iter) {}

  void operator()(group<1> g) const {
    int v = addend; // to check constant initializer works too
    size_t wg_offset = wg_chunk * g.get_group_id(0);
    size_t wg_size = g.get_local_range(0);

    PFWIFunctor PFWI(wg_chunk, wg_size, wg_offset, range_length, v, dev_ptr);

    for (int cnt = 0; cnt < n_iter; cnt++) {
      g.parallel_for_work_item(PFWI);
    }
  }
  // Dummy operator '()' to make sure compiler can handle multiple '()'
  // operators/ and pick the right one for PFWG kernel code generation.
  void operator()(int ind, int val) const { dev_ptr[ind] += val; }

  const size_t wg_chunk;
  const size_t range_length;
  const int n_iter;
  const int addend;
  mutable AccTy dev_ptr;
};

int main() {
  constexpr int N_WG = 7;
  constexpr int WG_SIZE_PHYSICAL = 3;
  constexpr int WG_SIZE_GREATER_THAN_PHYSICAL = 5;
  constexpr int WG_SIZE_LESS_THAN_PHYSICAL = 2;
  constexpr int N_ITER = 2;

  constexpr size_t range_length = N_WG * WG_SIZE_PHYSICAL;
  std::unique_ptr<int[]> data(new int[range_length]);
  int *ptr = data.get();
  bool passed = true;

  try {
    queue myQueue;

    std::cout << "Running on "
              << myQueue.get_device().get_info<sycl::info::device::name>()
              << "\n";
    {
      // Testcase1
      // - PFWG kernel and PFWI function are represented as functor objects
      // - PFWG functor contains extra dummy '()' operator
      // - handler::parallel_for_work_group w/o local size specification +
      //   group::parallel_for_work_item w/o flexible range
      // - h_item::get_global_id
      // The global size is not known, so kernel code needs to adapt
      // dynamically, hence the complex loop bound and index computation.
      std::memset(ptr, 0, range_length * sizeof(ptr[0]));
      buffer<int, 1> buf(ptr, range<1>(range_length));
      const int addend = 10;

      myQueue.submit([&](handler &cgh) {
        auto dev_ptr = buf.get_access<access::mode::read_write>(cgh);
        // number of 'buf' elements per work group:
        size_t wg_chunk = (range_length + N_WG - 1) / N_WG;
        PFWGFunctor PFWG(wg_chunk, range_length, addend, N_ITER, dev_ptr);
        cgh.parallel_for_work_group(range<1>(N_WG), PFWG);
      });
      host_accessor hostacc(buf, read_only);
      auto ptr1 = hostacc.get_pointer();
      passed &= verify(1, range_length, ptr1,
                       [&](int i) -> int { return N_ITER * addend; });
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
      host_accessor hostacc(buf, read_only);
      auto ptr1 = hostacc.get_pointer();
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
      host_accessor hostacc(buf, read_only);
      auto ptr1 = hostacc.get_pointer();
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
    {
      // Testcase4
      // - private_memory<int> private variable lives across two PFWI scopes -
      //   initialized in the first one and used in the second one
      const int WG_X_SIZE = 7;
      const int WG_Y_SIZE = 3;
      const int WG_LINEAR_SIZE = WG_X_SIZE * WG_Y_SIZE;
      const int range_length = N_WG * WG_LINEAR_SIZE;

      std::unique_ptr<int[]> data(new int[range_length]);
      int *ptr = data.get();

      std::memset(ptr, 0, range_length * sizeof(ptr[0]));
      buffer<int, 1> buf(ptr, range<1>(range_length));
      myQueue.submit([&](handler &cgh) {
        auto dev_ptr = buf.get_access<access::mode::read_write>(cgh);

        cgh.parallel_for_work_group<class hpar_priv_mem>(
            range<2>(N_WG, 1), range<2>(WG_X_SIZE, WG_Y_SIZE), [=](group<2> g) {
              private_memory<MyStruct, 2> priv(g);

              for (int cnt = 0; cnt < N_ITER; cnt++) {
                g.parallel_for_work_item(
                    range<2>(WG_X_SIZE, WG_Y_SIZE), [&](h_item<2> i) {
                      auto glob_id = i.get_global().get_linear_id();
                      dev_ptr[glob_id]++;
                      MyStruct &s = priv(i);
                      s.x = glob_id;
                      s.y = 5;
                    });
                g.parallel_for_work_item(
                    range<2>(WG_X_SIZE, WG_Y_SIZE), [&](h_item<2> i) {
                      const MyStruct &s = priv(i);
                      dev_ptr[i.get_global().get_linear_id()] += (s.x + s.y);
                    });
              }
            });
      });
      host_accessor hostacc(buf, read_only);
      auto ptr1 = hostacc.get_pointer();
      passed &= verify(3, range_length, ptr1,
                       [&](int i) -> int { return N_ITER * (1 + i + 5); });
    }
    {
      // Testcase5
      // - flexible range different from the physical one is used,
      // get_logical_local_range and get_physical_local_range apis are tested
      const int wi_chunk = 2;
      constexpr size_t range_length =
          N_WG * WG_SIZE_GREATER_THAN_PHYSICAL * wi_chunk;
      std::unique_ptr<int[]> data(new int[range_length]);
      int *ptr = data.get();

      std::memset(ptr, 0, range_length * sizeof(ptr[0]));
      buffer<int, 1> buf(ptr, range<1>(range_length));
      myQueue.submit([&](handler &cgh) {
        auto dev_ptr = buf.get_access<access::mode::read_write>(cgh);

        cgh.parallel_for_work_group<class hpar_ranges>(
            range<1>(N_WG), range<1>(WG_SIZE_PHYSICAL), [=](group<1> g) {
              for (int cnt = 0; cnt < N_ITER; cnt++) {
                g.parallel_for_work_item(
                    range<1>(WG_SIZE_GREATER_THAN_PHYSICAL), [&](h_item<1> i) {
                      size_t wg_offset = WG_SIZE_GREATER_THAN_PHYSICAL *
                                         wi_chunk * g.get_group_id(0);
                      size_t wi_offset =
                          wg_offset +
                          i.get_logical_local_id().get(0) * wi_chunk;
                      dev_ptr[wi_offset + 0] += i.get_logical_local_range()[0];
                      dev_ptr[wi_offset + 1] += i.get_physical_local_range()[0];
                    });
              }
            });
      });
      host_accessor hostacc(buf, read_only);
      auto ptr1 = hostacc.get_pointer();
      passed &= verify(5, range_length, ptr1, [&](int i) -> int {
        int gold =
            i % 2 == 0 ? WG_SIZE_GREATER_THAN_PHYSICAL : WG_SIZE_PHYSICAL;
        gold *= N_ITER;
        return gold;
      });
    }
  } catch (sycl::exception const &e) {
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
